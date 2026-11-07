//! End-to-End Pure-Rust Int8 Transformer Encoder for `snowflake-arctic-embed-s`.
//!
//! Architectural highlights:
//! - Zero heap allocations during `embed_tokens_into` via reusable `Workspace`
//! - Fused `LayerNorm + SmoothQuant + Dynamic Int8 Quantization` across all 12 layers
//! - Fused `GELU + SmoothQuant + Dynamic Int8 Quantization` for `[M, 1536]` FFN activations
//! - Fused residual connection inside Int8 GEMM register writeback
//! - Exact Layer-11 CLS-Token Pruning (`M = 1` for Q, Attention, AttnOut, LN1, FFN-Up, GELU, FFN-Down, LN2)

use crate::aligned::{AlignedBuf, MmapRegion};
use crate::gemm::{gemm_int8_fused, PackedLinearInt8, SimdBackend};
use crate::quantize::{
    QuantizedModelRaw, HEAD_DIM, HIDDEN_SIZE, INTERMEDIATE_SIZE, LAYER_NORM_EPS, MAX_POSITIONS,
    NUM_HEADS, NUM_LAYERS,
};
use crate::simd_math::{
    compute_scores_head32, dequant_add_embeddings_row, extract_kv_head32,
    gelu_smooth_quantize_batch, l2_normalize_inplace, layernorm_inplace_row,
    layernorm_smooth_quantize_row, smooth_quantize_row, softmax_inplace, weighted_sum_head32,
};
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

const AE8P_MAGIC: &[u8; 8] = b"ARCTIC8P";
const AE8P_VERSION: u8 = 1;

pub struct PackedLayer {
    pub qkv: PackedLinearInt8,
    pub q_only: Option<PackedLinearInt8>,
    pub kv_only: Option<PackedLinearInt8>,
    pub attn_out: PackedLinearInt8,
    pub ln1_weight: AlignedBuf<f32>,
    pub ln1_bias: AlignedBuf<f32>,
    pub ffn_inter: PackedLinearInt8,
    pub ffn_out: PackedLinearInt8,
    pub ln2_weight: AlignedBuf<f32>,
    pub ln2_bias: AlignedBuf<f32>,
}

pub struct PackedModel {
    pub backend: SimdBackend,
    pub word_emb_q: AlignedBuf<i8>,
    pub word_emb_scale: AlignedBuf<f32>,
    pub pos_emb_q: AlignedBuf<i8>,
    pub pos_emb_scale: AlignedBuf<f32>,
    /// Pre-dequantized `token_type_embeddings[0]` (`[384]`), added to every token.
    pub type0_emb_f32: AlignedBuf<f32>,
    pub emb_ln_weight: AlignedBuf<f32>,
    pub emb_ln_bias: AlignedBuf<f32>,
    pub layers: Vec<PackedLayer>,
}

impl PackedModel {
    pub fn from_raw(raw: &QuantizedModelRaw, backend: SimdBackend) -> Self {
        let mut type0_emb_f32 = AlignedBuf::<f32>::new_zeroed(HIDDEN_SIZE);
        let t_scale = raw.type_emb_scale[0];
        for j in 0..HIDDEN_SIZE {
            type0_emb_f32[j] = (raw.type_emb_q[j] as f32) * t_scale;
        }

        let layers = raw
            .layers
            .iter()
            .map(|l| PackedLayer {
                qkv: l.qkv.pack(backend),
                q_only: l.q_only.as_ref().map(|q| q.pack(backend)),
                kv_only: l.kv_only.as_ref().map(|kv| kv.pack(backend)),
                attn_out: l.attn_out.pack(backend),
                ln1_weight: AlignedBuf::from_slice(&l.ln1_weight),
                ln1_bias: AlignedBuf::from_slice(&l.ln1_bias),
                ffn_inter: l.ffn_inter.pack(backend),
                ffn_out: l.ffn_out.pack(backend),
                ln2_weight: AlignedBuf::from_slice(&l.ln2_weight),
                ln2_bias: AlignedBuf::from_slice(&l.ln2_bias),
            })
            .collect();

        Self {
            backend,
            word_emb_q: AlignedBuf::from_slice(&raw.word_emb_q),
            word_emb_scale: AlignedBuf::from_slice(&raw.word_emb_scale),
            pos_emb_q: AlignedBuf::from_slice(&raw.pos_emb_q),
            pos_emb_scale: AlignedBuf::from_slice(&raw.pos_emb_scale),
            type0_emb_f32,
            emb_ln_weight: AlignedBuf::from_slice(&raw.emb_ln_weight),
            emb_ln_bias: AlignedBuf::from_slice(&raw.emb_ln_bias),
            layers,
        }
    }

    /// Total memory footprint of packed weights in bytes.
    pub fn size_bytes(&self) -> usize {
        let mut total = self.word_emb_q.len()
            + self.pos_emb_q.len()
            + (self.word_emb_scale.len()
                + self.pos_emb_scale.len()
                + self.type0_emb_f32.len()
                + self.emb_ln_weight.len()
                + self.emb_ln_bias.len())
                * 4;
        for l in &self.layers {
            for lin in [&l.qkv, &l.attn_out, &l.ffn_inter, &l.ffn_out] {
                total += lin.packed_w.len()
                    + (lin.w_scale.len()
                        + lin.bias.len()
                        + lin.inv_smooth.len()
                        + lin.col_sum_comp.len())
                        * 4;
            }
            if let (Some(q), Some(kv)) = (&l.q_only, &l.kv_only) {
                for lin in [q, kv] {
                    total += lin.packed_w.len()
                        + (lin.w_scale.len()
                            + lin.bias.len()
                            + lin.inv_smooth.len()
                            + lin.col_sum_comp.len())
                            * 4;
                }
            }
            total +=
                (l.ln1_weight.len() + l.ln1_bias.len() + l.ln2_weight.len() + l.ln2_bias.len()) * 4;
        }
        total
    }

    /// Serializes the pre-packed model into a 64-byte-aligned `.ae8p` file for zero-copy `mmap` loading.
    pub fn save_ae8p<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut f = io::BufWriter::new(File::create(path)?);
        let mut header = [0u8; 64];
        header[0..8].copy_from_slice(AE8P_MAGIC);
        header[8] = AE8P_VERSION;
        header[9] = self.backend.to_u8();
        f.write_all(&header)?;

        write_aligned_slice(&mut f, self.word_emb_q.as_slice())?;
        write_aligned_slice(&mut f, self.word_emb_scale.as_slice())?;
        write_aligned_slice(&mut f, self.pos_emb_q.as_slice())?;
        write_aligned_slice(&mut f, self.pos_emb_scale.as_slice())?;
        write_aligned_slice(&mut f, self.type0_emb_f32.as_slice())?;
        write_aligned_slice(&mut f, self.emb_ln_weight.as_slice())?;
        write_aligned_slice(&mut f, self.emb_ln_bias.as_slice())?;

        for l in &self.layers {
            write_packed_linear(&mut f, &l.qkv)?;
            let mut split_hdr = [0u8; 64];
            if let (Some(q), Some(kv)) = (&l.q_only, &l.kv_only) {
                split_hdr[0] = 1;
                f.write_all(&split_hdr)?;
                write_packed_linear(&mut f, q)?;
                write_packed_linear(&mut f, kv)?;
            } else {
                f.write_all(&split_hdr)?;
            }
            write_packed_linear(&mut f, &l.attn_out)?;
            write_aligned_slice(&mut f, l.ln1_weight.as_slice())?;
            write_aligned_slice(&mut f, l.ln1_bias.as_slice())?;
            write_packed_linear(&mut f, &l.ffn_inter)?;
            write_packed_linear(&mut f, &l.ffn_out)?;
            write_aligned_slice(&mut f, l.ln2_weight.as_slice())?;
            write_aligned_slice(&mut f, l.ln2_bias.as_slice())?;
        }
        f.flush()
    }

    /// Maps a 64-byte-aligned `.ae8p` pre-packed model directly into memory using `mmap(2)` with zero copies
    /// and zero heap allocations for weight tensors.
    pub fn load_ae8p_mmap<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = MmapRegion::from_file(&file)?;
        let bytes = mmap.as_bytes();
        if bytes.len() < 64 || &bytes[0..8] != AE8P_MAGIC || bytes[8] != AE8P_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid .ae8p magic or version header",
            ));
        }
        let backend = SimdBackend::from_u8(bytes[9]).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "unknown backend ID in .ae8p")
        })?;
        let mut cur = 64usize;

        let word_emb_q = mmap_aligned_slice::<i8>(&mmap, &mut cur)?;
        let word_emb_scale = mmap_aligned_slice::<f32>(&mmap, &mut cur)?;
        let pos_emb_q = mmap_aligned_slice::<i8>(&mmap, &mut cur)?;
        let pos_emb_scale = mmap_aligned_slice::<f32>(&mmap, &mut cur)?;
        let type0_emb_f32 = mmap_aligned_slice::<f32>(&mmap, &mut cur)?;
        let emb_ln_weight = mmap_aligned_slice::<f32>(&mmap, &mut cur)?;
        let emb_ln_bias = mmap_aligned_slice::<f32>(&mmap, &mut cur)?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for _ in 0..NUM_LAYERS {
            let qkv = mmap_packed_linear(&mmap, &mut cur)?;
            if cur + 64 > bytes.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "truncated .ae8p",
                ));
            }
            let has_split = bytes[cur];
            cur += 64;
            let (q_only, kv_only) = if has_split == 1 {
                (
                    Some(mmap_packed_linear(&mmap, &mut cur)?),
                    Some(mmap_packed_linear(&mmap, &mut cur)?),
                )
            } else {
                (None, None)
            };
            let attn_out = mmap_packed_linear(&mmap, &mut cur)?;
            let ln1_weight = mmap_aligned_slice::<f32>(&mmap, &mut cur)?;
            let ln1_bias = mmap_aligned_slice::<f32>(&mmap, &mut cur)?;
            let ffn_inter = mmap_packed_linear(&mmap, &mut cur)?;
            let ffn_out = mmap_packed_linear(&mmap, &mut cur)?;
            let ln2_weight = mmap_aligned_slice::<f32>(&mmap, &mut cur)?;
            let ln2_bias = mmap_aligned_slice::<f32>(&mmap, &mut cur)?;

            layers.push(PackedLayer {
                qkv,
                q_only,
                kv_only,
                attn_out,
                ln1_weight,
                ln1_bias,
                ffn_inter,
                ffn_out,
                ln2_weight,
                ln2_bias,
            });
        }

        Ok(Self {
            backend,
            word_emb_q,
            word_emb_scale,
            pos_emb_q,
            pos_emb_scale,
            type0_emb_f32,
            emb_ln_weight,
            emb_ln_bias,
            layers,
        })
    }
}

fn write_aligned_slice<W: Write, T: Copy>(w: &mut W, slice: &[T]) -> io::Result<()> {
    let mut hdr = [0u8; 64];
    hdr[0..8].copy_from_slice(&(slice.len() as u64).to_le_bytes());
    w.write_all(&hdr)?;
    let byte_len = std::mem::size_of_val(slice);
    if byte_len > 0 {
        let raw_bytes =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_len) };
        w.write_all(raw_bytes)?;
        let rem = byte_len % 64;
        if rem != 0 {
            let pad = [0u8; 64];
            w.write_all(&pad[..64 - rem])?;
        }
    }
    Ok(())
}

fn write_packed_linear<W: Write>(w: &mut W, lin: &PackedLinearInt8) -> io::Result<()> {
    let mut hdr = [0u8; 64];
    hdr[0..8].copy_from_slice(&(lin.n_out as u64).to_le_bytes());
    hdr[8..16].copy_from_slice(&(lin.k_in as u64).to_le_bytes());
    hdr[16] = lin.backend.to_u8();
    w.write_all(&hdr)?;
    write_aligned_slice(w, lin.packed_w.as_slice())?;
    write_aligned_slice(w, lin.w_scale.as_slice())?;
    write_aligned_slice(w, lin.bias.as_slice())?;
    write_aligned_slice(w, lin.inv_smooth.as_slice())?;
    write_aligned_slice(w, lin.col_sum_comp.as_slice())
}

fn mmap_aligned_slice<T: Copy>(
    mmap: &Arc<MmapRegion>,
    cur: &mut usize,
) -> io::Result<AlignedBuf<T>> {
    let bytes = mmap.as_bytes();
    if *cur + 64 > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "truncated .ae8p tensor header",
        ));
    }
    let len = u64::from_le_bytes(bytes[*cur..*cur + 8].try_into().unwrap()) as usize;
    *cur += 64;
    let buf = AlignedBuf::from_mmap(Arc::clone(mmap), *cur, len)?;
    let byte_len = len * std::mem::size_of::<T>();
    *cur += byte_len.next_multiple_of(64);
    Ok(buf)
}

fn mmap_packed_linear(mmap: &Arc<MmapRegion>, cur: &mut usize) -> io::Result<PackedLinearInt8> {
    let bytes = mmap.as_bytes();
    if *cur + 64 > bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "truncated .ae8p linear header",
        ));
    }
    let n_out = u64::from_le_bytes(bytes[*cur..*cur + 8].try_into().unwrap()) as usize;
    let k_in = u64::from_le_bytes(bytes[*cur + 8..*cur + 16].try_into().unwrap()) as usize;
    let backend = SimdBackend::from_u8(bytes[*cur + 16]).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid backend ID in linear header",
        )
    })?;
    *cur += 64;

    let packed_w = mmap_aligned_slice::<i8>(mmap, cur)?;
    let w_scale = mmap_aligned_slice::<f32>(mmap, cur)?;
    let bias = mmap_aligned_slice::<f32>(mmap, cur)?;
    let inv_smooth = mmap_aligned_slice::<f32>(mmap, cur)?;
    let col_sum_comp = mmap_aligned_slice::<i32>(mmap, cur)?;

    Ok(PackedLinearInt8 {
        n_out,
        k_in,
        backend,
        packed_w,
        w_scale,
        bias,
        inv_smooth,
        col_sum_comp,
    })
}

/// Pre-allocated scratchpad for zero-allocation inference up to `MAX_POSITIONS = 512`.
pub struct Workspace {
    pub hidden: AlignedBuf<f32>,
    pub tmp_hidden: AlignedBuf<f32>,
    pub act_q: AlignedBuf<i8>,
    pub act_packed: AlignedBuf<i8>,
    pub act_scale: AlignedBuf<f32>,
    pub qkv_buf: AlignedBuf<f32>,
    pub q_cls: AlignedBuf<f32>,
    pub attn_ctx: AlignedBuf<f32>,
    pub k_head: AlignedBuf<f32>,
    pub v_head: AlignedBuf<f32>,
    pub scores: AlignedBuf<f32>,
    pub sme_q_t: AlignedBuf<f32>,
    pub sme_k_t: AlignedBuf<f32>,
    pub sme_scores_blk: AlignedBuf<f32>,
    pub sme_p_t: AlignedBuf<f32>,
    pub inter_buf: AlignedBuf<f32>,
    pub inter_q: AlignedBuf<i8>,
    pub inter_packed: AlignedBuf<i8>,
    pub token_ids: Vec<u16>,
    pub seg_ranges: Vec<(usize, usize)>,
}

impl Workspace {
    pub fn new() -> Self {
        let m = MAX_POSITIONS;
        let m_pad = m + 64;
        Self {
            hidden: AlignedBuf::new_zeroed(m * HIDDEN_SIZE),
            tmp_hidden: AlignedBuf::new_zeroed(m * HIDDEN_SIZE),
            act_q: AlignedBuf::new_zeroed(m * HIDDEN_SIZE),
            act_packed: AlignedBuf::new_zeroed((m + 16) * HIDDEN_SIZE),
            act_scale: AlignedBuf::new_zeroed(m),
            qkv_buf: AlignedBuf::new_zeroed(m * 3 * HIDDEN_SIZE),
            q_cls: AlignedBuf::new_zeroed(m * HIDDEN_SIZE),
            attn_ctx: AlignedBuf::new_zeroed(m * HIDDEN_SIZE),
            k_head: AlignedBuf::new_zeroed(m_pad * HEAD_DIM),
            v_head: AlignedBuf::new_zeroed(m_pad * HEAD_DIM),
            scores: AlignedBuf::new_zeroed(m_pad),
            sme_q_t: AlignedBuf::new_zeroed(m_pad * HEAD_DIM),
            sme_k_t: AlignedBuf::new_zeroed(m_pad * HEAD_DIM),
            sme_scores_blk: AlignedBuf::new_zeroed(16 * m_pad),
            sme_p_t: AlignedBuf::new_zeroed(16 * m_pad),
            inter_buf: AlignedBuf::new_zeroed(m * INTERMEDIATE_SIZE),
            inter_q: AlignedBuf::new_zeroed(m * INTERMEDIATE_SIZE),
            inter_packed: AlignedBuf::new_zeroed((m + 16) * INTERMEDIATE_SIZE),
            token_ids: Vec::with_capacity(m),
            seg_ranges: Vec::with_capacity(m),
        }
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

impl PackedModel {
    /// Runs the 12-layer Int8 transformer encoder on `input_ids` and writes the
    /// L2-normalized 384-dim `[CLS]` embedding into `out_emb`.
    /// Performs zero heap allocations.
    pub fn forward_cls(
        &self,
        input_ids: &[u16],
        ws: &mut Workspace,
        out_emb: &mut [f32; HIDDEN_SIZE],
    ) {
        let seq_len = input_ids.len().min(MAX_POSITIONS);
        assert!(seq_len >= 1, "input_ids must not be empty");

        // 1. Dequantize Int8 word + position + token_type embeddings & fuse initial LayerNorm + SmoothQuant + Quantize
        let first_inv_smooth = &self.layers[0].qkv.inv_smooth;
        for (t, &tok) in input_ids[..seq_len].iter().enumerate() {
            let tok_idx = tok as usize;
            let w_off = tok_idx * HIDDEN_SIZE;
            let p_off = t * HIDDEN_SIZE;
            let w_s = self.word_emb_scale[tok_idx];
            let p_s = self.pos_emb_scale[t];

            let h_row = &mut ws.hidden[p_off..p_off + HIDDEN_SIZE];
            let w_row = &self.word_emb_q[w_off..w_off + HIDDEN_SIZE];
            let p_row = &self.pos_emb_q[p_off..p_off + HIDDEN_SIZE];
            let t_row = &self.type0_emb_f32[..HIDDEN_SIZE];

            dequant_add_embeddings_row(w_row, w_s, p_row, p_s, t_row, h_row);

            let q_row = &mut ws.act_q[p_off..p_off + HIDDEN_SIZE];
            ws.act_scale[t] = layernorm_smooth_quantize_row(
                h_row,
                &self.emb_ln_weight,
                &self.emb_ln_bias,
                first_inv_smooth,
                q_row,
                LAYER_NORM_EPS,
            );
        }

        // 2. Layers 0..10 (Full sequence length `seq_len`)
        let inv_sqrt_d = 1.0 / (HEAD_DIM as f32).sqrt();
        let use_sme = self.backend == SimdBackend::ArmSme;
        for l_idx in 0..(NUM_LAYERS - 1) {
            let layer = &self.layers[l_idx];
            let next_qkv_smooth = &self.layers[l_idx + 1].qkv.inv_smooth;

            // 2a. Fused QKV Int8 GEMM: [seq_len, 384] x [384, 1152] -> [seq_len, 1152]
            gemm_int8_fused(
                seq_len,
                &ws.act_q[..seq_len * HIDDEN_SIZE],
                &mut ws.act_packed,
                &ws.act_scale[..seq_len],
                &layer.qkv,
                None,
                &mut ws.qkv_buf[..seq_len * 3 * HIDDEN_SIZE],
            );

            // 2b. Multi-Head Self-Attention (12 heads, head_dim = 32)
            // For `seq_len >= 16`, use L1-tiled SME `smopa` attention (int8 `8` K-steps for Q@K^T,
            // 2-pass exp2 int16 Softmax, `seq_len/2` int16 K-steps for P@V); measured crossover on M4 is ~16 tokens.
            // For `seq_len < 16`, use register-resident per-core NEON/AVX2 attention (`compute_scores_head32` +
            // `weighted_sum_head32`), avoiding `smstart` streaming transitions and cluster SME contention.
            #[cfg(target_arch = "aarch64")]
            if use_sme && seq_len >= 16 {
                unsafe {
                    crate::simd_math::attention_multihead_sme(
                        seq_len,
                        &ws.qkv_buf[..seq_len * 3 * HIDDEN_SIZE],
                        &mut ws.sme_q_t,
                        &mut ws.sme_k_t,
                        &mut ws.v_head,
                        &mut ws.sme_scores_blk,
                        &mut ws.sme_p_t,
                        &mut ws.attn_ctx[..seq_len * HIDDEN_SIZE],
                        None,
                    );
                }
            } else {
                for h in 0..NUM_HEADS {
                    let h_off = h * HEAD_DIM;
                    extract_kv_head32(
                        &ws.qkv_buf,
                        seq_len,
                        3 * HIDDEN_SIZE,
                        HIDDEN_SIZE + h_off,
                        2 * HIDDEN_SIZE + h_off,
                        &mut ws.k_head,
                        &mut ws.v_head,
                    );

                    for tq in 0..seq_len {
                        let q_slice = &ws.qkv_buf
                            [tq * 3 * HIDDEN_SIZE + h_off..tq * 3 * HIDDEN_SIZE + h_off + HEAD_DIM];
                        let scores_slice = &mut ws.scores[..seq_len];
                        compute_scores_head32(
                            q_slice,
                            &ws.k_head[..seq_len * HEAD_DIM],
                            inv_sqrt_d,
                            scores_slice,
                        );
                        softmax_inplace(scores_slice);
                        let out_head = &mut ws.attn_ctx
                            [tq * HIDDEN_SIZE + h_off..tq * HIDDEN_SIZE + h_off + HEAD_DIM];
                        weighted_sum_head32(
                            scores_slice,
                            &ws.v_head[..seq_len * HEAD_DIM],
                            out_head,
                        );
                    }
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for h in 0..NUM_HEADS {
                let h_off = h * HEAD_DIM;
                for t in 0..seq_len {
                    let qkv_row = &ws.qkv_buf[t * 3 * HIDDEN_SIZE..(t + 1) * 3 * HIDDEN_SIZE];
                    ws.k_head[t * HEAD_DIM..(t + 1) * HEAD_DIM].copy_from_slice(
                        &qkv_row[HIDDEN_SIZE + h_off..HIDDEN_SIZE + h_off + HEAD_DIM],
                    );
                    ws.v_head[t * HEAD_DIM..(t + 1) * HEAD_DIM].copy_from_slice(
                        &qkv_row[2 * HIDDEN_SIZE + h_off..2 * HIDDEN_SIZE + h_off + HEAD_DIM],
                    );
                }

                for tq in 0..seq_len {
                    let q_slice = &ws.qkv_buf
                        [tq * 3 * HIDDEN_SIZE + h_off..tq * 3 * HIDDEN_SIZE + h_off + HEAD_DIM];
                    let scores_slice = &mut ws.scores[..seq_len];
                    compute_scores_head32(
                        q_slice,
                        &ws.k_head[..seq_len * HEAD_DIM],
                        inv_sqrt_d,
                        scores_slice,
                    );
                    softmax_inplace(scores_slice);
                    let out_head = &mut ws.attn_ctx
                        [tq * HIDDEN_SIZE + h_off..tq * HIDDEN_SIZE + h_off + HEAD_DIM];
                    weighted_sum_head32(scores_slice, &ws.v_head[..seq_len * HEAD_DIM], out_head);
                }
            }

            // 2c. Quantize attention context + Fused AttnOut GEMM + Residual
            for t in 0..seq_len {
                let ctx_row = &ws.attn_ctx[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                let q_row = &mut ws.act_q[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                ws.act_scale[t] = smooth_quantize_row(ctx_row, &layer.attn_out.inv_smooth, q_row);
            }

            gemm_int8_fused(
                seq_len,
                &ws.act_q[..seq_len * HIDDEN_SIZE],
                &mut ws.act_packed,
                &ws.act_scale[..seq_len],
                &layer.attn_out,
                Some(&ws.hidden[..seq_len * HIDDEN_SIZE]),
                &mut ws.tmp_hidden[..seq_len * HIDDEN_SIZE],
            );
            std::mem::swap(&mut ws.hidden, &mut ws.tmp_hidden);

            // 2d. Fused LN1 + SmoothQuant + Quantize -> FFN Intermediate GEMM ([seq_len, 1536])
            for t in 0..seq_len {
                let h_row = &mut ws.hidden[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                let q_row = &mut ws.act_q[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                ws.act_scale[t] = layernorm_smooth_quantize_row(
                    h_row,
                    &layer.ln1_weight,
                    &layer.ln1_bias,
                    &layer.ffn_inter.inv_smooth,
                    q_row,
                    LAYER_NORM_EPS,
                );
            }

            gemm_int8_fused(
                seq_len,
                &ws.act_q[..seq_len * HIDDEN_SIZE],
                &mut ws.act_packed,
                &ws.act_scale[..seq_len],
                &layer.ffn_inter,
                None,
                &mut ws.inter_buf[..seq_len * INTERMEDIATE_SIZE],
            );

            // 2e. Fused 512-bit SVE / SIMD GELU + SmoothQuant + Quantize -> FFN Output GEMM + Residual -> Fused LN2 + Next Layer Quantize
            gelu_smooth_quantize_batch(
                seq_len,
                &mut ws.inter_buf[..seq_len * INTERMEDIATE_SIZE],
                &layer.ffn_out.inv_smooth,
                &mut ws.inter_q[..seq_len * INTERMEDIATE_SIZE],
                &mut ws.act_scale[..seq_len],
                use_sme,
            );

            gemm_int8_fused(
                seq_len,
                &ws.inter_q[..seq_len * INTERMEDIATE_SIZE],
                &mut ws.inter_packed,
                &ws.act_scale[..seq_len],
                &layer.ffn_out,
                Some(&ws.hidden[..seq_len * HIDDEN_SIZE]),
                &mut ws.tmp_hidden[..seq_len * HIDDEN_SIZE],
            );
            std::mem::swap(&mut ws.hidden, &mut ws.tmp_hidden);

            for t in 0..seq_len {
                let h_row = &mut ws.hidden[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                let q_row = &mut ws.act_q[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                ws.act_scale[t] = layernorm_smooth_quantize_row(
                    h_row,
                    &layer.ln2_weight,
                    &layer.ln2_bias,
                    next_qkv_smooth,
                    q_row,
                    LAYER_NORM_EPS,
                );
            }
        }

        // 3. Layer 11 (Final Layer): Exact CLS-Token Pruning!
        // Only token 0 (`[CLS]`, M = 1) is needed after Self-Attention!
        let last_layer = &self.layers[NUM_LAYERS - 1];
        let q_only = last_layer.q_only.as_ref().unwrap();
        let kv_only = last_layer.kv_only.as_ref().unwrap();

        // 3a. Compute Q only for token 0 (M = 1), and KV for all tokens (M = seq_len)
        gemm_int8_fused(
            1,
            &ws.act_q[..HIDDEN_SIZE],
            &mut ws.act_packed,
            &ws.act_scale[..1],
            q_only,
            None,
            &mut ws.q_cls[..HIDDEN_SIZE],
        );
        gemm_int8_fused(
            seq_len,
            &ws.act_q[..seq_len * HIDDEN_SIZE],
            &mut ws.act_packed,
            &ws.act_scale[..seq_len],
            kv_only,
            None,
            &mut ws.qkv_buf[..seq_len * 2 * HIDDEN_SIZE],
        );

        // 3b. Vectorized Single-Query Attention (Token 0 attending to 0..seq_len)
        for h in 0..NUM_HEADS {
            let h_off = h * HEAD_DIM;
            let q_slice = &ws.q_cls[h_off..h_off + HEAD_DIM];
            let scores_slice = &mut ws.scores[..seq_len];

            extract_kv_head32(
                &ws.qkv_buf,
                seq_len,
                2 * HIDDEN_SIZE,
                h_off,
                HIDDEN_SIZE + h_off,
                &mut ws.k_head,
                &mut ws.v_head,
            );
            compute_scores_head32(
                q_slice,
                &ws.k_head[..seq_len * HEAD_DIM],
                inv_sqrt_d,
                scores_slice,
            );
            softmax_inplace(scores_slice);
            let out_head = &mut ws.attn_ctx[h_off..h_off + HEAD_DIM];
            weighted_sum_head32(scores_slice, &ws.v_head[..seq_len * HEAD_DIM], out_head);
        }

        // 3c. Token 0 (M = 1) AttnOut + Residual, LN1, FFN Intermediate, GELU, FFN Output + Residual, LN2
        ws.act_scale[0] = smooth_quantize_row(
            &ws.attn_ctx[..HIDDEN_SIZE],
            &last_layer.attn_out.inv_smooth,
            &mut ws.act_q[..HIDDEN_SIZE],
        );
        gemm_int8_fused(
            1,
            &ws.act_q[..HIDDEN_SIZE],
            &mut ws.act_packed,
            &ws.act_scale[..1],
            &last_layer.attn_out,
            Some(&ws.hidden[..HIDDEN_SIZE]),
            &mut ws.tmp_hidden[..HIDDEN_SIZE],
        );

        ws.act_scale[0] = layernorm_smooth_quantize_row(
            &mut ws.tmp_hidden[..HIDDEN_SIZE],
            &last_layer.ln1_weight,
            &last_layer.ln1_bias,
            &last_layer.ffn_inter.inv_smooth,
            &mut ws.act_q[..HIDDEN_SIZE],
            LAYER_NORM_EPS,
        );

        gemm_int8_fused(
            1,
            &ws.act_q[..HIDDEN_SIZE],
            &mut ws.act_packed,
            &ws.act_scale[..1],
            &last_layer.ffn_inter,
            None,
            &mut ws.inter_buf[..INTERMEDIATE_SIZE],
        );

        gelu_smooth_quantize_batch(
            1,
            &mut ws.inter_buf[..INTERMEDIATE_SIZE],
            &last_layer.ffn_out.inv_smooth,
            &mut ws.inter_q[..INTERMEDIATE_SIZE],
            &mut ws.act_scale[..1],
            use_sme,
        );

        gemm_int8_fused(
            1,
            &ws.inter_q[..INTERMEDIATE_SIZE],
            &mut ws.inter_packed,
            &ws.act_scale[..1],
            &last_layer.ffn_out,
            Some(&ws.tmp_hidden[..HIDDEN_SIZE]),
            out_emb,
        );

        layernorm_inplace_row(
            out_emb,
            &last_layer.ln2_weight,
            &last_layer.ln2_bias,
            LAYER_NORM_EPS,
        );
        l2_normalize_inplace(out_emb);
    }

    /// Unpadded Ragged Multi-Sequence Forward Pass ("Flash-VarLen"):
    /// Concatenates `sequences` into a single unpadded `[total_tokens, 384]` buffer (`total_tokens <= MAX_POSITIONS`)
    /// so every GEMM, LayerNorm, and GELU in Layers 0..10 runs at full 16-row SME / 8-row I8MM tile occupancy,
    /// applies segment-bounded softmax in Self-Attention, and in Layer 11 gathers all `B = sequences.len()`
    /// `[CLS]` tokens so `q_only`, `attn_out`, `ln1`, `ffn_inter`, `gelu`, and `ffn_out` execute in a single
    /// batched `M = B` tile!
    pub fn forward_ragged_cls(
        &self,
        sequences: &[&[u16]],
        ws: &mut Workspace,
        out_embs: &mut [[f32; HIDDEN_SIZE]],
    ) {
        let b_count = sequences.len();
        assert_eq!(b_count, out_embs.len());
        if b_count == 0 {
            return;
        }
        if b_count == 1 {
            self.forward_cls(sequences[0], ws, &mut out_embs[0]);
            return;
        }

        ws.seg_ranges.clear();
        let mut cls_offsets = [0usize; 64];
        let mut seg_bounds = [(0usize, 0usize); 64];
        assert!(
            b_count <= 64,
            "forward_ragged_cls supports up to 64 packed sequences per call"
        );

        let first_inv_smooth = &self.layers[0].qkv.inv_smooth;
        let mut cursor = 0usize;

        // 1. Dequantize embeddings for each packed sequence using its local position `0..seq_len`
        for (b, seq) in sequences.iter().enumerate() {
            let s_len = seq.len().min(MAX_POSITIONS - cursor);
            assert!(s_len >= 1);
            let s_start = cursor;
            let s_end = cursor + s_len;
            cls_offsets[b] = s_start;
            seg_bounds[b] = (s_start, s_end);

            for (local_t, &tok) in seq[..s_len].iter().enumerate() {
                let global_t = s_start + local_t;
                ws.seg_ranges.push((s_start, s_end));

                let tok_idx = tok as usize;
                let w_off = tok_idx * HIDDEN_SIZE;
                let p_off = local_t * HIDDEN_SIZE;
                let dst_off = global_t * HIDDEN_SIZE;
                let w_s = self.word_emb_scale[tok_idx];
                let p_s = self.pos_emb_scale[local_t];

                let h_row = &mut ws.hidden[dst_off..dst_off + HIDDEN_SIZE];
                let w_row = &self.word_emb_q[w_off..w_off + HIDDEN_SIZE];
                let p_row = &self.pos_emb_q[p_off..p_off + HIDDEN_SIZE];
                let t_row = &self.type0_emb_f32[..HIDDEN_SIZE];

                dequant_add_embeddings_row(w_row, w_s, p_row, p_s, t_row, h_row);

                let q_row = &mut ws.act_q[dst_off..dst_off + HIDDEN_SIZE];
                ws.act_scale[global_t] = layernorm_smooth_quantize_row(
                    h_row,
                    &self.emb_ln_weight,
                    &self.emb_ln_bias,
                    first_inv_smooth,
                    q_row,
                    LAYER_NORM_EPS,
                );
            }
            cursor = s_end;
        }
        let total_tokens = cursor;

        // 2. Layers 0..10 across all `total_tokens` rows
        let inv_sqrt_d = 1.0 / (HEAD_DIM as f32).sqrt();
        let use_sme = self.backend == SimdBackend::ArmSme;

        for l_idx in 0..(NUM_LAYERS - 1) {
            let layer = &self.layers[l_idx];
            let next_qkv_smooth = &self.layers[l_idx + 1].qkv.inv_smooth;

            gemm_int8_fused(
                total_tokens,
                &ws.act_q[..total_tokens * HIDDEN_SIZE],
                &mut ws.act_packed,
                &ws.act_scale[..total_tokens],
                &layer.qkv,
                None,
                &mut ws.qkv_buf[..total_tokens * 3 * HIDDEN_SIZE],
            );

            #[cfg(target_arch = "aarch64")]
            if use_sme && total_tokens >= 32 {
                unsafe {
                    crate::simd_math::attention_multihead_sme(
                        total_tokens,
                        &ws.qkv_buf[..total_tokens * 3 * HIDDEN_SIZE],
                        &mut ws.sme_q_t,
                        &mut ws.sme_k_t,
                        &mut ws.v_head,
                        &mut ws.sme_scores_blk,
                        &mut ws.sme_p_t,
                        &mut ws.attn_ctx[..total_tokens * HIDDEN_SIZE],
                        Some(&ws.seg_ranges[..total_tokens]),
                    );
                }
            } else {
                for &(s_start, s_end) in &seg_bounds[..b_count] {
                    let s_len = s_end - s_start;
                    for h in 0..NUM_HEADS {
                        let h_off = h * HEAD_DIM;
                        for lt in 0..s_len {
                            let gt = s_start + lt;
                            let qkv_row =
                                &ws.qkv_buf[gt * 3 * HIDDEN_SIZE..(gt + 1) * 3 * HIDDEN_SIZE];
                            ws.k_head[lt * HEAD_DIM..(lt + 1) * HEAD_DIM].copy_from_slice(
                                &qkv_row[HIDDEN_SIZE + h_off..HIDDEN_SIZE + h_off + HEAD_DIM],
                            );
                            ws.v_head[lt * HEAD_DIM..(lt + 1) * HEAD_DIM].copy_from_slice(
                                &qkv_row
                                    [2 * HIDDEN_SIZE + h_off..2 * HIDDEN_SIZE + h_off + HEAD_DIM],
                            );
                        }
                        for ltq in 0..s_len {
                            let gtq = s_start + ltq;
                            let q_slice = &ws.qkv_buf[gtq * 3 * HIDDEN_SIZE + h_off
                                ..gtq * 3 * HIDDEN_SIZE + h_off + HEAD_DIM];
                            let scores_slice = &mut ws.scores[..s_len];
                            compute_scores_head32(
                                q_slice,
                                &ws.k_head[..s_len * HEAD_DIM],
                                inv_sqrt_d,
                                scores_slice,
                            );
                            softmax_inplace(scores_slice);
                            let out_head = &mut ws.attn_ctx
                                [gtq * HIDDEN_SIZE + h_off..gtq * HIDDEN_SIZE + h_off + HEAD_DIM];
                            weighted_sum_head32(
                                scores_slice,
                                &ws.v_head[..s_len * HEAD_DIM],
                                out_head,
                            );
                        }
                    }
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            for &(s_start, s_end) in &seg_bounds[..b_count] {
                let s_len = s_end - s_start;
                for h in 0..NUM_HEADS {
                    let h_off = h * HEAD_DIM;
                    for lt in 0..s_len {
                        let gt = s_start + lt;
                        let qkv_row = &ws.qkv_buf[gt * 3 * HIDDEN_SIZE..(gt + 1) * 3 * HIDDEN_SIZE];
                        ws.k_head[lt * HEAD_DIM..(lt + 1) * HEAD_DIM].copy_from_slice(
                            &qkv_row[HIDDEN_SIZE + h_off..HIDDEN_SIZE + h_off + HEAD_DIM],
                        );
                        ws.v_head[lt * HEAD_DIM..(lt + 1) * HEAD_DIM].copy_from_slice(
                            &qkv_row[2 * HIDDEN_SIZE + h_off..2 * HIDDEN_SIZE + h_off + HEAD_DIM],
                        );
                    }
                    for ltq in 0..s_len {
                        let gtq = s_start + ltq;
                        let q_slice = &ws.qkv_buf[gtq * 3 * HIDDEN_SIZE + h_off
                            ..gtq * 3 * HIDDEN_SIZE + h_off + HEAD_DIM];
                        let scores_slice = &mut ws.scores[..s_len];
                        compute_scores_head32(
                            q_slice,
                            &ws.k_head[..s_len * HEAD_DIM],
                            inv_sqrt_d,
                            scores_slice,
                        );
                        softmax_inplace(scores_slice);
                        let out_head = &mut ws.attn_ctx
                            [gtq * HIDDEN_SIZE + h_off..gtq * HIDDEN_SIZE + h_off + HEAD_DIM];
                        weighted_sum_head32(scores_slice, &ws.v_head[..s_len * HEAD_DIM], out_head);
                    }
                }
            }

            for t in 0..total_tokens {
                let ctx_row = &ws.attn_ctx[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                let q_row = &mut ws.act_q[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                ws.act_scale[t] = smooth_quantize_row(ctx_row, &layer.attn_out.inv_smooth, q_row);
            }

            gemm_int8_fused(
                total_tokens,
                &ws.act_q[..total_tokens * HIDDEN_SIZE],
                &mut ws.act_packed,
                &ws.act_scale[..total_tokens],
                &layer.attn_out,
                Some(&ws.hidden[..total_tokens * HIDDEN_SIZE]),
                &mut ws.tmp_hidden[..total_tokens * HIDDEN_SIZE],
            );
            std::mem::swap(&mut ws.hidden, &mut ws.tmp_hidden);

            for t in 0..total_tokens {
                let h_row = &mut ws.hidden[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                let q_row = &mut ws.act_q[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                ws.act_scale[t] = layernorm_smooth_quantize_row(
                    h_row,
                    &layer.ln1_weight,
                    &layer.ln1_bias,
                    &layer.ffn_inter.inv_smooth,
                    q_row,
                    LAYER_NORM_EPS,
                );
            }

            gemm_int8_fused(
                total_tokens,
                &ws.act_q[..total_tokens * HIDDEN_SIZE],
                &mut ws.act_packed,
                &ws.act_scale[..total_tokens],
                &layer.ffn_inter,
                None,
                &mut ws.inter_buf[..total_tokens * INTERMEDIATE_SIZE],
            );

            gelu_smooth_quantize_batch(
                total_tokens,
                &mut ws.inter_buf[..total_tokens * INTERMEDIATE_SIZE],
                &layer.ffn_out.inv_smooth,
                &mut ws.inter_q[..total_tokens * INTERMEDIATE_SIZE],
                &mut ws.act_scale[..total_tokens],
                use_sme,
            );

            gemm_int8_fused(
                total_tokens,
                &ws.inter_q[..total_tokens * INTERMEDIATE_SIZE],
                &mut ws.inter_packed,
                &ws.act_scale[..total_tokens],
                &layer.ffn_out,
                Some(&ws.hidden[..total_tokens * HIDDEN_SIZE]),
                &mut ws.tmp_hidden[..total_tokens * HIDDEN_SIZE],
            );
            std::mem::swap(&mut ws.hidden, &mut ws.tmp_hidden);

            for t in 0..total_tokens {
                let h_row = &mut ws.hidden[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                let q_row = &mut ws.act_q[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                ws.act_scale[t] = layernorm_smooth_quantize_row(
                    h_row,
                    &layer.ln2_weight,
                    &layer.ln2_bias,
                    next_qkv_smooth,
                    q_row,
                    LAYER_NORM_EPS,
                );
            }
        }

        // 3. Layer 11 (Final Layer): Batched Multi-CLS Token Pruning (`M = b_count` instead of `total_tokens`!)
        let last_layer = &self.layers[NUM_LAYERS - 1];
        let q_only = last_layer.q_only.as_ref().unwrap();
        let kv_only = last_layer.kv_only.as_ref().unwrap();

        // 3a. Compute KV for all `total_tokens` first (before compacting `act_q` to the `b_count` CLS rows)
        gemm_int8_fused(
            total_tokens,
            &ws.act_q[..total_tokens * HIDDEN_SIZE],
            &mut ws.act_packed,
            &ws.act_scale[..total_tokens],
            kv_only,
            None,
            &mut ws.qkv_buf[..total_tokens * 2 * HIDDEN_SIZE],
        );

        // Compact the `b_count` CLS rows in `act_q`, `act_scale`, and `hidden` into rows `0..b_count`
        for b in 0..b_count {
            let cls_t = cls_offsets[b];
            if cls_t != b {
                ws.act_q.copy_within(
                    cls_t * HIDDEN_SIZE..(cls_t + 1) * HIDDEN_SIZE,
                    b * HIDDEN_SIZE,
                );
                ws.act_scale[b] = ws.act_scale[cls_t];
                ws.hidden.copy_within(
                    cls_t * HIDDEN_SIZE..(cls_t + 1) * HIDDEN_SIZE,
                    b * HIDDEN_SIZE,
                );
            }
        }

        // Compute Q only for the `b_count` CLS tokens (`M = b_count`) in a single batched GEMM!
        gemm_int8_fused(
            b_count,
            &ws.act_q[..b_count * HIDDEN_SIZE],
            &mut ws.act_packed,
            &ws.act_scale[..b_count],
            q_only,
            None,
            &mut ws.q_cls[..b_count * HIDDEN_SIZE],
        );

        // 3b. Per-sequence CLS attention (sequence `b`'s CLS token attends to `seg_bounds[b]`)
        for b in 0..b_count {
            let (s_start, s_end) = seg_bounds[b];
            let s_len = s_end - s_start;
            for h in 0..NUM_HEADS {
                let h_off = h * HEAD_DIM;
                let q_slice =
                    &ws.q_cls[b * HIDDEN_SIZE + h_off..b * HIDDEN_SIZE + h_off + HEAD_DIM];
                let scores_slice = &mut ws.scores[..s_len];

                for lt in 0..s_len {
                    let gt = s_start + lt;
                    let kv_row = &ws.qkv_buf[gt * 2 * HIDDEN_SIZE..(gt + 1) * 2 * HIDDEN_SIZE];
                    ws.k_head[lt * HEAD_DIM..(lt + 1) * HEAD_DIM]
                        .copy_from_slice(&kv_row[h_off..h_off + HEAD_DIM]);
                    ws.v_head[lt * HEAD_DIM..(lt + 1) * HEAD_DIM].copy_from_slice(
                        &kv_row[HIDDEN_SIZE + h_off..HIDDEN_SIZE + h_off + HEAD_DIM],
                    );
                }
                compute_scores_head32(
                    q_slice,
                    &ws.k_head[..s_len * HEAD_DIM],
                    inv_sqrt_d,
                    scores_slice,
                );
                softmax_inplace(scores_slice);
                let out_head =
                    &mut ws.attn_ctx[b * HIDDEN_SIZE + h_off..b * HIDDEN_SIZE + h_off + HEAD_DIM];
                weighted_sum_head32(scores_slice, &ws.v_head[..s_len * HEAD_DIM], out_head);
            }
        }

        // 3c. Batched CLS post-attention pipeline (`M = b_count` rows in a single SME tile!)
        for b in 0..b_count {
            ws.act_scale[b] = smooth_quantize_row(
                &ws.attn_ctx[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
                &last_layer.attn_out.inv_smooth,
                &mut ws.act_q[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
            );
        }
        gemm_int8_fused(
            b_count,
            &ws.act_q[..b_count * HIDDEN_SIZE],
            &mut ws.act_packed,
            &ws.act_scale[..b_count],
            &last_layer.attn_out,
            Some(&ws.hidden[..b_count * HIDDEN_SIZE]),
            &mut ws.tmp_hidden[..b_count * HIDDEN_SIZE],
        );

        for b in 0..b_count {
            ws.act_scale[b] = layernorm_smooth_quantize_row(
                &mut ws.tmp_hidden[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
                &last_layer.ln1_weight,
                &last_layer.ln1_bias,
                &last_layer.ffn_inter.inv_smooth,
                &mut ws.act_q[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE],
                LAYER_NORM_EPS,
            );
        }

        gemm_int8_fused(
            b_count,
            &ws.act_q[..b_count * HIDDEN_SIZE],
            &mut ws.act_packed,
            &ws.act_scale[..b_count],
            &last_layer.ffn_inter,
            None,
            &mut ws.inter_buf[..b_count * INTERMEDIATE_SIZE],
        );

        gelu_smooth_quantize_batch(
            b_count,
            &mut ws.inter_buf[..b_count * INTERMEDIATE_SIZE],
            &last_layer.ffn_out.inv_smooth,
            &mut ws.inter_q[..b_count * INTERMEDIATE_SIZE],
            &mut ws.act_scale[..b_count],
            use_sme,
        );

        gemm_int8_fused(
            b_count,
            &ws.inter_q[..b_count * INTERMEDIATE_SIZE],
            &mut ws.inter_packed,
            &ws.act_scale[..b_count],
            &last_layer.ffn_out,
            Some(&ws.tmp_hidden[..b_count * HIDDEN_SIZE]),
            &mut ws.hidden[..b_count * HIDDEN_SIZE],
        );

        for b in 0..b_count {
            let row = &mut ws.hidden[b * HIDDEN_SIZE..(b + 1) * HIDDEN_SIZE];
            layernorm_inplace_row(
                row,
                &last_layer.ln2_weight,
                &last_layer.ln2_bias,
                LAYER_NORM_EPS,
            );
            l2_normalize_inplace(row);
            out_embs[b].copy_from_slice(row);
        }
    }
}
