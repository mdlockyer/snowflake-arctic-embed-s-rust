//! The portable `.ae8` quantized model pack (the only format the runtime reads), and — behind the
//! `convert` cargo feature — the zero-dependency `safetensors` parser and SmoothQuant-calibrated
//! W8A8 quantizer that produce it from the original FP32 checkpoint.

use crate::gemm::{PackedLinearInt8, SimdBackend};
#[cfg(feature = "convert")]
use crate::simd_math::scalar_gelu;
#[cfg(feature = "convert")]
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

pub const HIDDEN_SIZE: usize = 384;
pub const INTERMEDIATE_SIZE: usize = 1536;
pub const NUM_HEADS: usize = 12;
pub const HEAD_DIM: usize = 32;
pub const NUM_LAYERS: usize = 12;
pub const VOCAB_SIZE: usize = 30522;
pub const MAX_POSITIONS: usize = 512;
pub const TYPE_VOCAB_SIZE: usize = 2;
pub const LAYER_NORM_EPS: f32 = 1e-12;
pub const SMOOTHQUANT_ALPHA: f32 = 0.40;

const AE8_MAGIC: &[u8; 8] = b"ARCTIC8\x01";

#[derive(Clone)]
pub struct QuantizedLinearRaw {
    pub n_out: usize,
    pub k_in: usize,
    pub w_q: Vec<i8>,
    pub w_scale: Vec<f32>,
    pub bias: Vec<f32>,
    pub inv_smooth: Vec<f32>,
}

impl QuantizedLinearRaw {
    pub fn pack(&self, backend: SimdBackend) -> PackedLinearInt8 {
        PackedLinearInt8::from_raw_row_major(
            self.n_out,
            self.k_in,
            &self.w_q,
            &self.w_scale,
            &self.bias,
            &self.inv_smooth,
            backend,
        )
    }
}

#[derive(Clone)]
pub struct QuantizedLayerRaw {
    pub qkv: QuantizedLinearRaw,
    /// Separate Q (`[384, 384]`) and KV (`[768, 384]`) for Layer 11 CLS-only pruning.
    pub q_only: Option<QuantizedLinearRaw>,
    pub kv_only: Option<QuantizedLinearRaw>,
    pub attn_out: QuantizedLinearRaw,
    pub ln1_weight: Vec<f32>,
    pub ln1_bias: Vec<f32>,
    pub ffn_inter: QuantizedLinearRaw,
    pub ffn_out: QuantizedLinearRaw,
    pub ln2_weight: Vec<f32>,
    pub ln2_bias: Vec<f32>,
}

#[derive(Clone)]
pub struct QuantizedModelRaw {
    pub word_emb_q: Vec<i8>,
    pub word_emb_scale: Vec<f32>,
    pub pos_emb_q: Vec<i8>,
    pub pos_emb_scale: Vec<f32>,
    pub type_emb_q: Vec<i8>,
    pub type_emb_scale: Vec<f32>,
    pub emb_ln_weight: Vec<f32>,
    pub emb_ln_bias: Vec<f32>,
    pub layers: Vec<QuantizedLayerRaw>,
}

impl QuantizedModelRaw {
    #[cfg(feature = "convert")]
    /// Loads FP32 weights from `model.safetensors`, runs deterministic SmoothQuant calibration
    /// (`alpha = 0.40`), and quantizes all weights & embedding tables to symmetric Int8.
    pub fn from_safetensors_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        Self::from_safetensors_bytes(&bytes)
    }

    #[cfg(feature = "convert")]
    pub fn from_safetensors_bytes(bytes: &[u8]) -> io::Result<Self> {
        let st = SafeTensorsView::parse(bytes)?;

        let word_emb = st.get_f32("embeddings.word_embeddings.weight")?;
        let pos_emb = st.get_f32("embeddings.position_embeddings.weight")?;
        let type_emb = st.get_f32("embeddings.token_type_embeddings.weight")?;
        let emb_ln_weight = st.get_f32("embeddings.LayerNorm.weight")?;
        let emb_ln_bias = st.get_f32("embeddings.LayerNorm.bias")?;

        let (word_emb_q, word_emb_scale) = quantize_rows_i8(&word_emb, VOCAB_SIZE, HIDDEN_SIZE);
        let (pos_emb_q, pos_emb_scale) = quantize_rows_i8(&pos_emb, MAX_POSITIONS, HIDDEN_SIZE);
        let (type_emb_q, type_emb_scale) =
            quantize_rows_i8(&type_emb, TYPE_VOCAB_SIZE, HIDDEN_SIZE);

        // Deterministic calibration sequence `[CLS, 2000..2030, SEP]` (length 32)
        // Captures persistent outlier channels (ch=162, ch=323, L0 inter[711], L3 inter[115], L7 inter[1146])
        let mut calib_seq = Vec::with_capacity(32);
        calib_seq.push(101usize);
        for id in 2000..2030 {
            calib_seq.push(id);
        }
        calib_seq.push(102usize);
        let seq_len = calib_seq.len();

        let mut x = vec![0.0f32; seq_len * HIDDEN_SIZE];
        for (t, &tok_id) in calib_seq.iter().enumerate() {
            let row = &mut x[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
            let w_row = &word_emb[tok_id * HIDDEN_SIZE..(tok_id + 1) * HIDDEN_SIZE];
            let p_row = &pos_emb[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
            let t_row = &type_emb[0..HIDDEN_SIZE];
            for j in 0..HIDDEN_SIZE {
                row[j] = w_row[j] + p_row[j] + t_row[j];
            }
            layernorm_f32_row(row, &emb_ln_weight, &emb_ln_bias);
        }

        let mut layers = Vec::with_capacity(NUM_LAYERS);

        for l in 0..NUM_LAYERS {
            let prefix = format!("encoder.layer.{}.", l);
            let wq = st.get_f32(&format!("{}attention.self.query.weight", prefix))?;
            let bq = st.get_f32(&format!("{}attention.self.query.bias", prefix))?;
            let wk = st.get_f32(&format!("{}attention.self.key.weight", prefix))?;
            let bk = st.get_f32(&format!("{}attention.self.key.bias", prefix))?;
            let wv = st.get_f32(&format!("{}attention.self.value.weight", prefix))?;
            let bv = st.get_f32(&format!("{}attention.self.value.bias", prefix))?;

            let mut w_qkv = Vec::with_capacity(3 * HIDDEN_SIZE * HIDDEN_SIZE);
            w_qkv.extend_from_slice(&wq);
            w_qkv.extend_from_slice(&wk);
            w_qkv.extend_from_slice(&wv);

            let mut b_qkv = Vec::with_capacity(3 * HIDDEN_SIZE);
            b_qkv.extend_from_slice(&bq);
            b_qkv.extend_from_slice(&bk);
            b_qkv.extend_from_slice(&bv);

            let s_qkv = compute_smooth_scales(
                &x,
                seq_len,
                HIDDEN_SIZE,
                &w_qkv,
                3 * HIDDEN_SIZE,
                SMOOTHQUANT_ALPHA,
            );
            let (qkv_quant, q_only, kv_only) = if l == NUM_LAYERS - 1 {
                let mut w_kv = Vec::with_capacity(2 * HIDDEN_SIZE * HIDDEN_SIZE);
                w_kv.extend_from_slice(&wk);
                w_kv.extend_from_slice(&wv);
                let mut b_kv = Vec::with_capacity(2 * HIDDEN_SIZE);
                b_kv.extend_from_slice(&bk);
                b_kv.extend_from_slice(&bv);
                let inv_smooth: Vec<f32> = s_qkv.iter().map(|&s| 1.0 / s).collect();
                let pruned_qkv = QuantizedLinearRaw {
                    n_out: 0,
                    k_in: HIDDEN_SIZE,
                    w_q: Vec::new(),
                    w_scale: Vec::new(),
                    bias: Vec::new(),
                    inv_smooth,
                };
                (
                    pruned_qkv,
                    Some(quantize_linear_with_smooth(
                        HIDDEN_SIZE,
                        HIDDEN_SIZE,
                        &wq,
                        &bq,
                        &s_qkv,
                    )),
                    Some(quantize_linear_with_smooth(
                        2 * HIDDEN_SIZE,
                        HIDDEN_SIZE,
                        &w_kv,
                        &b_kv,
                        &s_qkv,
                    )),
                )
            } else {
                (
                    quantize_linear_with_smooth(
                        3 * HIDDEN_SIZE,
                        HIDDEN_SIZE,
                        &w_qkv,
                        &b_qkv,
                        &s_qkv,
                    ),
                    None,
                    None,
                )
            };

            // Propagate FP32 reference activation to calibrate attention output
            let q = matmul_fp32(&x, seq_len, HIDDEN_SIZE, &wq, HIDDEN_SIZE, &bq);
            let k = matmul_fp32(&x, seq_len, HIDDEN_SIZE, &wk, HIDDEN_SIZE, &bk);
            let v = matmul_fp32(&x, seq_len, HIDDEN_SIZE, &wv, HIDDEN_SIZE, &bv);
            let ctx = attention_fp32(&q, &k, &v, seq_len);

            let w_ao = st.get_f32(&format!("{}attention.output.dense.weight", prefix))?;
            let b_ao = st.get_f32(&format!("{}attention.output.dense.bias", prefix))?;
            let s_ao = compute_smooth_scales(
                &ctx,
                seq_len,
                HIDDEN_SIZE,
                &w_ao,
                HIDDEN_SIZE,
                SMOOTHQUANT_ALPHA,
            );
            let attn_out_quant =
                quantize_linear_with_smooth(HIDDEN_SIZE, HIDDEN_SIZE, &w_ao, &b_ao, &s_ao);

            let attn_out = matmul_fp32(&ctx, seq_len, HIDDEN_SIZE, &w_ao, HIDDEN_SIZE, &b_ao);
            let ln1_weight = st.get_f32(&format!("{}attention.output.LayerNorm.weight", prefix))?;
            let ln1_bias = st.get_f32(&format!("{}attention.output.LayerNorm.bias", prefix))?;
            for t in 0..seq_len {
                let row = &mut x[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                let ao_row = &attn_out[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                for j in 0..HIDDEN_SIZE {
                    row[j] += ao_row[j];
                }
                layernorm_f32_row(row, &ln1_weight, &ln1_bias);
            }

            let w_inter = st.get_f32(&format!("{}intermediate.dense.weight", prefix))?;
            let b_inter = st.get_f32(&format!("{}intermediate.dense.bias", prefix))?;
            let s_inter = compute_smooth_scales(
                &x,
                seq_len,
                HIDDEN_SIZE,
                &w_inter,
                INTERMEDIATE_SIZE,
                SMOOTHQUANT_ALPHA,
            );
            let ffn_inter_quant = quantize_linear_with_smooth(
                INTERMEDIATE_SIZE,
                HIDDEN_SIZE,
                &w_inter,
                &b_inter,
                &s_inter,
            );

            let mut inter = matmul_fp32(
                &x,
                seq_len,
                HIDDEN_SIZE,
                &w_inter,
                INTERMEDIATE_SIZE,
                &b_inter,
            );
            for v in inter.iter_mut() {
                *v = scalar_gelu(*v);
            }

            let w_out = st.get_f32(&format!("{}output.dense.weight", prefix))?;
            let b_out = st.get_f32(&format!("{}output.dense.bias", prefix))?;
            let s_out = compute_smooth_scales(
                &inter,
                seq_len,
                INTERMEDIATE_SIZE,
                &w_out,
                HIDDEN_SIZE,
                SMOOTHQUANT_ALPHA,
            );
            let ffn_out_quant =
                quantize_linear_with_smooth(HIDDEN_SIZE, INTERMEDIATE_SIZE, &w_out, &b_out, &s_out);

            let ffn_out = matmul_fp32(
                &inter,
                seq_len,
                INTERMEDIATE_SIZE,
                &w_out,
                HIDDEN_SIZE,
                &b_out,
            );
            let ln2_weight = st.get_f32(&format!("{}output.LayerNorm.weight", prefix))?;
            let ln2_bias = st.get_f32(&format!("{}output.LayerNorm.bias", prefix))?;
            for t in 0..seq_len {
                let row = &mut x[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                let fo_row = &ffn_out[t * HIDDEN_SIZE..(t + 1) * HIDDEN_SIZE];
                for j in 0..HIDDEN_SIZE {
                    row[j] += fo_row[j];
                }
                layernorm_f32_row(row, &ln2_weight, &ln2_bias);
            }

            layers.push(QuantizedLayerRaw {
                qkv: qkv_quant,
                q_only,
                kv_only,
                attn_out: attn_out_quant,
                ln1_weight,
                ln1_bias,
                ffn_inter: ffn_inter_quant,
                ffn_out: ffn_out_quant,
                ln2_weight,
                ln2_bias,
            });
        }

        Ok(Self {
            word_emb_q,
            word_emb_scale,
            pos_emb_q,
            pos_emb_scale,
            type_emb_q,
            type_emb_scale,
            emb_ln_weight,
            emb_ln_bias,
            layers,
        })
    }

    /// Saves the quantized model in portable `.ae8` binary format (~33.5 MB).
    pub fn save_ae8<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut f = io::BufWriter::new(fs::File::create(path)?);
        f.write_all(AE8_MAGIC)?;
        write_i8_slice(&mut f, &self.word_emb_q)?;
        write_f32_slice(&mut f, &self.word_emb_scale)?;
        write_i8_slice(&mut f, &self.pos_emb_q)?;
        write_f32_slice(&mut f, &self.pos_emb_scale)?;
        write_i8_slice(&mut f, &self.type_emb_q)?;
        write_f32_slice(&mut f, &self.type_emb_scale)?;
        write_f32_slice(&mut f, &self.emb_ln_weight)?;
        write_f32_slice(&mut f, &self.emb_ln_bias)?;

        for l in &self.layers {
            write_linear(&mut f, &l.qkv)?;
            if let (Some(q), Some(kv)) = (&l.q_only, &l.kv_only) {
                f.write_all(&[1u8])?;
                write_linear(&mut f, q)?;
                write_linear(&mut f, kv)?;
            } else {
                f.write_all(&[0u8])?;
            }
            write_linear(&mut f, &l.attn_out)?;
            write_f32_slice(&mut f, &l.ln1_weight)?;
            write_f32_slice(&mut f, &l.ln1_bias)?;
            write_linear(&mut f, &l.ffn_inter)?;
            write_linear(&mut f, &l.ffn_out)?;
            write_f32_slice(&mut f, &l.ln2_weight)?;
            write_f32_slice(&mut f, &l.ln2_bias)?;
        }
        f.flush()
    }

    /// Loads the quantized model from portable `.ae8` binary format.
    pub fn load_ae8<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        let mut cur = &bytes[..];
        let mut magic = [0u8; 8];
        cur.read_exact(&mut magic)?;
        if &magic != AE8_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid .ae8 magic header",
            ));
        }
        let word_emb_q = read_i8_vec(&mut cur)?;
        let word_emb_scale = read_f32_vec(&mut cur)?;
        let pos_emb_q = read_i8_vec(&mut cur)?;
        let pos_emb_scale = read_f32_vec(&mut cur)?;
        let type_emb_q = read_i8_vec(&mut cur)?;
        let type_emb_scale = read_f32_vec(&mut cur)?;
        let emb_ln_weight = read_f32_vec(&mut cur)?;
        let emb_ln_bias = read_f32_vec(&mut cur)?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for _ in 0..NUM_LAYERS {
            let qkv = read_linear(&mut cur)?;
            let mut has_split = [0u8; 1];
            cur.read_exact(&mut has_split)?;
            let (q_only, kv_only) = if has_split[0] == 1 {
                (Some(read_linear(&mut cur)?), Some(read_linear(&mut cur)?))
            } else {
                (None, None)
            };
            let attn_out = read_linear(&mut cur)?;
            let ln1_weight = read_f32_vec(&mut cur)?;
            let ln1_bias = read_f32_vec(&mut cur)?;
            let ffn_inter = read_linear(&mut cur)?;
            let ffn_out = read_linear(&mut cur)?;
            let ln2_weight = read_f32_vec(&mut cur)?;
            let ln2_bias = read_f32_vec(&mut cur)?;
            layers.push(QuantizedLayerRaw {
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
            word_emb_q,
            word_emb_scale,
            pos_emb_q,
            pos_emb_scale,
            type_emb_q,
            type_emb_scale,
            emb_ln_weight,
            emb_ln_bias,
            layers,
        })
    }

    /// Total memory footprint of quantized weights in bytes.
    pub fn size_bytes(&self) -> usize {
        let mut total = self.word_emb_q.len()
            + self.pos_emb_q.len()
            + self.type_emb_q.len()
            + (self.word_emb_scale.len()
                + self.pos_emb_scale.len()
                + self.type_emb_scale.len()
                + self.emb_ln_weight.len()
                + self.emb_ln_bias.len())
                * 4;
        for l in &self.layers {
            for lin in [&l.qkv, &l.attn_out, &l.ffn_inter, &l.ffn_out] {
                total +=
                    lin.w_q.len() + (lin.w_scale.len() + lin.bias.len() + lin.inv_smooth.len()) * 4;
            }
            if let (Some(q), Some(kv)) = (&l.q_only, &l.kv_only) {
                for lin in [q, kv] {
                    total += lin.w_q.len()
                        + (lin.w_scale.len() + lin.bias.len() + lin.inv_smooth.len()) * 4;
                }
            }
            total +=
                (l.ln1_weight.len() + l.ln1_bias.len() + l.ln2_weight.len() + l.ln2_bias.len()) * 4;
        }
        total
    }
}

#[cfg(feature = "convert")]
fn quantize_rows_i8(data: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let q_row = &mut q[r * cols..(r + 1) * cols];
        let mut max_abs = 1e-8f32;
        for &v in row {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = max_abs / 127.0;
        let inv_scale = 127.0 / max_abs;
        scales[r] = scale;
        for c in 0..cols {
            q_row[c] = (row[c] * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, scales)
}

#[cfg(feature = "convert")]
fn compute_smooth_scales(
    x: &[f32],
    m: usize,
    k: usize,
    w: &[f32],
    n: usize,
    alpha: f32,
) -> Vec<f32> {
    let mut act_max = vec![1e-5f32; k];
    for r in 0..m {
        let row = &x[r * k..(r + 1) * k];
        for j in 0..k {
            let a = row[j].abs();
            if a > act_max[j] {
                act_max[j] = a;
            }
        }
    }
    let mut w_max = vec![1e-5f32; k];
    for row_idx in 0..n {
        let row = &w[row_idx * k..(row_idx + 1) * k];
        for j in 0..k {
            let a = row[j].abs();
            if a > w_max[j] {
                w_max[j] = a;
            }
        }
    }
    let mut s = vec![1.0f32; k];
    for j in 0..k {
        s[j] = act_max[j].powf(alpha) / w_max[j].powf(1.0 - alpha);
    }
    s
}

#[cfg(feature = "convert")]
fn quantize_linear_with_smooth(
    n_out: usize,
    k_in: usize,
    w: &[f32],
    bias: &[f32],
    smooth: &[f32],
) -> QuantizedLinearRaw {
    let mut w_smoothed = vec![0.0f32; n_out * k_in];
    for r in 0..n_out {
        let src = &w[r * k_in..(r + 1) * k_in];
        let dst = &mut w_smoothed[r * k_in..(r + 1) * k_in];
        for j in 0..k_in {
            dst[j] = src[j] * smooth[j];
        }
    }
    let (w_q, w_scale) = quantize_rows_i8(&w_smoothed, n_out, k_in);
    let inv_smooth: Vec<f32> = smooth.iter().map(|&s| 1.0 / s).collect();
    QuantizedLinearRaw {
        n_out,
        k_in,
        w_q,
        w_scale,
        bias: bias.to_vec(),
        inv_smooth,
    }
}

#[cfg(feature = "convert")]
fn layernorm_f32_row(row: &mut [f32], gamma: &[f32], beta: &[f32]) {
    let n = row.len();
    let mean = row.iter().sum::<f32>() / (n as f32);
    let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / (n as f32);
    let rstd = 1.0 / (var + LAYER_NORM_EPS).sqrt();
    for j in 0..n {
        row[j] = (row[j] - mean) * rstd * gamma[j] + beta[j];
    }
}

#[cfg(feature = "convert")]
fn matmul_fp32(a: &[f32], m: usize, k: usize, w: &[f32], n: usize, bias: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for r in 0..m {
        let a_row = &a[r * k..(r + 1) * k];
        let out_row = &mut out[r * n..(r + 1) * n];
        for col in 0..n {
            let w_row = &w[col * k..(col + 1) * k];
            let mut sum = bias[col];
            for j in 0..k {
                sum += a_row[j] * w_row[j];
            }
            out_row[col] = sum;
        }
    }
    out
}

#[cfg(feature = "convert")]
fn attention_fp32(q: &[f32], k: &[f32], v: &[f32], seq_len: usize) -> Vec<f32> {
    let mut ctx = vec![0.0f32; seq_len * HIDDEN_SIZE];
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    for h in 0..NUM_HEADS {
        let h_off = h * HEAD_DIM;
        for tq in 0..seq_len {
            let q_slice = &q[tq * HIDDEN_SIZE + h_off..tq * HIDDEN_SIZE + h_off + HEAD_DIM];
            let mut scores = vec![0.0f32; seq_len];
            let mut max_s = f32::NEG_INFINITY;
            for tk in 0..seq_len {
                let k_slice = &k[tk * HIDDEN_SIZE + h_off..tk * HIDDEN_SIZE + h_off + HEAD_DIM];
                let mut dot = 0.0f32;
                for d in 0..HEAD_DIM {
                    dot += q_slice[d] * k_slice[d];
                }
                let s = dot * scale;
                scores[tk] = s;
                if s > max_s {
                    max_s = s;
                }
            }
            let mut sum_exp = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum_exp;
            }
            let out_slice = &mut ctx[tq * HIDDEN_SIZE + h_off..tq * HIDDEN_SIZE + h_off + HEAD_DIM];
            for tk in 0..seq_len {
                let p = scores[tk];
                let v_slice = &v[tk * HIDDEN_SIZE + h_off..tk * HIDDEN_SIZE + h_off + HEAD_DIM];
                for d in 0..HEAD_DIM {
                    out_slice[d] += p * v_slice[d];
                }
            }
        }
    }
    ctx
}

#[cfg(feature = "convert")]
/// Minimal zero-dependency `safetensors` parser.
struct SafeTensorsView<'a> {
    data: &'a [u8],
    offsets: HashMap<String, (usize, usize)>,
}

#[cfg(feature = "convert")]
impl<'a> SafeTensorsView<'a> {
    fn parse(bytes: &'a [u8]) -> io::Result<Self> {
        if bytes.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors file too short",
            ));
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if bytes.len() < 8 + header_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors header truncated",
            ));
        }
        let header_str = std::str::from_utf8(&bytes[8..8 + header_len])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let offsets = parse_safetensors_header(header_str)?;
        Ok(Self {
            data: &bytes[8 + header_len..],
            offsets,
        })
    }

    fn get_f32(&self, name: &str) -> io::Result<Vec<f32>> {
        let &(start, end) = self.offsets.get(name).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("tensor '{}' not found in safetensors", name),
            )
        })?;
        let slice = &self.data[start..end];
        let count = slice.len() / 4;
        let mut out = vec![0.0f32; count];
        #[cfg(target_endian = "little")]
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), out.as_mut_ptr() as *mut u8, count * 4);
        }
        #[cfg(not(target_endian = "little"))]
        for i in 0..count {
            out[i] = f32::from_le_bytes([
                slice[i * 4],
                slice[i * 4 + 1],
                slice[i * 4 + 2],
                slice[i * 4 + 3],
            ]);
        }
        Ok(out)
    }
}

/// Parses `"tensor_name":{"dtype":"F32","shape":[...],"data_offsets":[start,end]}` entries.
#[cfg(feature = "convert")]
fn parse_safetensors_header(s: &str) -> io::Result<HashMap<String, (usize, usize)>> {
    let mut map = HashMap::new();
    let mut pos = 0usize;
    let bytes = s.as_bytes();

    while let Some(q1_rel) = s[pos..].find('"') {
        let q1 = pos + q1_rel + 1;
        let Some(q2_rel) = s[q1..].find('"') else {
            break;
        };
        let q2 = q1 + q2_rel;
        let key = &s[q1..q2];
        pos = q2 + 1;

        // Expect `:` followed by `{` for top-level keys
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() || bytes[pos] != b':' {
            continue;
        }
        pos += 1;
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() || bytes[pos] != b'{' {
            continue;
        }
        let obj_start = pos;
        let mut depth = 0i32;
        while pos < bytes.len() {
            if bytes[pos] == b'{' {
                depth += 1;
            } else if bytes[pos] == b'}' {
                depth -= 1;
                if depth == 0 {
                    pos += 1;
                    break;
                }
            }
            pos += 1;
        }
        let obj_str = &s[obj_start..pos];
        if key == "__metadata__" {
            continue;
        }
        if let Some(off_idx) = obj_str.find("\"data_offsets\"") {
            let after = &obj_str[off_idx + 14..];
            if let (Some(b_open), Some(b_close)) = (after.find('['), after.find(']')) {
                let nums = &after[b_open + 1..b_close];
                let mut parts = nums.split(',');
                if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
                    if let (Ok(start), Ok(end)) =
                        (a.trim().parse::<usize>(), b.trim().parse::<usize>())
                    {
                        map.insert(key.to_string(), (start, end));
                    }
                }
            }
        }
    }
    Ok(map)
}

fn write_i8_slice<W: Write>(w: &mut W, s: &[i8]) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    let u8_slice = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()) };
    w.write_all(u8_slice)
}

fn write_f32_slice<W: Write>(w: &mut W, s: &[f32]) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    for &v in s {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn write_linear<W: Write>(w: &mut W, lin: &QuantizedLinearRaw) -> io::Result<()> {
    w.write_all(&(lin.n_out as u64).to_le_bytes())?;
    w.write_all(&(lin.k_in as u64).to_le_bytes())?;
    write_i8_slice(w, &lin.w_q)?;
    write_f32_slice(w, &lin.w_scale)?;
    write_f32_slice(w, &lin.bias)?;
    write_f32_slice(w, &lin.inv_smooth)
}

fn read_i8_vec<R: Read>(r: &mut R) -> io::Result<Vec<i8>> {
    let mut len_b = [0u8; 8];
    r.read_exact(&mut len_b)?;
    let len = u64::from_le_bytes(len_b) as usize;
    let mut out = vec![0i8; len];
    let u8_slice = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, len) };
    r.read_exact(u8_slice)?;
    Ok(out)
}

fn read_f32_vec<R: Read>(r: &mut R) -> io::Result<Vec<f32>> {
    let mut len_b = [0u8; 8];
    r.read_exact(&mut len_b)?;
    let len = u64::from_le_bytes(len_b) as usize;
    let mut raw = vec![0u8; len * 4];
    r.read_exact(&mut raw)?;
    let mut out = vec![0.0f32; len];
    for i in 0..len {
        out[i] = f32::from_le_bytes([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
    }
    Ok(out)
}

fn read_linear<R: Read>(r: &mut R) -> io::Result<QuantizedLinearRaw> {
    let mut b8 = [0u8; 8];
    r.read_exact(&mut b8)?;
    let n_out = u64::from_le_bytes(b8) as usize;
    r.read_exact(&mut b8)?;
    let k_in = u64::from_le_bytes(b8) as usize;
    let w_q = read_i8_vec(r)?;
    let w_scale = read_f32_vec(r)?;
    let bias = read_f32_vec(r)?;
    let inv_smooth = read_f32_vec(r)?;
    Ok(QuantizedLinearRaw {
        n_out,
        k_in,
        w_q,
        w_scale,
        bias,
        inv_smooth,
    })
}
