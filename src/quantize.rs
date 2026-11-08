//! Loader for the portable `.ae8` quantized model pack produced by
//! `snowflake-arctic-embed-s-weights`.

use crate::gemm::{PackedLinearInt8, SimdBackend};
use std::fs;
use std::io::{self, Read};
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
