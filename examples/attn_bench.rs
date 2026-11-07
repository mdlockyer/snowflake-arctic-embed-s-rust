//! Micro-benchmark of the SME attention kernel vs. the NEON per-head path (Apple M4 / SME2 only).
#[cfg(target_arch = "aarch64")]
use arctic_embed_s_int8::{gemm::SimdBackend, model::Workspace, simd_math::*};
#[cfg(target_arch = "aarch64")]
use std::time::Instant;

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("attn_bench benchmarks the SME kernels and only runs on aarch64");
}

#[cfg(target_arch = "aarch64")]
fn main() {
    if !SimdBackend::ArmSme.is_supported() {
        eprintln!("attn_bench needs a CPU with SME2 (e.g. Apple M4)");
        return;
    }
    let mut ws = Workspace::new();
    for (i, v) in ws.qkv_buf.iter_mut().enumerate() {
        *v = ((i * 2654435761usize) % 1000) as f32 / 300.0 - 1.6;
    }
    for &seq in &[4usize, 6, 8, 10, 12, 16, 32, 64, 128, 512] {
        let iters = 20000 / seq;
        let t = Instant::now();
        for _ in 0..iters {
            unsafe {
                attention_multihead_sme(
                    seq,
                    &ws.qkv_buf[..seq * 1152],
                    &mut ws.sme_q_t,
                    &mut ws.sme_k_t,
                    &mut ws.v_head,
                    &mut ws.sme_scores_blk,
                    &mut ws.sme_p_t,
                    &mut ws.attn_ctx[..seq * 384],
                    None,
                );
            }
        }
        let sme = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let inv = 1.0 / 32f32.sqrt();
        let t = Instant::now();
        for _ in 0..iters {
            for h in 0..12 {
                let h_off = h * 32;
                extract_kv_head32(
                    &ws.qkv_buf,
                    seq,
                    1152,
                    384 + h_off,
                    768 + h_off,
                    &mut ws.k_head,
                    &mut ws.v_head,
                );
                for tq in 0..seq {
                    let q = &ws.qkv_buf[tq * 1152 + h_off..tq * 1152 + h_off + 32];
                    let s = &mut ws.scores[..seq];
                    compute_scores_head32(q, &ws.k_head[..seq * 32], inv, s);
                    softmax_inplace(s);
                    weighted_sum_head32(
                        s,
                        &ws.v_head[..seq * 32],
                        &mut ws.attn_ctx[tq * 384 + h_off..tq * 384 + h_off + 32],
                    );
                }
            }
        }
        let neon = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!("seq={seq:4}: SME int8 attn {sme:8.2} us | NEON f32 attn {neon:8.2} us");
    }
}
