use arctic_embed_s_int8::gemm::gemm_int8_fused;
use arctic_embed_s_int8::model::Workspace;
use arctic_embed_s_int8::simd_math::*;
use arctic_embed_s_int8::ArcticEmbedS;
use std::time::Instant;
fn t<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    for _ in 0..20 {
        f();
    }
    let s = Instant::now();
    for _ in 0..iters {
        f();
    }
    s.elapsed().as_secs_f64() * 1e6 / iters as f64
}
fn main() {
    let e = ArcticEmbedS::from_dir(arctic_embed_s_int8::default_weights_dir()).unwrap();
    let mut ws = Workspace::new();
    for (i, v) in ws.act_q.iter_mut().enumerate() {
        *v = ((i * 7919) % 255) as i8;
    }
    for (i, v) in ws.inter_q.iter_mut().enumerate() {
        *v = ((i * 7919) % 255) as i8;
    }
    for v in ws.act_scale.iter_mut() {
        *v = 0.01;
    }
    for (i, v) in ws.inter_buf.iter_mut().enumerate() {
        *v = ((i * 7919) % 1000) as f32 / 250.0 - 2.0;
    }
    for (i, v) in ws.hidden.iter_mut().enumerate() {
        *v = ((i * 7919) % 1000) as f32 / 250.0 - 2.0;
    }
    // Rotate through all 11 layers so weights come from wherever they'd really live (L2/SLC/DRAM).
    let ls = &e.model.layers;
    for &m in &[8usize, 16, 32, 128] {
        let it = 4000 / m.max(8) * 4;
        let mut l = 0;
        let qkv = t(it, || {
            l = (l + 1) % 11;
            gemm_int8_fused(
                m,
                &ws.act_q[..m * 384],
                &mut ws.act_packed,
                &ws.act_scale[..m],
                &ls[l].qkv,
                None,
                &mut ws.qkv_buf[..m * 1152],
            );
        });
        let ao = t(it, || {
            l = (l + 1) % 11;
            gemm_int8_fused(
                m,
                &ws.act_q[..m * 384],
                &mut ws.act_packed,
                &ws.act_scale[..m],
                &ls[l].attn_out,
                Some(&ws.hidden[..m * 384]),
                &mut ws.tmp_hidden[..m * 384],
            );
        });
        let fu = t(it, || {
            l = (l + 1) % 11;
            gemm_int8_fused(
                m,
                &ws.act_q[..m * 384],
                &mut ws.act_packed,
                &ws.act_scale[..m],
                &ls[l].ffn_inter,
                None,
                &mut ws.inter_buf[..m * 1536],
            );
        });
        let fd = t(it, || {
            l = (l + 1) % 11;
            gemm_int8_fused(
                m,
                &ws.inter_q[..m * 1536],
                &mut ws.inter_packed,
                &ws.act_scale[..m],
                &ls[l].ffn_out,
                Some(&ws.hidden[..m * 384]),
                &mut ws.tmp_hidden[..m * 384],
            );
        });
        let ln = t(it, || {
            for r in 0..m {
                layernorm_smooth_quantize_row(
                    &mut ws.tmp_hidden[r * 384..(r + 1) * 384],
                    &ls[0].ln1_weight,
                    &ls[0].ln1_bias,
                    &ls[0].ffn_inter.inv_smooth,
                    &mut ws.act_q[r * 384..(r + 1) * 384],
                    1e-12,
                );
            }
        });
        let ge = t(it, || {
            for (i, v) in ws.inter_buf[..m * 1536].iter_mut().enumerate() {
                *v = (i % 97) as f32 * 0.05 - 2.4;
            }
            gelu_smooth_quantize_batch(
                m,
                &mut ws.inter_buf[..m * 1536],
                &ls[0].ffn_out.inv_smooth,
                &mut ws.inter_q[..m * 1536],
                &mut ws.act_scale[..m],
                true,
            );
        });
        let gemm = qkv + ao + fu + fd;
        println!("M={m:3}: QKV {qkv:6.1} AO {ao:6.1} FFNup {fu:6.1} FFNdn {fd:6.1} | GEMM sum {gemm:6.1} ({:.0} GB/s weights, {:.2} TMAC/s) | LN {ln:5.1} GELU(+fill) {ge:5.1} us",
            1_769_472.0 / gemm / 1e3, (m as f64 * 1_769_472.0) / gemm / 1e6);
    }
}
