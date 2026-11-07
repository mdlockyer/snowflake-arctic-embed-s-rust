use arctic_embed_s_int8::simd_math::*;
use std::time::Instant;
fn main() {
    let n = 1536;
    let m = 32;
    let s: Vec<f32> = (0..n)
        .map(|i| 0.3 + ((i * 37) % 100) as f32 / 60.0)
        .collect();
    let mut x: Vec<f32> = (0..m * n)
        .map(|i| {
            let t = ((i * 2654435761usize) % 100000) as f32 / 100000.0;
            (t - 0.6) * 9.0
        })
        .collect();
    x[5] = 25.0;
    x[n + 7] = -30.0;
    // reference: exact GELU * s, quantized in f64
    let mut worst = 0i32;
    let mut mism = 0usize;
    let mut max_scale_rel = 0f64;
    let mut buf = x.clone();
    let mut q = vec![0i8; m * n];
    let mut sc = vec![0f32; m];
    gelu_smooth_quantize_batch(m, &mut buf, &s, &mut q, &mut sc, true);
    for r in 0..m {
        let g: Vec<f64> = (0..n)
            .map(|j| {
                let v = x[r * n + j] as f64;
                0.5 * v * (1.0 + erf(v / 2f64.sqrt())) * s[j] as f64
            })
            .collect();
        let mx = g.iter().fold(0f64, |a, b| a.max(b.abs()));
        max_scale_rel = max_scale_rel.max(((sc[r] as f64) / (mx / 127.0) - 1.0).abs());
        for j in 0..n {
            let rq = (g[j] * 127.0 / mx).round() as i32;
            let d = (rq - q[r * n + j] as i32).abs();
            if d > 0 {
                mism += 1;
            }
            worst = worst.max(d);
        }
    }
    println!("vs exact erf-GELU: max |dq| = {worst} levels, mismatched = {:.3}% , max scale rel err = {:.2e}", 100.0*mism as f64/(m*n) as f64, max_scale_rel);
    let it = 20000;
    let t = Instant::now();
    for _ in 0..it {
        buf.copy_from_slice(&x);
        gelu_smooth_quantize_batch(m, std::hint::black_box(&mut buf), &s, &mut q, &mut sc, true);
    }
    let a = t.elapsed().as_secs_f64() * 1e6 / it as f64;
    let t = Instant::now();
    for _ in 0..it {
        buf.copy_from_slice(&x);
        std::hint::black_box(&mut buf);
    }
    let c = t.elapsed().as_secs_f64() * 1e6 / it as f64;
    let t = Instant::now();
    for _ in 0..it {
        buf.copy_from_slice(&x);
        for r in 0..m {
            sc[r] = gelu_smooth_quantize_row(
                std::hint::black_box(&mut buf[r * n..(r + 1) * n]),
                &s,
                &mut q[r * n..(r + 1) * n],
            );
        }
    }
    let b = t.elapsed().as_secs_f64() * 1e6 / it as f64;
    println!("M=32 GELU+quant: batch(f16) {:.2} us | f32 rows {:.2} us  (copy overhead {:.2} us subtracted)", a - c, b - c, c);
}
fn erf(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26 is too coarse; use series/continued fraction via erfc
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let y = 1.0
        - t * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    if x >= 0.0 {
        y
    } else {
        -y
    }
}
