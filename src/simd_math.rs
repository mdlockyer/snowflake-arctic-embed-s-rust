//! Vectorized SIMD math & fused operator kernels for ARM64 NEON and x86_64 AVX2+FMA.
//!
//! Includes:
//! - Cody-Waite SIMD `exp` and Abramowitz-Stegun minimax SIMD `GELU` (< 5e-7 max error)
//! - Fused `LayerNorm + SmoothQuant + Dynamic Int8 Quantization`
//! - Fused `GELU + SmoothQuant + Dynamic Int8 Quantization`
//! - Register-resident `head_dim = 32` Multi-Head Self-Attention

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub fn scalar_gelu(x: f32) -> f32 {
    let z = x.abs() * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf_val = (1.0 - poly * (-z * z).exp()).copysign(x);
    0.5 * x * (1.0 + erf_val)
}

/// Vectorized `exp(x)` (inputs clamped to `[-87.3, 88]`).
///
/// # Safety
/// Requires NEON (always present on AArch64).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_exp_f32(y_in: float32x4_t) -> float32x4_t {
    let y = vmaxq_f32(vminq_f32(y_in, vdupq_n_f32(88.0)), vdupq_n_f32(-87.3));
    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2_hi = vdupq_n_f32(0.693145751953125);
    let ln2_lo = vdupq_n_f32(1.428606765330187e-6);

    let n = vrndnq_f32(vmulq_f32(y, log2e));
    let r = vfmsq_f32(vfmsq_f32(y, n, ln2_hi), n, ln2_lo);

    let c5 = vdupq_n_f32(0.008333334);
    let c4 = vdupq_n_f32(0.041666668);
    let c3 = vdupq_n_f32(0.16666667);
    let c2 = vdupq_n_f32(0.5);
    let c1 = vdupq_n_f32(1.0);

    let mut p = vfmaq_f32(c4, r, c5);
    p = vfmaq_f32(c3, r, p);
    p = vfmaq_f32(c2, r, p);
    p = vfmaq_f32(c1, r, p);
    p = vfmaq_f32(c1, r, p);

    let ni = vaddq_s32(vcvtnq_s32_f32(n), vdupq_n_s32(127));
    let pow2n = vreinterpretq_f32_s32(vshlq_n_s32(ni, 23));
    vmulq_f32(p, pow2n)
}

/// GELU of two 4-lane vectors (rational approximation, inputs clamped to `[-4.8, 4.8]`).
///
/// # Safety
/// Requires NEON (always present on AArch64).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn neon_gelu_pair_f32(x0: float32x4_t, x1: float32x4_t) -> (float32x4_t, float32x4_t) {
    let vmin = vdupq_n_f32(-4.8);
    let vmax = vdupq_n_f32(4.8);
    let xc0 = vminq_f32(vmaxq_f32(x0, vmin), vmax);
    let xc1 = vminq_f32(vmaxq_f32(x1, vmin), vmax);
    let u0 = vmulq_f32(xc0, xc0);
    let u1 = vmulq_f32(xc1, xc1);

    let p3 = vdupq_n_f32(3.214524e-5);
    let p2 = vdupq_n_f32(0.0037863153);
    let p1 = vdupq_n_f32(0.027641433);
    let p0 = vdupq_n_f32(0.39894123);

    let mut num0 = vfmaq_f32(p2, u0, p3);
    let mut num1 = vfmaq_f32(p2, u1, p3);
    num0 = vfmaq_f32(p1, u0, num0);
    num1 = vfmaq_f32(p1, u1, num1);
    num0 = vfmaq_f32(p0, u0, num0);
    num1 = vfmaq_f32(p0, u1, num1);

    let q3 = vdupq_n_f32(0.001139532);
    let q2 = vdupq_n_f32(0.02381135);
    let q1 = vdupq_n_f32(0.2359431);
    let one = vdupq_n_f32(1.0);

    let mut den0 = vfmaq_f32(q2, u0, q3);
    let mut den1 = vfmaq_f32(q2, u1, q3);
    den0 = vfmaq_f32(q1, u0, den0);
    den1 = vfmaq_f32(q1, u1, den1);
    den0 = vfmaq_f32(one, u0, den0);
    den1 = vfmaq_f32(one, u1, den1);

    let est0 = vrecpeq_f32(den0);
    let est1 = vrecpeq_f32(den1);
    let inv0 = vmulq_f32(est0, vrecpsq_f32(den0, est0));
    let inv1 = vmulq_f32(est1, vrecpsq_f32(den1, est1));

    let half = vdupq_n_f32(0.5);
    let gate0 = vfmaq_f32(half, xc0, vmulq_f32(num0, inv0));
    let gate1 = vfmaq_f32(half, xc1, vmulq_f32(num1, inv1));
    (vmulq_f32(x0, gate0), vmulq_f32(x1, gate1))
}

/// Vectorized `exp(x)` (inputs clamped to `[-87.3, 88]`).
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn avx2_exp_f32(y_in: __m256) -> __m256 {
    let y = _mm256_max_ps(
        _mm256_min_ps(y_in, _mm256_set1_ps(88.0)),
        _mm256_set1_ps(-87.3),
    );
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let ln2_hi = _mm256_set1_ps(0.693145751953125);
    let ln2_lo = _mm256_set1_ps(1.428606765330187e-6);

    let n = _mm256_round_ps(
        _mm256_mul_ps(y, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm256_fnmadd_ps(n, ln2_lo, _mm256_fnmadd_ps(n, ln2_hi, y));

    let c5 = _mm256_set1_ps(0.008333334);
    let c4 = _mm256_set1_ps(0.041666668);
    let c3 = _mm256_set1_ps(0.16666667);
    let c2 = _mm256_set1_ps(0.5);
    let c1 = _mm256_set1_ps(1.0);

    let mut p = _mm256_fmadd_ps(r, c5, c4);
    p = _mm256_fmadd_ps(r, p, c3);
    p = _mm256_fmadd_ps(r, p, c2);
    p = _mm256_fmadd_ps(r, p, c1);
    p = _mm256_fmadd_ps(r, p, c1);

    let ni = _mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127));
    let pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(ni, 23));
    _mm256_mul_ps(p, pow2n)
}

/// GELU (rational approximation, inputs clamped to `[-4.8, 4.8]`).
///
/// # Safety
/// The CPU must support AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn avx2_gelu_f32(x: __m256) -> __m256 {
    let xc = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-4.8)), _mm256_set1_ps(4.8));
    let u = _mm256_mul_ps(xc, xc);

    let mut num = _mm256_fmadd_ps(u, _mm256_set1_ps(6.4292e-5), _mm256_set1_ps(0.007572631));
    num = _mm256_fmadd_ps(u, num, _mm256_set1_ps(0.055282866));
    num = _mm256_fmadd_ps(u, num, _mm256_set1_ps(0.797882463));

    let mut den = _mm256_fmadd_ps(u, _mm256_set1_ps(0.001139532), _mm256_set1_ps(0.02381135));
    den = _mm256_fmadd_ps(u, den, _mm256_set1_ps(0.235943093));
    den = _mm256_fmadd_ps(u, den, _mm256_set1_ps(1.0));

    let est = _mm256_rcp_ps(den);
    let inv_den = _mm256_mul_ps(est, _mm256_fnmadd_ps(den, est, _mm256_set1_ps(2.0)));
    let erf_approx = _mm256_min_ps(
        _mm256_max_ps(
            _mm256_mul_ps(xc, _mm256_mul_ps(num, inv_den)),
            _mm256_set1_ps(-1.0),
        ),
        _mm256_set1_ps(1.0),
    );
    let half_x = _mm256_mul_ps(x, _mm256_set1_ps(0.5));
    _mm256_mul_ps(half_x, _mm256_add_ps(_mm256_set1_ps(1.0), erf_approx))
}

/// Fused in-place LayerNorm + SmoothQuant scaling + Dynamic Int8 quantization for a single row.
/// `x_row` is updated in-place to `LayerNorm(x_row)` (preserving the exact FP32 residual),
/// while `out_q` receives `round_clip(LayerNorm(x_row) * inv_smooth / scale, -127, 127)`.
#[inline]
pub fn layernorm_smooth_quantize_row(
    x_row: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    inv_smooth: &[f32],
    out_q: &mut [i8],
    eps: f32,
) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe {
            layernorm_smooth_quantize_row_neon(x_row, gamma, beta, inv_smooth, out_q, eps)
        };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe {
                layernorm_smooth_quantize_row_avx2(x_row, gamma, beta, inv_smooth, out_q, eps)
            };
        }
    }
    #[allow(unreachable_code)]
    layernorm_smooth_quantize_row_scalar(x_row, gamma, beta, inv_smooth, out_q, eps)
}

#[inline]
pub fn layernorm_smooth_quantize_row_scalar(
    x_row: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    inv_smooth: &[f32],
    out_q: &mut [i8],
    eps: f32,
) -> f32 {
    let n = x_row.len();
    let mut sum = 0.0f32;
    for &v in x_row.iter() {
        sum += v;
    }
    let mean = sum / (n as f32);
    let mut var_sum = 0.0f32;
    for &v in x_row.iter() {
        let d = v - mean;
        var_sum += d * d;
    }
    let rstd = 1.0 / (var_sum / (n as f32) + eps).sqrt();

    let mut max_abs = 1e-8f32;
    for j in 0..n {
        let y = (x_row[j] - mean) * rstd * gamma[j] + beta[j];
        x_row[j] = y;
        let a = (y * inv_smooth[j]).abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let scale = max_abs / 127.0;
    let inv_scale = 127.0 / max_abs;
    for j in 0..n {
        let v = (x_row[j] * inv_smooth[j] * inv_scale)
            .round()
            .clamp(-127.0, 127.0);
        out_q[j] = v as i8;
    }
    scale
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn layernorm_smooth_quantize_row_neon(
    x_row: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    inv_smooth: &[f32],
    out_q: &mut [i8],
    eps: f32,
) -> f32 {
    let n = x_row.len();
    debug_assert_eq!(n % 16, 0);
    let x_ptr = x_row.as_mut_ptr();
    let g_ptr = gamma.as_ptr();
    let b_ptr = beta.as_ptr();
    let s_ptr = inv_smooth.as_ptr();
    let q_ptr = out_q.as_mut_ptr();

    let mut sum0 = vdupq_n_f32(0.0);
    let mut sum1 = vdupq_n_f32(0.0);
    let mut sum2 = vdupq_n_f32(0.0);
    let mut sum3 = vdupq_n_f32(0.0);
    let mut i = 0;
    while i < n {
        sum0 = vaddq_f32(sum0, vld1q_f32(x_ptr.add(i)));
        sum1 = vaddq_f32(sum1, vld1q_f32(x_ptr.add(i + 4)));
        sum2 = vaddq_f32(sum2, vld1q_f32(x_ptr.add(i + 8)));
        sum3 = vaddq_f32(sum3, vld1q_f32(x_ptr.add(i + 12)));
        i += 16;
    }
    let mean = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3))) / (n as f32);
    let vmean = vdupq_n_f32(mean);

    let mut vsum0 = vdupq_n_f32(0.0);
    let mut vsum1 = vdupq_n_f32(0.0);
    let mut vsum2 = vdupq_n_f32(0.0);
    let mut vsum3 = vdupq_n_f32(0.0);
    i = 0;
    while i < n {
        let d0 = vsubq_f32(vld1q_f32(x_ptr.add(i)), vmean);
        let d1 = vsubq_f32(vld1q_f32(x_ptr.add(i + 4)), vmean);
        let d2 = vsubq_f32(vld1q_f32(x_ptr.add(i + 8)), vmean);
        let d3 = vsubq_f32(vld1q_f32(x_ptr.add(i + 12)), vmean);
        vsum0 = vfmaq_f32(vsum0, d0, d0);
        vsum1 = vfmaq_f32(vsum1, d1, d1);
        vsum2 = vfmaq_f32(vsum2, d2, d2);
        vsum3 = vfmaq_f32(vsum3, d3, d3);
        i += 16;
    }
    let var = vaddvq_f32(vaddq_f32(vaddq_f32(vsum0, vsum1), vaddq_f32(vsum2, vsum3))) / (n as f32);
    let vrstd = vdupq_n_f32(1.0 / (var + eps).sqrt());

    let mut vmax0 = vdupq_n_f32(1e-8);
    let mut vmax1 = vdupq_n_f32(1e-8);
    // Uninitialized scratch: every lane is written before it is read (zero-filling it cost ~4% of runtime).
    debug_assert!(n <= 384);
    let mut ys_buf = std::mem::MaybeUninit::<[f32; 384]>::uninit();
    let ys_ptr = ys_buf.as_mut_ptr() as *mut f32;
    i = 0;
    while i < n {
        let y0 = vfmaq_f32(
            vld1q_f32(b_ptr.add(i)),
            vmulq_f32(vsubq_f32(vld1q_f32(x_ptr.add(i)), vmean), vrstd),
            vld1q_f32(g_ptr.add(i)),
        );
        let y1 = vfmaq_f32(
            vld1q_f32(b_ptr.add(i + 4)),
            vmulq_f32(vsubq_f32(vld1q_f32(x_ptr.add(i + 4)), vmean), vrstd),
            vld1q_f32(g_ptr.add(i + 4)),
        );
        let y2 = vfmaq_f32(
            vld1q_f32(b_ptr.add(i + 8)),
            vmulq_f32(vsubq_f32(vld1q_f32(x_ptr.add(i + 8)), vmean), vrstd),
            vld1q_f32(g_ptr.add(i + 8)),
        );
        let y3 = vfmaq_f32(
            vld1q_f32(b_ptr.add(i + 12)),
            vmulq_f32(vsubq_f32(vld1q_f32(x_ptr.add(i + 12)), vmean), vrstd),
            vld1q_f32(g_ptr.add(i + 12)),
        );
        vst1q_f32(x_ptr.add(i), y0);
        vst1q_f32(x_ptr.add(i + 4), y1);
        vst1q_f32(x_ptr.add(i + 8), y2);
        vst1q_f32(x_ptr.add(i + 12), y3);

        let ys0 = vmulq_f32(y0, vld1q_f32(s_ptr.add(i)));
        let ys1 = vmulq_f32(y1, vld1q_f32(s_ptr.add(i + 4)));
        let ys2 = vmulq_f32(y2, vld1q_f32(s_ptr.add(i + 8)));
        let ys3 = vmulq_f32(y3, vld1q_f32(s_ptr.add(i + 12)));
        vst1q_f32(ys_ptr.add(i), ys0);
        vst1q_f32(ys_ptr.add(i + 4), ys1);
        vst1q_f32(ys_ptr.add(i + 8), ys2);
        vst1q_f32(ys_ptr.add(i + 12), ys3);
        vmax0 = vmaxq_f32(vmax0, vmaxq_f32(vabsq_f32(ys0), vabsq_f32(ys1)));
        vmax1 = vmaxq_f32(vmax1, vmaxq_f32(vabsq_f32(ys2), vabsq_f32(ys3)));
        i += 16;
    }

    let max_abs = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
    let scale = max_abs / 127.0;
    let vinv_scale = vdupq_n_f32(127.0 / max_abs);

    i = 0;
    while i < n {
        let q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(ys_ptr.add(i)), vinv_scale));
        let q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(ys_ptr.add(i + 4)), vinv_scale));
        let q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(ys_ptr.add(i + 8)), vinv_scale));
        let q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(ys_ptr.add(i + 12)), vinv_scale));

        let q01 = vqmovn_high_s32(vqmovn_s32(q0), q1);
        let q23 = vqmovn_high_s32(vqmovn_s32(q2), q3);
        let q_bytes = vqmovn_high_s16(vqmovn_s16(q01), q23);
        vst1q_s8(q_ptr.add(i), q_bytes);
        i += 16;
    }
    scale
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layernorm_smooth_quantize_row_avx2(
    x_row: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    inv_smooth: &[f32],
    out_q: &mut [i8],
    eps: f32,
) -> f32 {
    let n = x_row.len();
    debug_assert_eq!(n % 16, 0);
    let x_ptr = x_row.as_mut_ptr();
    let g_ptr = gamma.as_ptr();
    let b_ptr = beta.as_ptr();
    let s_ptr = inv_smooth.as_ptr();
    let q_ptr = out_q.as_mut_ptr();

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0;
    while i < n {
        sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(x_ptr.add(i)));
        sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(x_ptr.add(i + 8)));
        i += 16;
    }
    let mean = hsum256_ps(_mm256_add_ps(sum0, sum1)) / (n as f32);
    let vmean = _mm256_set1_ps(mean);

    let mut vsum0 = _mm256_setzero_ps();
    let mut vsum1 = _mm256_setzero_ps();
    i = 0;
    while i < n {
        let d0 = _mm256_sub_ps(_mm256_loadu_ps(x_ptr.add(i)), vmean);
        let d1 = _mm256_sub_ps(_mm256_loadu_ps(x_ptr.add(i + 8)), vmean);
        vsum0 = _mm256_fmadd_ps(d0, d0, vsum0);
        vsum1 = _mm256_fmadd_ps(d1, d1, vsum1);
        i += 16;
    }
    let var = hsum256_ps(_mm256_add_ps(vsum0, vsum1)) / (n as f32);
    let vrstd = _mm256_set1_ps(1.0 / (var + eps).sqrt());
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut vmax = _mm256_set1_ps(1e-8);

    i = 0;
    while i < n {
        let y0 = _mm256_fmadd_ps(
            _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x_ptr.add(i)), vmean), vrstd),
            _mm256_loadu_ps(g_ptr.add(i)),
            _mm256_loadu_ps(b_ptr.add(i)),
        );
        let y1 = _mm256_fmadd_ps(
            _mm256_mul_ps(
                _mm256_sub_ps(_mm256_loadu_ps(x_ptr.add(i + 8)), vmean),
                vrstd,
            ),
            _mm256_loadu_ps(g_ptr.add(i + 8)),
            _mm256_loadu_ps(b_ptr.add(i + 8)),
        );
        _mm256_storeu_ps(x_ptr.add(i), y0);
        _mm256_storeu_ps(x_ptr.add(i + 8), y1);

        let ys0 = _mm256_andnot_ps(sign_mask, _mm256_mul_ps(y0, _mm256_loadu_ps(s_ptr.add(i))));
        let ys1 = _mm256_andnot_ps(
            sign_mask,
            _mm256_mul_ps(y1, _mm256_loadu_ps(s_ptr.add(i + 8))),
        );
        vmax = _mm256_max_ps(vmax, _mm256_max_ps(ys0, ys1));
        i += 16;
    }

    let max_abs = hmax256_ps(vmax);
    let scale = max_abs / 127.0;
    let vinv_scale = _mm256_set1_ps(127.0 / max_abs);
    let vmin = _mm256_set1_ps(-127.0);
    let vmax_c = _mm256_set1_ps(127.0);

    i = 0;
    while i < n {
        let v0 = _mm256_min_ps(
            _mm256_max_ps(
                _mm256_mul_ps(
                    _mm256_loadu_ps(x_ptr.add(i)),
                    _mm256_mul_ps(_mm256_loadu_ps(s_ptr.add(i)), vinv_scale),
                ),
                vmin,
            ),
            vmax_c,
        );
        let v1 = _mm256_min_ps(
            _mm256_max_ps(
                _mm256_mul_ps(
                    _mm256_loadu_ps(x_ptr.add(i + 8)),
                    _mm256_mul_ps(_mm256_loadu_ps(s_ptr.add(i + 8)), vinv_scale),
                ),
                vmin,
            ),
            vmax_c,
        );
        let q0 = _mm256_cvtps_epi32(v0);
        let q1 = _mm256_cvtps_epi32(v1);
        let q16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), 0xD8);
        let q8 = _mm_packs_epi16(
            _mm256_castsi256_si128(q16),
            _mm256_extracti128_si256(q16, 1),
        );
        _mm_storeu_si128(q_ptr.add(i) as *mut __m128i, q8);
        i += 16;
    }
    scale
}

/// In-place LayerNorm without quantization (used at the end of Layer 10 and Layer 11).
#[inline]
pub fn layernorm_inplace_row(x_row: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) {
    let n = x_row.len();
    let mut sum = 0.0f32;
    for &v in x_row.iter() {
        sum += v;
    }
    let mean = sum / (n as f32);
    let mut var_sum = 0.0f32;
    for &v in x_row.iter() {
        let d = v - mean;
        var_sum += d * d;
    }
    let rstd = 1.0 / (var_sum / (n as f32) + eps).sqrt();
    for j in 0..n {
        x_row[j] = (x_row[j] - mean) * rstd * gamma[j] + beta[j];
    }
}

/// SmoothQuant scaling + Dynamic Int8 quantization for a single row (`[384]`).
#[inline]
pub fn smooth_quantize_row(x_row: &[f32], inv_smooth: &[f32], out_q: &mut [i8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { smooth_quantize_row_neon(x_row, inv_smooth, out_q) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { smooth_quantize_row_avx2(x_row, inv_smooth, out_q) };
        }
    }
    #[allow(unreachable_code)]
    {
        let n = x_row.len();
        let mut max_abs = 1e-8f32;
        for j in 0..n {
            let a = (x_row[j] * inv_smooth[j]).abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = max_abs / 127.0;
        let inv_scale = 127.0 / max_abs;
        for j in 0..n {
            out_q[j] = (x_row[j] * inv_smooth[j] * inv_scale)
                .round()
                .clamp(-127.0, 127.0) as i8;
        }
        scale
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn smooth_quantize_row_neon(x_row: &[f32], inv_smooth: &[f32], out_q: &mut [i8]) -> f32 {
    let n = x_row.len();
    debug_assert_eq!(n % 16, 0);
    let x_ptr = x_row.as_ptr();
    let s_ptr = inv_smooth.as_ptr();
    let q_ptr = out_q.as_mut_ptr();

    let mut vmax0 = vdupq_n_f32(1e-8);
    let mut vmax1 = vdupq_n_f32(1e-8);
    // Uninitialized scratch: every lane is written before it is read (zero-filling it cost ~4% of runtime).
    debug_assert!(n <= 384);
    let mut ys_buf = std::mem::MaybeUninit::<[f32; 384]>::uninit();
    let ys_ptr = ys_buf.as_mut_ptr() as *mut f32;
    let mut i = 0;
    while i < n {
        let ys0 = vmulq_f32(vld1q_f32(x_ptr.add(i)), vld1q_f32(s_ptr.add(i)));
        let ys1 = vmulq_f32(vld1q_f32(x_ptr.add(i + 4)), vld1q_f32(s_ptr.add(i + 4)));
        let ys2 = vmulq_f32(vld1q_f32(x_ptr.add(i + 8)), vld1q_f32(s_ptr.add(i + 8)));
        let ys3 = vmulq_f32(vld1q_f32(x_ptr.add(i + 12)), vld1q_f32(s_ptr.add(i + 12)));
        vst1q_f32(ys_ptr.add(i), ys0);
        vst1q_f32(ys_ptr.add(i + 4), ys1);
        vst1q_f32(ys_ptr.add(i + 8), ys2);
        vst1q_f32(ys_ptr.add(i + 12), ys3);
        vmax0 = vmaxq_f32(vmax0, vmaxq_f32(vabsq_f32(ys0), vabsq_f32(ys1)));
        vmax1 = vmaxq_f32(vmax1, vmaxq_f32(vabsq_f32(ys2), vabsq_f32(ys3)));
        i += 16;
    }

    let max_abs = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
    let scale = max_abs / 127.0;
    let vinv_scale = vdupq_n_f32(127.0 / max_abs);

    i = 0;
    while i < n {
        let q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(ys_ptr.add(i)), vinv_scale));
        let q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(ys_ptr.add(i + 4)), vinv_scale));
        let q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(ys_ptr.add(i + 8)), vinv_scale));
        let q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(ys_ptr.add(i + 12)), vinv_scale));

        let q01 = vqmovn_high_s32(vqmovn_s32(q0), q1);
        let q23 = vqmovn_high_s32(vqmovn_s32(q2), q3);
        vst1q_s8(q_ptr.add(i), vqmovn_high_s16(vqmovn_s16(q01), q23));
        i += 16;
    }
    scale
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn smooth_quantize_row_avx2(x_row: &[f32], inv_smooth: &[f32], out_q: &mut [i8]) -> f32 {
    let n = x_row.len();
    debug_assert_eq!(n % 16, 0);
    let x_ptr = x_row.as_ptr();
    let s_ptr = inv_smooth.as_ptr();
    let q_ptr = out_q.as_mut_ptr();
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut vmax = _mm256_set1_ps(1e-8);

    let mut i = 0;
    while i < n {
        let ys0 = _mm256_andnot_ps(
            sign_mask,
            _mm256_mul_ps(_mm256_loadu_ps(x_ptr.add(i)), _mm256_loadu_ps(s_ptr.add(i))),
        );
        let ys1 = _mm256_andnot_ps(
            sign_mask,
            _mm256_mul_ps(
                _mm256_loadu_ps(x_ptr.add(i + 8)),
                _mm256_loadu_ps(s_ptr.add(i + 8)),
            ),
        );
        vmax = _mm256_max_ps(vmax, _mm256_max_ps(ys0, ys1));
        i += 16;
    }

    let max_abs = hmax256_ps(vmax);
    let scale = max_abs / 127.0;
    let vinv_scale = _mm256_set1_ps(127.0 / max_abs);
    let vmin = _mm256_set1_ps(-127.0);
    let vmax_c = _mm256_set1_ps(127.0);

    i = 0;
    while i < n {
        let v0 = _mm256_min_ps(
            _mm256_max_ps(
                _mm256_mul_ps(
                    _mm256_loadu_ps(x_ptr.add(i)),
                    _mm256_mul_ps(_mm256_loadu_ps(s_ptr.add(i)), vinv_scale),
                ),
                vmin,
            ),
            vmax_c,
        );
        let v1 = _mm256_min_ps(
            _mm256_max_ps(
                _mm256_mul_ps(
                    _mm256_loadu_ps(x_ptr.add(i + 8)),
                    _mm256_mul_ps(_mm256_loadu_ps(s_ptr.add(i + 8)), vinv_scale),
                ),
                vmin,
            ),
            vmax_c,
        );
        let q0 = _mm256_cvtps_epi32(v0);
        let q1 = _mm256_cvtps_epi32(v1);
        let q16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), 0xD8);
        let q8 = _mm_packs_epi16(
            _mm256_castsi256_si128(q16),
            _mm256_extracti128_si256(q16, 1),
        );
        _mm_storeu_si128(q_ptr.add(i) as *mut __m128i, q8);
        i += 16;
    }
    scale
}

/// Fused SIMD GELU + SmoothQuant scaling + Dynamic Int8 quantization for a batch of `m` rows (`[m, 1536]`).
/// Executes on the dedicated per-core 128-bit NEON / 256-bit AVX2 execution pipelines so it never contends
/// on the shared cluster SME matrix coprocessor.
#[inline]
pub fn gelu_smooth_quantize_batch(
    m: usize,
    inter_buf: &mut [f32],
    inv_smooth: &[f32],
    inter_q: &mut [i8],
    act_scale: &mut [f32],
    _use_sme: bool,
) {
    let n = inv_smooth.len();
    #[cfg(target_arch = "aarch64")]
    if n % 16 == 0 && n <= GELU_F16_MAX_N && has_fp16_arith() {
        unsafe { gelu_smooth_quantize_batch_f16(m, inter_buf, inv_smooth, inter_q, act_scale) };
        return;
    }
    for r in 0..m {
        let row = &mut inter_buf[r * n..(r + 1) * n];
        let q_row = &mut inter_q[r * n..(r + 1) * n];
        act_scale[r] = gelu_smooth_quantize_row(row, inv_smooth, q_row);
    }
}

#[cfg(target_arch = "aarch64")]
const GELU_F16_MAX_N: usize = 1536;

/// Runtime check for ARMv8.2 `FEAT_FP16` half-precision vector arithmetic (all Apple M-series,
/// Graviton2+, Ampere Altra, Cortex-A55/A76+; absent on e.g. Cortex-A72 / Raspberry Pi 4).
#[cfg(target_arch = "aarch64")]
#[inline]
fn has_fp16_arith() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(0); // 0 = unknown, 1 = no, 2 = yes
    match STATE.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let yes = std::arch::is_aarch64_feature_detected!("fp16");
            STATE.store(if yes { 2 } else { 1 }, Ordering::Relaxed);
            yes
        }
    }
}

/// Same rational GELU as `neon_gelu_pair_f32`, followed by SmoothQuant scale, as f32 constants that the
/// kernel narrows to f16 lanes: `[vmin, vmax, p3, p2, p1, p0, q3, q2, q1, one, half, pad]`.
#[cfg(target_arch = "aarch64")]
static GELU_F16_CONSTS: [f32; 12] = [
    -4.8,
    4.8,
    3.214524e-5,
    0.0037863153,
    0.027641433,
    0.39894123,
    0.001139532,
    0.02381135,
    0.2359431,
    1.0,
    0.5,
    0.0,
];

/// Half-precision (8 lanes per NEON op) fused GELU + SmoothQuant + dynamic Int8 quantization for `[m, n]`.
///
/// The intermediate GELU output is re-quantized to Int8 with a per-row scale right after, so fp16's
/// ~1e-3 relative precision is ~50x finer than one quantization step; the rational approximation's
/// coefficients are all positive (no cancellation), so evaluating it in fp16 is well conditioned.
/// Doubles the lanes per instruction vs f32 and halves the bytes of the intermediate pass, which this
/// writes as f16 into the first half of each (no longer needed) f32 input row.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn gelu_smooth_quantize_batch_f16(
    m: usize,
    inter_buf: &mut [f32],
    inv_smooth: &[f32],
    inter_q: &mut [i8],
    act_scale: &mut [f32],
) {
    let n = inv_smooth.len();
    debug_assert!(n % 16 == 0 && n <= GELU_F16_MAX_N);
    debug_assert!(inter_buf.len() >= m * n && inter_q.len() >= m * n && act_scale.len() >= m);

    let mut s16 = std::mem::MaybeUninit::<[u16; GELU_F16_MAX_N]>::uninit();
    let s16_ptr = s16.as_mut_ptr() as *mut u16;
    f32_to_f16_neon(inv_smooth.as_ptr(), s16_ptr, n);

    for r in 0..m {
        let x_row = inter_buf.as_mut_ptr().add(r * n);
        let max_abs = gelu_f16_pass1(x_row, s16_ptr, n);
        act_scale[r] = gelu_f16_pass2(x_row, n, max_abs, inter_q.as_mut_ptr().add(r * n));
    }
}

/// Converts `n` (multiple of 16) f32s to f16 bit patterns.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn f32_to_f16_neon(src: *const f32, dst: *mut u16, n: usize) {
    debug_assert!(n % 16 == 0 && n > 0);
    core::arch::asm!(
        "2:",
        "ld1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{src}], #64",
        "fcvtn v4.4h, v0.4s",
        "fcvtn2 v4.8h, v1.4s",
        "fcvtn v5.4h, v2.4s",
        "fcvtn2 v5.8h, v3.4s",
        "st1 {{v4.8h, v5.8h}}, [{dst}], #32",
        "subs {cnt}, {cnt}, #1",
        "b.ne 2b",
        src = inout(reg) src => _,
        dst = inout(reg) dst => _,
        cnt = inout(reg) n / 16 => _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
        options(nostack)
    );
}

/// GELU pass 1 in f16 over `n` (multiple of 16) contiguous f32 inputs at `x`: writes
/// `g = GELU(x) * inv_smooth` as f16 into the first `2n` bytes of the same memory and returns `max|g|`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn gelu_f16_pass1(x: *mut f32, s16: *const u16, n: usize) -> f32 {
    debug_assert!(n % 16 == 0 && n > 0);
    let mut max_bits: u32;
    core::arch::asm!(
        ".arch_extension fp16",
        "ld1 {{v16.4s, v17.4s, v18.4s}}, [{c}]",
        "fcvtn v29.4h, v16.4s", "fcvtn2 v29.8h, v16.4s",
        "fcvtn v30.4h, v17.4s", "fcvtn2 v30.8h, v17.4s",
        "fcvtn v31.4h, v18.4s", "fcvtn2 v31.8h, v18.4s",
        // v29 = [vmin, vmax, p3, p2], v30 = [p1, p0, q3, q2], v31 = [q1, one, half, pad] (x2)
        "dup v16.8h, v29.h[0]",
        "dup v17.8h, v29.h[1]",
        "dup v18.8h, v29.h[2]",
        "dup v19.8h, v29.h[3]",
        "dup v20.8h, v30.h[0]",
        "dup v21.8h, v30.h[1]",
        "dup v22.8h, v30.h[2]",
        "dup v23.8h, v30.h[3]",
        "dup v24.8h, v31.h[0]",
        "dup v25.8h, v31.h[1]",
        "dup v26.8h, v31.h[2]",
        "movi v27.8h, #0",
        "movi v28.8h, #0",

        "2:",
        "ld1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{x}], #64",
        "ld1 {{v6.8h, v7.8h}}, [{s}], #32",
        "fcvtn v4.4h, v0.4s",
        "fcvtn2 v4.8h, v1.4s",
        "fcvtn v5.4h, v2.4s",
        "fcvtn2 v5.8h, v3.4s",
        // xc = clamp(x, -4.8, 4.8); u = xc^2
        "fmax v0.8h, v4.8h, v16.8h",
        "fmax v1.8h, v5.8h, v16.8h",
        "fmin v0.8h, v0.8h, v17.8h",
        "fmin v1.8h, v1.8h, v17.8h",
        "fmul v2.8h, v0.8h, v0.8h",
        "fmul v3.8h, v1.8h, v1.8h",
        // N(u) = p0 + u*(p1 + u*(p2 + u*p3))
        "mov v8.16b, v19.16b",
        "mov v9.16b, v19.16b",
        "fmla v8.8h, v2.8h, v18.8h",
        "fmla v9.8h, v3.8h, v18.8h",
        "mov v10.16b, v20.16b",
        "mov v11.16b, v20.16b",
        "fmla v10.8h, v2.8h, v8.8h",
        "fmla v11.8h, v3.8h, v9.8h",
        "mov v8.16b, v21.16b",
        "mov v9.16b, v21.16b",
        "fmla v8.8h, v2.8h, v10.8h",
        "fmla v9.8h, v3.8h, v11.8h",
        // D(u) = 1 + u*(q1 + u*(q2 + u*q3))
        "mov v10.16b, v23.16b",
        "mov v11.16b, v23.16b",
        "fmla v10.8h, v2.8h, v22.8h",
        "fmla v11.8h, v3.8h, v22.8h",
        "mov v12.16b, v24.16b",
        "mov v13.16b, v24.16b",
        "fmla v12.8h, v2.8h, v10.8h",
        "fmla v13.8h, v3.8h, v11.8h",
        "mov v10.16b, v25.16b",
        "mov v11.16b, v25.16b",
        "fmla v10.8h, v2.8h, v12.8h",
        "fmla v11.8h, v3.8h, v13.8h",
        // 1/D with one Newton step
        "frecpe v12.8h, v10.8h",
        "frecpe v13.8h, v11.8h",
        "frecps v10.8h, v10.8h, v12.8h",
        "frecps v11.8h, v11.8h, v13.8h",
        "fmul v12.8h, v12.8h, v10.8h",
        "fmul v13.8h, v13.8h, v11.8h",
        "fmul v8.8h, v8.8h, v12.8h",
        "fmul v9.8h, v9.8h, v13.8h",
        // gate = 0.5 + xc*N/D ; g = x * gate * inv_smooth
        "mov v10.16b, v26.16b",
        "mov v11.16b, v26.16b",
        "fmla v10.8h, v0.8h, v8.8h",
        "fmla v11.8h, v1.8h, v9.8h",
        "fmul v4.8h, v4.8h, v10.8h",
        "fmul v5.8h, v5.8h, v11.8h",
        "fmul v4.8h, v4.8h, v6.8h",
        "fmul v5.8h, v5.8h, v7.8h",
        "st1 {{v4.8h, v5.8h}}, [{h}], #32",
        "fabs v4.8h, v4.8h",
        "fabs v5.8h, v5.8h",
        "fmax v27.8h, v27.8h, v4.8h",
        "fmax v28.8h, v28.8h, v5.8h",
        "subs {cnt}, {cnt}, #1",
        "b.ne 2b",

        "fmax v27.8h, v27.8h, v28.8h",
        "fmaxv h27, v27.8h",
        "fcvt s27, h27",
        "fmov {mx:w}, s27",
        c = in(reg) GELU_F16_CONSTS.as_ptr(),
        x = inout(reg) x as *const f32 => _,
        s = inout(reg) s16 => _,
        h = inout(reg) x as *mut u16 => _,
        cnt = inout(reg) n / 16 => _,
        mx = lateout(reg) max_bits,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
        out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _, out("v25") _,
        out("v26") _, out("v27") _, out("v28") _, out("v29") _, out("v30") _, out("v31") _,
        options(nostack)
    );
    f32::from_bits(max_bits)
}

/// GELU pass 2: quantizes the `n` f16 values written by [`gelu_f16_pass1`] at `x` (as f16) with the
/// row maximum `max_abs` into `q`, returning the row scale `max_abs / 127`. Rows whose maximum is tiny
/// or non-finite (where `127 / max` would leave f16 range) are quantized in f32 instead.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,fp16")]
unsafe fn gelu_f16_pass2(x: *mut f32, n: usize, max_abs: f32, q: *mut i8) -> f32 {
    let h = x as *const u16;
    if !(max_abs >= 1e-2 && max_abs.is_finite()) {
        let mx = if max_abs.is_finite() {
            max_abs.max(1e-8)
        } else {
            f32::MAX
        };
        let inv = 127.0 / mx;
        let mut j = 0;
        while j < n {
            let mut tmp = [0.0f32; 8];
            core::arch::asm!(
                "ld1 {{v0.8h}}, [{h}]",
                "fcvtl v1.4s, v0.4h",
                "fcvtl2 v2.4s, v0.8h",
                "st1 {{v1.4s, v2.4s}}, [{t}]",
                h = in(reg) h.add(j),
                t = in(reg) tmp.as_mut_ptr(),
                out("v0") _, out("v1") _, out("v2") _,
                options(nostack)
            );
            for k in 0..8 {
                let v = tmp[k] * inv;
                *q.add(j + k) = if v.is_nan() {
                    0
                } else {
                    v.round().clamp(-127.0, 127.0) as i8
                };
            }
            j += 8;
        }
        return mx / 127.0;
    }
    let inv_scale = 127.0f32 / max_abs;
    core::arch::asm!(
        ".arch_extension fp16",
        "fmov s29, {inv:w}",
        "fcvt h29, s29",
        "dup v29.8h, v29.h[0]",
        "2:",
        "ld1 {{v0.8h, v1.8h}}, [{h}], #32",
        "fmul v0.8h, v0.8h, v29.8h",
        "fmul v1.8h, v1.8h, v29.8h",
        "fcvtns v0.8h, v0.8h",
        "fcvtns v1.8h, v1.8h",
        "sqxtn v2.8b, v0.8h",
        "sqxtn2 v2.16b, v1.8h",
        "st1 {{v2.16b}}, [{q}], #16",
        "subs {cnt}, {cnt}, #1",
        "b.ne 2b",
        inv = in(reg) inv_scale.to_bits(),
        h = inout(reg) h => _,
        q = inout(reg) q => _,
        cnt = inout(reg) n / 16 => _,
        out("v0") _, out("v1") _, out("v2") _, out("v29") _,
        options(nostack)
    );
    max_abs / 127.0
}

/// Column-segment GELU for intra-op parallelism (`n_cols` multiple of 16, `<= 1536`), pass 1 over
/// `rows` rows of a row-major `[rows, row_stride]` f32 buffer starting at column `col0`:
/// computes `GELU(x) * inv_smooth[col0..]` in place (f16 when available) and writes each row's
/// partial `max|g|` into `row_max`. Finish with [`gelu_segment_quantize`] once every segment's
/// maxima are known.
///
/// # Safety
/// `x` must be valid for the addressed segment and no other thread may touch it concurrently.
#[cfg(target_arch = "aarch64")]
pub unsafe fn gelu_segment_pass1(
    rows: usize,
    x: *mut f32,
    row_stride: usize,
    col0: usize,
    n_cols: usize,
    inv_smooth: &[f32],
    row_max: &mut [f32],
) {
    debug_assert!(
        n_cols % 16 == 0 && n_cols <= GELU_F16_MAX_N && col0 + n_cols <= inv_smooth.len()
    );
    if has_fp16_arith() {
        let mut s16 = std::mem::MaybeUninit::<[u16; GELU_F16_MAX_N]>::uninit();
        let s16_ptr = s16.as_mut_ptr() as *mut u16;
        f32_to_f16_neon(inv_smooth.as_ptr().add(col0), s16_ptr, n_cols);
        for r in 0..rows {
            row_max[r] = gelu_f16_pass1(x.add(r * row_stride + col0), s16_ptr, n_cols);
        }
    } else {
        for r in 0..rows {
            let seg = std::slice::from_raw_parts_mut(x.add(r * row_stride + col0), n_cols);
            let mut mx = 0.0f32;
            let mut j = 0;
            while j < n_cols {
                let (g0, g1) = neon_gelu_pair_f32(
                    vld1q_f32(seg.as_ptr().add(j)),
                    vld1q_f32(seg.as_ptr().add(j + 4)),
                );
                let g0 = vmulq_f32(g0, vld1q_f32(inv_smooth.as_ptr().add(col0 + j)));
                let g1 = vmulq_f32(g1, vld1q_f32(inv_smooth.as_ptr().add(col0 + j + 4)));
                vst1q_f32(seg.as_mut_ptr().add(j), g0);
                vst1q_f32(seg.as_mut_ptr().add(j + 4), g1);
                mx = mx.max(vmaxvq_f32(vmaxq_f32(vabsq_f32(g0), vabsq_f32(g1))));
                j += 8;
            }
            row_max[r] = mx;
        }
    }
}

/// Pass 2 of [`gelu_segment_pass1`]: quantizes the segment with the full-row maxima `row_max_full`
/// into `q` (row-major `[rows, row_stride]` i8 at the same columns).
///
/// # Safety
/// Same contract as [`gelu_segment_pass1`]; `q` must be valid for the addressed segment.
#[cfg(target_arch = "aarch64")]
pub unsafe fn gelu_segment_quantize(
    rows: usize,
    x: *mut f32,
    row_stride: usize,
    col0: usize,
    n_cols: usize,
    row_max_full: &[f32],
    q: *mut i8,
) {
    let f16 = has_fp16_arith();
    for r in 0..rows {
        let xs = x.add(r * row_stride + col0);
        let qs = q.add(r * row_stride + col0);
        if f16 {
            gelu_f16_pass2(xs, n_cols, row_max_full[r], qs);
        } else {
            let mx = row_max_full[r].max(1e-8);
            let inv = vdupq_n_f32(127.0 / mx);
            let mut j = 0;
            while j < n_cols {
                let q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xs.add(j)), inv));
                let q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xs.add(j + 4)), inv));
                let q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xs.add(j + 8)), inv));
                let q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xs.add(j + 12)), inv));
                let q01 = vqmovn_high_s32(vqmovn_s32(q0), q1);
                let q23 = vqmovn_high_s32(vqmovn_s32(q2), q3);
                vst1q_s8(qs.add(j), vqmovn_high_s16(vqmovn_s16(q01), q23));
                j += 16;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
static GELU_SVE_CONSTS: [f32; 16] = [
    -4.8,         // [0]  vmin -> z20.s
    4.8,          // [1]  vmax -> z21.s
    3.214524e-5,  // [2]  p3   -> z22.s
    0.0037863153, // [3]  p2   -> z23.s
    0.027641433,  // [4]  p1   -> z24.s
    0.39894123,   // [5]  p0   -> z25.s
    0.001139532,  // [6]  q3   -> z26.s
    0.02381135,   // [7]  q2   -> z27.s
    0.2359431,    // [8]  q1   -> z28.s
    1.0,          // [9]  one  -> z29.s
    0.5,          // [10] half -> z30.s
    1e-8,         // [11] eps_max
    127.0,        // [12] qmax
    -127.0,       // [13] qmin
    0.0078740157, // [14] 1.0 / 127.0
    0.0,          // [15] pad
];

/// 512-bit Streaming SVE (`FEAT_SME`) batch GELU + SmoothQuant + Int8 dynamic quantizer for `[m, 1536]`.
/// Evaluates 64 Rational Padé GELUs in parallel across `z0..z19` with all 11 polynomial constants
/// pinned in `z20..z30`, then narrows 64 `i32`s to 64 `i8`s via 3 `uzp1` instructions per 64-column block.
///
/// # Safety
/// The CPU must support `FEAT_SME` (callers dispatch on [`crate::gemm::SimdBackend::ArmSme`]); all
/// slices must hold at least the sizes implied by `m`/`seq_len` and the model dimensions.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub unsafe fn gelu_smooth_quantize_batch_sme(
    m: usize,
    inter_buf: &mut [f32],
    inv_smooth: &[f32],
    inter_q: &mut [i8],
    act_scale: &mut [f32],
) {
    let x_row = inter_buf.as_mut_ptr();
    let s_base = inv_smooth.as_ptr();
    let pq = inter_q.as_mut_ptr();
    let pscale = act_scale.as_mut_ptr();
    let c_ptr = GELU_SVE_CONSTS.as_ptr();

    core::arch::asm!(
        ".arch_extension sme",
        "smstart",
        "ptrue p0.b",
        "ptrue p1.s",
        "ld1rw {{z20.s}}, p1/z, [{c_ptr}, #0]",
        "ld1rw {{z21.s}}, p1/z, [{c_ptr}, #4]",
        "ld1rw {{z22.s}}, p1/z, [{c_ptr}, #8]",
        "ld1rw {{z23.s}}, p1/z, [{c_ptr}, #12]",
        "ld1rw {{z24.s}}, p1/z, [{c_ptr}, #16]",
        "ld1rw {{z25.s}}, p1/z, [{c_ptr}, #20]",
        "ld1rw {{z26.s}}, p1/z, [{c_ptr}, #24]",
        "ld1rw {{z27.s}}, p1/z, [{c_ptr}, #28]",
        "ld1rw {{z28.s}}, p1/z, [{c_ptr}, #32]",
        "ld1rw {{z29.s}}, p1/z, [{c_ptr}, #36]",
        "ld1rw {{z30.s}}, p1/z, [{c_ptr}, #40]",

        "1:",
        "ld1rw {{z31.s}}, p1/z, [{c_ptr}, #44]",
        "mov {px}, {x_row}",
        "mov {ps}, {s_base}",
        "mov {blk_cnt}, #24",

        "2:",
        "ld1w {{z0.s}}, p1/z, [{px}]",
        "ld1w {{z1.s}}, p1/z, [{px}, #1, mul vl]",
        "ld1w {{z2.s}}, p1/z, [{px}, #2, mul vl]",
        "ld1w {{z3.s}}, p1/z, [{px}, #3, mul vl]",

        "mov z4.d, z0.d",
        "mov z5.d, z1.d",
        "mov z6.d, z2.d",
        "mov z7.d, z3.d",
        "fmaxnm z4.s, p1/m, z4.s, z20.s",
        "fmaxnm z5.s, p1/m, z5.s, z20.s",
        "fmaxnm z6.s, p1/m, z6.s, z20.s",
        "fmaxnm z7.s, p1/m, z7.s, z20.s",
        "fminnm z4.s, p1/m, z4.s, z21.s",
        "fminnm z5.s, p1/m, z5.s, z21.s",
        "fminnm z6.s, p1/m, z6.s, z21.s",
        "fminnm z7.s, p1/m, z7.s, z21.s",

        "fmul z8.s, z4.s, z4.s",
        "fmul z9.s, z5.s, z5.s",
        "fmul z10.s, z6.s, z6.s",
        "fmul z11.s, z7.s, z7.s",

        "mov z12.d, z22.d",
        "mov z13.d, z22.d",
        "mov z14.d, z22.d",
        "mov z15.d, z22.d",
        "fmad z12.s, p1/m, z8.s, z23.s",
        "fmad z13.s, p1/m, z9.s, z23.s",
        "fmad z14.s, p1/m, z10.s, z23.s",
        "fmad z15.s, p1/m, z11.s, z23.s",
        "fmad z12.s, p1/m, z8.s, z24.s",
        "fmad z13.s, p1/m, z9.s, z24.s",
        "fmad z14.s, p1/m, z10.s, z24.s",
        "fmad z15.s, p1/m, z11.s, z24.s",
        "fmad z12.s, p1/m, z8.s, z25.s",
        "fmad z13.s, p1/m, z9.s, z25.s",
        "fmad z14.s, p1/m, z10.s, z25.s",
        "fmad z15.s, p1/m, z11.s, z25.s",

        "mov z16.d, z26.d",
        "mov z17.d, z26.d",
        "mov z18.d, z26.d",
        "mov z19.d, z26.d",
        "fmad z16.s, p1/m, z8.s, z27.s",
        "fmad z17.s, p1/m, z9.s, z27.s",
        "fmad z18.s, p1/m, z10.s, z27.s",
        "fmad z19.s, p1/m, z11.s, z27.s",
        "fmad z16.s, p1/m, z8.s, z28.s",
        "fmad z17.s, p1/m, z9.s, z28.s",
        "fmad z18.s, p1/m, z10.s, z28.s",
        "fmad z19.s, p1/m, z11.s, z28.s",
        "fmad z16.s, p1/m, z8.s, z29.s",
        "fmad z17.s, p1/m, z9.s, z29.s",
        "fmad z18.s, p1/m, z10.s, z29.s",
        "fmad z19.s, p1/m, z11.s, z29.s",

        "frecpe z8.s, z16.s",
        "frecpe z9.s, z17.s",
        "frecpe z10.s, z18.s",
        "frecpe z11.s, z19.s",
        "frecps z16.s, z16.s, z8.s",
        "frecps z17.s, z17.s, z9.s",
        "frecps z18.s, z18.s, z10.s",
        "frecps z19.s, z19.s, z11.s",
        "fmul z8.s, z8.s, z16.s",
        "fmul z9.s, z9.s, z17.s",
        "fmul z10.s, z10.s, z18.s",
        "fmul z11.s, z11.s, z19.s",

        "fmul z12.s, z12.s, z8.s",
        "fmul z13.s, z13.s, z9.s",
        "fmul z14.s, z14.s, z10.s",
        "fmul z15.s, z15.s, z11.s",
        "fmad z4.s, p1/m, z12.s, z30.s",
        "fmad z5.s, p1/m, z13.s, z30.s",
        "fmad z6.s, p1/m, z14.s, z30.s",
        "fmad z7.s, p1/m, z15.s, z30.s",
        "fmul z0.s, z0.s, z4.s",
        "fmul z1.s, z1.s, z5.s",
        "fmul z2.s, z2.s, z6.s",
        "fmul z3.s, z3.s, z7.s",

        "ld1w {{z4.s}}, p1/z, [{ps}]",
        "ld1w {{z5.s}}, p1/z, [{ps}, #1, mul vl]",
        "ld1w {{z6.s}}, p1/z, [{ps}, #2, mul vl]",
        "ld1w {{z7.s}}, p1/z, [{ps}, #3, mul vl]",
        "add {ps}, {ps}, #256",
        "fmul z0.s, z0.s, z4.s",
        "fmul z1.s, z1.s, z5.s",
        "fmul z2.s, z2.s, z6.s",
        "fmul z3.s, z3.s, z7.s",
        "st1w {{z0.s}}, p1, [{px}]",
        "st1w {{z1.s}}, p1, [{px}, #1, mul vl]",
        "st1w {{z2.s}}, p1, [{px}, #2, mul vl]",
        "st1w {{z3.s}}, p1, [{px}, #3, mul vl]",
        "add {px}, {px}, #256",

        "fabs z8.s, p1/m, z0.s",
        "fabs z9.s, p1/m, z1.s",
        "fabs z10.s, p1/m, z2.s",
        "fabs z11.s, p1/m, z3.s",
        "fmaxnm z8.s, p1/m, z8.s, z9.s",
        "fmaxnm z10.s, p1/m, z10.s, z11.s",
        "fmaxnm z8.s, p1/m, z8.s, z10.s",
        "fmaxnm z31.s, p1/m, z31.s, z8.s",
        "subs {blk_cnt}, {blk_cnt}, #1",
        "b.ne 2b",

        "fmaxnmv s4, p1, z31.s",
        "ld1rw {{z18.s}}, p1/z, [{c_ptr}, #48]",
        "ld1rw {{z19.s}}, p1/z, [{c_ptr}, #52]",
        "ld1rw {{z17.s}}, p1/z, [{c_ptr}, #56]",
        "fmul s5, s4, s17",
        "str s5, [{pscale}], #4",
        "fdiv s4, s18, s4",
        "dup z4.s, z4.s[0]",

        "mov {blk_cnt}, #24",
        "3:",
        "ld1w {{z0.s}}, p1/z, [{x_row}]",
        "ld1w {{z1.s}}, p1/z, [{x_row}, #1, mul vl]",
        "ld1w {{z2.s}}, p1/z, [{x_row}, #2, mul vl]",
        "ld1w {{z3.s}}, p1/z, [{x_row}, #3, mul vl]",
        "add {x_row}, {x_row}, #256",
        "fmul z0.s, z0.s, z4.s",
        "fmul z1.s, z1.s, z4.s",
        "fmul z2.s, z2.s, z4.s",
        "fmul z3.s, z3.s, z4.s",
        "fmaxnm z0.s, p1/m, z0.s, z19.s",
        "fmaxnm z1.s, p1/m, z1.s, z19.s",
        "fmaxnm z2.s, p1/m, z2.s, z19.s",
        "fmaxnm z3.s, p1/m, z3.s, z19.s",
        "fminnm z0.s, p1/m, z0.s, z18.s",
        "fminnm z1.s, p1/m, z1.s, z18.s",
        "fminnm z2.s, p1/m, z2.s, z18.s",
        "fminnm z3.s, p1/m, z3.s, z18.s",
        "frintn z0.s, p1/m, z0.s",
        "frintn z1.s, p1/m, z1.s",
        "frintn z2.s, p1/m, z2.s",
        "frintn z3.s, p1/m, z3.s",
        "fcvtzs z0.s, p1/m, z0.s",
        "fcvtzs z1.s, p1/m, z1.s",
        "fcvtzs z2.s, p1/m, z2.s",
        "fcvtzs z3.s, p1/m, z3.s",
        "uzp1 z0.h, z0.h, z1.h",
        "uzp1 z2.h, z2.h, z3.h",
        "uzp1 z0.b, z0.b, z2.b",
        "st1b {{z0.b}}, p0, [{pq}]",
        "add {pq}, {pq}, #64",
        "subs {blk_cnt}, {blk_cnt}, #1",
        "b.ne 3b",

        "subs {r_cnt}, {r_cnt}, #1",
        "b.ne 1b",
        "smstop",
        x_row = inout(reg) x_row => _,
        s_base = inout(reg) s_base => _,
        pq = inout(reg) pq => _,
        pscale = inout(reg) pscale => _,
        c_ptr = inout(reg) c_ptr => _,
        r_cnt = inout(reg) m => _,
        px = lateout(reg) _,
        ps = lateout(reg) _,
        blk_cnt = lateout(reg) _,
        out("v0") _, out("v1") _, out("v2") _, out("v3") _,
        out("v4") _, out("v5") _, out("v6") _, out("v7") _,
        out("v8") _, out("v9") _, out("v10") _, out("v11") _,
        out("v12") _, out("v13") _, out("v14") _, out("v15") _,
        out("v16") _, out("v17") _, out("v18") _, out("v19") _,
        out("v20") _, out("v21") _, out("v22") _, out("v23") _,
        out("v24") _, out("v25") _, out("v26") _, out("v27") _,
        out("v28") _, out("v29") _, out("v30") _, out("v31") _,
        options(nostack)
    );
}

/// Fused SIMD GELU + SmoothQuant scaling + Dynamic Int8 quantization for a single row (`[1536]`).
#[inline]
pub fn gelu_smooth_quantize_row(x_row: &mut [f32], inv_smooth: &[f32], out_q: &mut [i8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { gelu_smooth_quantize_row_neon(x_row, inv_smooth, out_q) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { gelu_smooth_quantize_row_avx2(x_row, inv_smooth, out_q) };
        }
    }
    #[allow(unreachable_code)]
    {
        let n = x_row.len();
        let mut max_abs = 1e-8f32;
        for j in 0..n {
            let g = scalar_gelu(x_row[j]) * inv_smooth[j];
            x_row[j] = g;
            let a = g.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = max_abs / 127.0;
        let inv_scale = 127.0 / max_abs;
        for j in 0..n {
            out_q[j] = (x_row[j] * inv_scale).round().clamp(-127.0, 127.0) as i8;
        }
        scale
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gelu_smooth_quantize_row_neon(
    x_row: &mut [f32],
    inv_smooth: &[f32],
    out_q: &mut [i8],
) -> f32 {
    let n = x_row.len();
    debug_assert_eq!(n % 16, 0);
    let x_ptr = x_row.as_mut_ptr();
    let s_ptr = inv_smooth.as_ptr();
    let q_ptr = out_q.as_mut_ptr();

    let mut vmax0 = vdupq_n_f32(1e-8);
    let mut vmax1 = vdupq_n_f32(1e-8);
    let mut i = 0;
    while i < n {
        let (g0_raw, g1_raw) =
            neon_gelu_pair_f32(vld1q_f32(x_ptr.add(i)), vld1q_f32(x_ptr.add(i + 4)));
        let g0 = vmulq_f32(g0_raw, vld1q_f32(s_ptr.add(i)));
        let g1 = vmulq_f32(g1_raw, vld1q_f32(s_ptr.add(i + 4)));
        vst1q_f32(x_ptr.add(i), g0);
        vst1q_f32(x_ptr.add(i + 4), g1);
        vmax0 = vmaxq_f32(vmax0, vmaxq_f32(vabsq_f32(g0), vabsq_f32(g1)));

        let (g2_raw, g3_raw) =
            neon_gelu_pair_f32(vld1q_f32(x_ptr.add(i + 8)), vld1q_f32(x_ptr.add(i + 12)));
        let g2 = vmulq_f32(g2_raw, vld1q_f32(s_ptr.add(i + 8)));
        let g3 = vmulq_f32(g3_raw, vld1q_f32(s_ptr.add(i + 12)));
        vst1q_f32(x_ptr.add(i + 8), g2);
        vst1q_f32(x_ptr.add(i + 12), g3);
        vmax1 = vmaxq_f32(vmax1, vmaxq_f32(vabsq_f32(g2), vabsq_f32(g3)));
        i += 16;
    }

    let max_abs = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
    let scale = max_abs / 127.0;
    let vinv_scale = vdupq_n_f32(127.0 / max_abs);

    i = 0;
    while i < n {
        let q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x_ptr.add(i)), vinv_scale));
        let q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x_ptr.add(i + 4)), vinv_scale));
        let q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x_ptr.add(i + 8)), vinv_scale));
        let q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x_ptr.add(i + 12)), vinv_scale));

        let q01 = vqmovn_high_s32(vqmovn_s32(q0), q1);
        let q23 = vqmovn_high_s32(vqmovn_s32(q2), q3);
        vst1q_s8(q_ptr.add(i), vqmovn_high_s16(vqmovn_s16(q01), q23));
        i += 16;
    }
    scale
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_smooth_quantize_row_avx2(
    x_row: &mut [f32],
    inv_smooth: &[f32],
    out_q: &mut [i8],
) -> f32 {
    let n = x_row.len();
    debug_assert_eq!(n % 16, 0);
    let x_ptr = x_row.as_mut_ptr();
    let s_ptr = inv_smooth.as_ptr();
    let q_ptr = out_q.as_mut_ptr();
    let sign_mask = _mm256_set1_ps(-0.0);
    let mut vmax = _mm256_set1_ps(1e-8);

    let mut i = 0;
    while i < n {
        let g0 = _mm256_mul_ps(
            avx2_gelu_f32(_mm256_loadu_ps(x_ptr.add(i))),
            _mm256_loadu_ps(s_ptr.add(i)),
        );
        let g1 = _mm256_mul_ps(
            avx2_gelu_f32(_mm256_loadu_ps(x_ptr.add(i + 8))),
            _mm256_loadu_ps(s_ptr.add(i + 8)),
        );
        _mm256_storeu_ps(x_ptr.add(i), g0);
        _mm256_storeu_ps(x_ptr.add(i + 8), g1);
        let a0 = _mm256_andnot_ps(sign_mask, g0);
        let a1 = _mm256_andnot_ps(sign_mask, g1);
        vmax = _mm256_max_ps(vmax, _mm256_max_ps(a0, a1));
        i += 16;
    }

    let max_abs = hmax256_ps(vmax);
    let scale = max_abs / 127.0;
    let vinv_scale = _mm256_set1_ps(127.0 / max_abs);
    let vmin = _mm256_set1_ps(-127.0);
    let vmax_c = _mm256_set1_ps(127.0);

    i = 0;
    while i < n {
        let v0 = _mm256_min_ps(
            _mm256_max_ps(
                _mm256_mul_ps(_mm256_loadu_ps(x_ptr.add(i)), vinv_scale),
                vmin,
            ),
            vmax_c,
        );
        let v1 = _mm256_min_ps(
            _mm256_max_ps(
                _mm256_mul_ps(_mm256_loadu_ps(x_ptr.add(i + 8)), vinv_scale),
                vmin,
            ),
            vmax_c,
        );
        let q0 = _mm256_cvtps_epi32(v0);
        let q1 = _mm256_cvtps_epi32(v1);
        let q16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(q0, q1), 0xD8);
        let q8 = _mm_packs_epi16(
            _mm256_castsi256_si128(q16),
            _mm256_extracti128_si256(q16, 1),
        );
        _mm_storeu_si128(q_ptr.add(i) as *mut __m128i, q8);
        i += 16;
    }
    scale
}

/// Register-resident `head_dim = 32` dot product between two 32-element slices.
#[inline(always)]
pub fn dot32_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert!(a.len() >= 32 && b.len() >= 32);
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let mut s0 = vmulq_f32(vld1q_f32(pa), vld1q_f32(pb));
        let mut s1 = vmulq_f32(vld1q_f32(pa.add(4)), vld1q_f32(pb.add(4)));
        let mut s2 = vmulq_f32(vld1q_f32(pa.add(8)), vld1q_f32(pb.add(8)));
        let mut s3 = vmulq_f32(vld1q_f32(pa.add(12)), vld1q_f32(pb.add(12)));
        s0 = vfmaq_f32(s0, vld1q_f32(pa.add(16)), vld1q_f32(pb.add(16)));
        s1 = vfmaq_f32(s1, vld1q_f32(pa.add(20)), vld1q_f32(pb.add(20)));
        s2 = vfmaq_f32(s2, vld1q_f32(pa.add(24)), vld1q_f32(pb.add(24)));
        s3 = vfmaq_f32(s3, vld1q_f32(pa.add(28)), vld1q_f32(pb.add(28)));
        return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    }
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            return unsafe { dot32_f32_avx2(a.as_ptr(), b.as_ptr()) };
        }
    }
    #[allow(unreachable_code)]
    {
        let mut sum = 0.0f32;
        for i in 0..32 {
            sum += a[i] * b[i];
        }
        sum
    }
}

/// Computes `scores[tk] = dot(q_slice[0..32], k_head[tk*32..(tk+1)*32]) * scale` for `tk in 0..seq_len`
/// keeping `q_slice[0..32]` in 8 SIMD registers and unrolling 4 keys at a time.
#[inline]
pub fn compute_scores_head32(q_slice: &[f32], k_head: &[f32], scale: f32, scores: &mut [f32]) {
    let seq_len = scores.len();
    debug_assert!(q_slice.len() >= 32);
    debug_assert!(k_head.len() >= seq_len * 32);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let qp = q_slice.as_ptr();
        let vscale = vdupq_n_f32(scale);
        let q0 = vmulq_f32(vld1q_f32(qp), vscale);
        let q1 = vmulq_f32(vld1q_f32(qp.add(4)), vscale);
        let q2 = vmulq_f32(vld1q_f32(qp.add(8)), vscale);
        let q3 = vmulq_f32(vld1q_f32(qp.add(12)), vscale);
        let q4 = vmulq_f32(vld1q_f32(qp.add(16)), vscale);
        let q5 = vmulq_f32(vld1q_f32(qp.add(20)), vscale);
        let q6 = vmulq_f32(vld1q_f32(qp.add(24)), vscale);
        let q7 = vmulq_f32(vld1q_f32(qp.add(28)), vscale);

        let kp = k_head.as_ptr();
        let sp = scores.as_mut_ptr();
        let mut tk = 0usize;
        while tk + 4 <= seq_len {
            let k0 = kp.add(tk * 32);
            let k1 = k0.add(32);
            let k2 = k0.add(64);
            let k3 = k0.add(96);

            let mut s0 = vaddq_f32(
                vmulq_f32(q0, vld1q_f32(k0)),
                vmulq_f32(q1, vld1q_f32(k0.add(4))),
            );
            let mut s1 = vaddq_f32(
                vmulq_f32(q0, vld1q_f32(k1)),
                vmulq_f32(q1, vld1q_f32(k1.add(4))),
            );
            let mut s2 = vaddq_f32(
                vmulq_f32(q0, vld1q_f32(k2)),
                vmulq_f32(q1, vld1q_f32(k2.add(4))),
            );
            let mut s3 = vaddq_f32(
                vmulq_f32(q0, vld1q_f32(k3)),
                vmulq_f32(q1, vld1q_f32(k3.add(4))),
            );

            s0 = vfmaq_f32(s0, q2, vld1q_f32(k0.add(8)));
            s1 = vfmaq_f32(s1, q2, vld1q_f32(k1.add(8)));
            s2 = vfmaq_f32(s2, q2, vld1q_f32(k2.add(8)));
            s3 = vfmaq_f32(s3, q2, vld1q_f32(k3.add(8)));

            s0 = vfmaq_f32(s0, q3, vld1q_f32(k0.add(12)));
            s1 = vfmaq_f32(s1, q3, vld1q_f32(k1.add(12)));
            s2 = vfmaq_f32(s2, q3, vld1q_f32(k2.add(12)));
            s3 = vfmaq_f32(s3, q3, vld1q_f32(k3.add(12)));

            s0 = vfmaq_f32(s0, q4, vld1q_f32(k0.add(16)));
            s1 = vfmaq_f32(s1, q4, vld1q_f32(k1.add(16)));
            s2 = vfmaq_f32(s2, q4, vld1q_f32(k2.add(16)));
            s3 = vfmaq_f32(s3, q4, vld1q_f32(k3.add(16)));

            s0 = vfmaq_f32(s0, q5, vld1q_f32(k0.add(20)));
            s1 = vfmaq_f32(s1, q5, vld1q_f32(k1.add(20)));
            s2 = vfmaq_f32(s2, q5, vld1q_f32(k2.add(20)));
            s3 = vfmaq_f32(s3, q5, vld1q_f32(k3.add(20)));

            s0 = vfmaq_f32(s0, q6, vld1q_f32(k0.add(24)));
            s1 = vfmaq_f32(s1, q6, vld1q_f32(k1.add(24)));
            s2 = vfmaq_f32(s2, q6, vld1q_f32(k2.add(24)));
            s3 = vfmaq_f32(s3, q6, vld1q_f32(k3.add(24)));

            s0 = vfmaq_f32(s0, q7, vld1q_f32(k0.add(28)));
            s1 = vfmaq_f32(s1, q7, vld1q_f32(k1.add(28)));
            s2 = vfmaq_f32(s2, q7, vld1q_f32(k2.add(28)));
            s3 = vfmaq_f32(s3, q7, vld1q_f32(k3.add(28)));

            // Pairwise horizontal reduction of 4 vectors into 1 float32x4_t `[dot0, dot1, dot2, dot3]`
            let p01 = vpaddq_f32(s0, s1);
            let p23 = vpaddq_f32(s2, s3);
            let sum4 = vpaddq_f32(p01, p23);
            vst1q_f32(sp.add(tk), sum4);
            tk += 4;
        }
        while tk < seq_len {
            let k0 = kp.add(tk * 32);
            let mut s0 = vaddq_f32(
                vmulq_f32(q0, vld1q_f32(k0)),
                vmulq_f32(q1, vld1q_f32(k0.add(4))),
            );
            s0 = vfmaq_f32(s0, q2, vld1q_f32(k0.add(8)));
            s0 = vfmaq_f32(s0, q3, vld1q_f32(k0.add(12)));
            s0 = vfmaq_f32(s0, q4, vld1q_f32(k0.add(16)));
            s0 = vfmaq_f32(s0, q5, vld1q_f32(k0.add(20)));
            s0 = vfmaq_f32(s0, q6, vld1q_f32(k0.add(24)));
            s0 = vfmaq_f32(s0, q7, vld1q_f32(k0.add(28)));
            *sp.add(tk) = vaddvq_f32(s0);
            tk += 1;
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe {
            compute_scores_head32_avx2(
                q_slice.as_ptr(),
                k_head.as_ptr(),
                scale,
                scores.as_mut_ptr(),
                seq_len,
            )
        };
        return;
    }

    #[allow(unreachable_code)]
    for tk in 0..seq_len {
        scores[tk] = dot32_f32(q_slice, &k_head[tk * 32..(tk + 1) * 32]) * scale;
    }
}

/// Fast in-place softmax over a slice `scores[0..seq_len]` with 4x NEON unrolling.
#[inline]
pub fn softmax_inplace(scores: &mut [f32]) {
    let len = scores.len();
    if len == 0 {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let ptr = scores.as_mut_ptr();
        let mut vmax0 = vdupq_n_f32(f32::NEG_INFINITY);
        let mut vmax1 = vdupq_n_f32(f32::NEG_INFINITY);
        let mut i = 0;
        while i + 16 <= len {
            let m0 = vmaxq_f32(vld1q_f32(ptr.add(i)), vld1q_f32(ptr.add(i + 4)));
            let m1 = vmaxq_f32(vld1q_f32(ptr.add(i + 8)), vld1q_f32(ptr.add(i + 12)));
            vmax0 = vmaxq_f32(vmax0, m0);
            vmax1 = vmaxq_f32(vmax1, m1);
            i += 16;
        }
        let mut vmax_v = vmaxq_f32(vmax0, vmax1);
        while i + 4 <= len {
            vmax_v = vmaxq_f32(vmax_v, vld1q_f32(ptr.add(i)));
            i += 4;
        }
        let mut max_val = vmaxvq_f32(vmax_v);
        while i < len {
            if *ptr.add(i) > max_val {
                max_val = *ptr.add(i);
            }
            i += 1;
        }

        let vmax = vdupq_n_f32(max_val);
        let mut vsum0 = vdupq_n_f32(0.0);
        let mut vsum1 = vdupq_n_f32(0.0);
        i = 0;
        while i + 16 <= len {
            let e0 = neon_exp_f32(vsubq_f32(vld1q_f32(ptr.add(i)), vmax));
            let e1 = neon_exp_f32(vsubq_f32(vld1q_f32(ptr.add(i + 4)), vmax));
            let e2 = neon_exp_f32(vsubq_f32(vld1q_f32(ptr.add(i + 8)), vmax));
            let e3 = neon_exp_f32(vsubq_f32(vld1q_f32(ptr.add(i + 12)), vmax));
            vst1q_f32(ptr.add(i), e0);
            vst1q_f32(ptr.add(i + 4), e1);
            vst1q_f32(ptr.add(i + 8), e2);
            vst1q_f32(ptr.add(i + 12), e3);
            vsum0 = vaddq_f32(vsum0, vaddq_f32(e0, e1));
            vsum1 = vaddq_f32(vsum1, vaddq_f32(e2, e3));
            i += 16;
        }
        let mut vsum = vaddq_f32(vsum0, vsum1);
        while i + 4 <= len {
            let e = neon_exp_f32(vsubq_f32(vld1q_f32(ptr.add(i)), vmax));
            vst1q_f32(ptr.add(i), e);
            vsum = vaddq_f32(vsum, e);
            i += 4;
        }
        let mut sum = vaddvq_f32(vsum);
        while i < len {
            let e = (*ptr.add(i) - max_val).exp();
            *ptr.add(i) = e;
            sum += e;
            i += 1;
        }

        let inv_sum = 1.0 / sum;
        let vinv = vdupq_n_f32(inv_sum);
        i = 0;
        while i + 16 <= len {
            vst1q_f32(ptr.add(i), vmulq_f32(vld1q_f32(ptr.add(i)), vinv));
            vst1q_f32(ptr.add(i + 4), vmulq_f32(vld1q_f32(ptr.add(i + 4)), vinv));
            vst1q_f32(ptr.add(i + 8), vmulq_f32(vld1q_f32(ptr.add(i + 8)), vinv));
            vst1q_f32(ptr.add(i + 12), vmulq_f32(vld1q_f32(ptr.add(i + 12)), vinv));
            i += 16;
        }
        while i + 4 <= len {
            vst1q_f32(ptr.add(i), vmulq_f32(vld1q_f32(ptr.add(i)), vinv));
            i += 4;
        }
        while i < len {
            *ptr.add(i) *= inv_sum;
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        unsafe { softmax_inplace_avx2(scores) };
        return;
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut max_val = f32::NEG_INFINITY;
        for &s in scores.iter() {
            if s > max_val {
                max_val = s;
            }
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            let e = (*s - max_val).exp();
            *s = e;
            sum += e;
        }
        let inv_sum = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }
    }
}

/// Accumulates `out[0..32] = sum_{t=0..seq_len} probs[t] * v_head[t * 32 .. (t + 1) * 32]`
/// keeping all 32 floats in 8 NEON registers (or 4 AVX2 registers) across the entire `t` loop.
#[inline]
pub fn weighted_sum_head32(probs: &[f32], v_head: &[f32], out_32: &mut [f32]) {
    let seq_len = probs.len();
    debug_assert_eq!(v_head.len(), seq_len * 32);
    debug_assert!(out_32.len() >= 32);

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);
        let mut acc4 = vdupq_n_f32(0.0);
        let mut acc5 = vdupq_n_f32(0.0);
        let mut acc6 = vdupq_n_f32(0.0);
        let mut acc7 = vdupq_n_f32(0.0);

        let v_ptr = v_head.as_ptr();
        let p_ptr = probs.as_ptr();
        let mut t = 0usize;
        while t + 4 <= seq_len {
            let vp = vld1q_f32(p_ptr.add(t));
            let r0 = v_ptr.add(t * 32);
            let r1 = r0.add(32);
            let r2 = r0.add(64);
            let r3 = r0.add(96);

            acc0 = vfmaq_laneq_f32::<0>(acc0, vld1q_f32(r0), vp);
            acc1 = vfmaq_laneq_f32::<0>(acc1, vld1q_f32(r0.add(4)), vp);
            acc2 = vfmaq_laneq_f32::<0>(acc2, vld1q_f32(r0.add(8)), vp);
            acc3 = vfmaq_laneq_f32::<0>(acc3, vld1q_f32(r0.add(12)), vp);
            acc4 = vfmaq_laneq_f32::<0>(acc4, vld1q_f32(r0.add(16)), vp);
            acc5 = vfmaq_laneq_f32::<0>(acc5, vld1q_f32(r0.add(20)), vp);
            acc6 = vfmaq_laneq_f32::<0>(acc6, vld1q_f32(r0.add(24)), vp);
            acc7 = vfmaq_laneq_f32::<0>(acc7, vld1q_f32(r0.add(28)), vp);

            acc0 = vfmaq_laneq_f32::<1>(acc0, vld1q_f32(r1), vp);
            acc1 = vfmaq_laneq_f32::<1>(acc1, vld1q_f32(r1.add(4)), vp);
            acc2 = vfmaq_laneq_f32::<1>(acc2, vld1q_f32(r1.add(8)), vp);
            acc3 = vfmaq_laneq_f32::<1>(acc3, vld1q_f32(r1.add(12)), vp);
            acc4 = vfmaq_laneq_f32::<1>(acc4, vld1q_f32(r1.add(16)), vp);
            acc5 = vfmaq_laneq_f32::<1>(acc5, vld1q_f32(r1.add(20)), vp);
            acc6 = vfmaq_laneq_f32::<1>(acc6, vld1q_f32(r1.add(24)), vp);
            acc7 = vfmaq_laneq_f32::<1>(acc7, vld1q_f32(r1.add(28)), vp);

            acc0 = vfmaq_laneq_f32::<2>(acc0, vld1q_f32(r2), vp);
            acc1 = vfmaq_laneq_f32::<2>(acc1, vld1q_f32(r2.add(4)), vp);
            acc2 = vfmaq_laneq_f32::<2>(acc2, vld1q_f32(r2.add(8)), vp);
            acc3 = vfmaq_laneq_f32::<2>(acc3, vld1q_f32(r2.add(12)), vp);
            acc4 = vfmaq_laneq_f32::<2>(acc4, vld1q_f32(r2.add(16)), vp);
            acc5 = vfmaq_laneq_f32::<2>(acc5, vld1q_f32(r2.add(20)), vp);
            acc6 = vfmaq_laneq_f32::<2>(acc6, vld1q_f32(r2.add(24)), vp);
            acc7 = vfmaq_laneq_f32::<2>(acc7, vld1q_f32(r2.add(28)), vp);

            acc0 = vfmaq_laneq_f32::<3>(acc0, vld1q_f32(r3), vp);
            acc1 = vfmaq_laneq_f32::<3>(acc1, vld1q_f32(r3.add(4)), vp);
            acc2 = vfmaq_laneq_f32::<3>(acc2, vld1q_f32(r3.add(8)), vp);
            acc3 = vfmaq_laneq_f32::<3>(acc3, vld1q_f32(r3.add(12)), vp);
            acc4 = vfmaq_laneq_f32::<3>(acc4, vld1q_f32(r3.add(16)), vp);
            acc5 = vfmaq_laneq_f32::<3>(acc5, vld1q_f32(r3.add(20)), vp);
            acc6 = vfmaq_laneq_f32::<3>(acc6, vld1q_f32(r3.add(24)), vp);
            acc7 = vfmaq_laneq_f32::<3>(acc7, vld1q_f32(r3.add(28)), vp);

            t += 4;
        }
        while t < seq_len {
            let vp = vdupq_n_f32(*p_ptr.add(t));
            let row = v_ptr.add(t * 32);
            acc0 = vfmaq_f32(acc0, vld1q_f32(row), vp);
            acc1 = vfmaq_f32(acc1, vld1q_f32(row.add(4)), vp);
            acc2 = vfmaq_f32(acc2, vld1q_f32(row.add(8)), vp);
            acc3 = vfmaq_f32(acc3, vld1q_f32(row.add(12)), vp);
            acc4 = vfmaq_f32(acc4, vld1q_f32(row.add(16)), vp);
            acc5 = vfmaq_f32(acc5, vld1q_f32(row.add(20)), vp);
            acc6 = vfmaq_f32(acc6, vld1q_f32(row.add(24)), vp);
            acc7 = vfmaq_f32(acc7, vld1q_f32(row.add(28)), vp);
            t += 1;
        }
        let dst = out_32.as_mut_ptr();
        vst1q_f32(dst, acc0);
        vst1q_f32(dst.add(4), acc1);
        vst1q_f32(dst.add(8), acc2);
        vst1q_f32(dst.add(12), acc3);
        vst1q_f32(dst.add(16), acc4);
        vst1q_f32(dst.add(20), acc5);
        vst1q_f32(dst.add(24), acc6);
        vst1q_f32(dst.add(28), acc7);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            unsafe {
                weighted_sum_head32_avx2(
                    probs.as_ptr(),
                    v_head.as_ptr(),
                    seq_len,
                    out_32.as_mut_ptr(),
                )
            };
            return;
        }
    }

    #[allow(unreachable_code)]
    {
        out_32[..32].fill(0.0);
        for (t, &p) in probs.iter().enumerate() {
            let row = &v_head[t * 32..(t + 1) * 32];
            for d in 0..32 {
                out_32[d] += p * row[d];
            }
        }
    }
}

/// Vectorized dequantization and addition of Int8 word + position + token_type embeddings:
/// `h_row[j] = w_row[j] * w_s + p_row[j] * p_s + t_row[j]` for `j in 0..384`.
#[inline]
pub fn dequant_add_embeddings_row(
    w_row: &[i8],
    w_s: f32,
    p_row: &[i8],
    p_s: f32,
    t_row: &[f32],
    h_row: &mut [f32],
) {
    let n = h_row.len();
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let wp = w_row.as_ptr();
        let pp = p_row.as_ptr();
        let tp = t_row.as_ptr();
        let hp = h_row.as_mut_ptr();
        let vws = vdupq_n_f32(w_s);
        let vps = vdupq_n_f32(p_s);
        let mut i = 0;
        while i + 16 <= n {
            let vw8 = vld1q_s8(wp.add(i));
            let vp8 = vld1q_s8(pp.add(i));
            let vw_lo16 = vmovl_s8(vget_low_s8(vw8));
            let vw_hi16 = vmovl_s8(vget_high_s8(vw8));
            let vp_lo16 = vmovl_s8(vget_low_s8(vp8));
            let vp_hi16 = vmovl_s8(vget_high_s8(vp8));

            let w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vw_lo16)));
            let w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vw_lo16)));
            let w2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vw_hi16)));
            let w3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vw_hi16)));

            let p0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vp_lo16)));
            let p1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vp_lo16)));
            let p2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vp_hi16)));
            let p3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vp_hi16)));

            let r0 = vfmaq_f32(vfmaq_f32(vld1q_f32(tp.add(i)), w0, vws), p0, vps);
            let r1 = vfmaq_f32(vfmaq_f32(vld1q_f32(tp.add(i + 4)), w1, vws), p1, vps);
            let r2 = vfmaq_f32(vfmaq_f32(vld1q_f32(tp.add(i + 8)), w2, vws), p2, vps);
            let r3 = vfmaq_f32(vfmaq_f32(vld1q_f32(tp.add(i + 12)), w3, vws), p3, vps);

            vst1q_f32(hp.add(i), r0);
            vst1q_f32(hp.add(i + 4), r1);
            vst1q_f32(hp.add(i + 8), r2);
            vst1q_f32(hp.add(i + 12), r3);
            i += 16;
        }
        return;
    }

    #[allow(unreachable_code)]
    for j in 0..n {
        h_row[j] = (w_row[j] as f32) * w_s + (p_row[j] as f32) * p_s + t_row[j];
    }
}

/// L1-Tiled Full-Int8 Multi-Head Self-Attention using ARMv9 `FEAT_SME` (`smopa`):
/// - `Q @ K^T` uses **Int8 `smopa`** (`16x64` outer product in just 8 K-steps for `head_dim = 32`,
///   4x fewer matrix instructions and 4x less L1D bandwidth than FP32 `fmopa`) with per-token dynamic scaling.
/// - **2-Pass Exact-Range Int8 Softmax**: exploits the identity $\max_k \exp(s_k - m_r) \equiv 1.0$ to quantize
///   unnormalized exponentials directly to `i8` in `[0, 127]` during Pass 2 while accumulating the exact integer
///   sum $S_r = \sum_k q_{r,k}$ for row normalization `p_scale[r] = 1.0 / S_r`, eliminating Softmax Pass 3.
/// - `P @ V` uses **2-tile Int8 `smopa`** (`16x32` outer product in `seq_len.div_ceil(4)` K-steps, 4x fewer steps
///   and 4x less bandwidth than FP32 `fmopa`) with per-column `v_scale[0..32]` dequantization.
///
/// # Safety
/// The CPU must support `FEAT_SME` (callers dispatch on [`crate::gemm::SimdBackend::ArmSme`]); all
/// slices must hold at least the sizes implied by `m`/`seq_len` and the model dimensions.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub unsafe fn attention_multihead_sme(
    seq_len: usize,
    qkv_buf: &[f32],
    q_t_f32: &mut [f32],
    k_t_f32: &mut [f32],
    v_contig_f32: &mut [f32],
    scores_blk: &mut [f32],
    p_t_f32: &mut [f32],
    attn_ctx: &mut [f32],
    seg_ranges: Option<&[(usize, usize)]>,
) {
    debug_assert!(attn_ctx.len() >= seq_len * 384);
    attention_heads_sme(
        seq_len,
        qkv_buf,
        q_t_f32,
        k_t_f32,
        v_contig_f32,
        scores_blk,
        p_t_f32,
        attn_ctx.as_mut_ptr(),
        seg_ranges,
        0..12,
    );
}

/// Head-range variant of [`attention_multihead_sme`] used for intra-op parallelism: computes heads
/// `heads` only, writing their 32-column slices of the row-major `[seq_len, 384]` context through a raw
/// pointer so threads can fill disjoint head columns of one shared buffer.
///
/// # Safety
/// `attn_ctx` must be valid for `seq_len * 384` f32s and no other thread may access the written head
/// columns concurrently; the scratch slices are per-thread.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_heads_sme(
    seq_len: usize,
    qkv_buf: &[f32],
    q_t_f32: &mut [f32],
    k_t_f32: &mut [f32],
    v_contig_f32: &mut [f32],
    scores_blk: &mut [f32],
    p_t_f32: &mut [f32],
    attn_ctx: *mut f32,
    seg_ranges: Option<&[(usize, usize)]>,
    heads: std::ops::Range<usize>,
) {
    const HIDDEN: usize = 384;
    const HEADS: usize = 12;
    const HDIM: usize = 32;
    let inv_sqrt_d = 1.0 / (HDIM as f32).sqrt();
    let m_blocks = seq_len.div_ceil(16);
    let k_super = m_blocks.div_ceil(4);
    let seq_stride = k_super * 64;
    // P @ V runs as 2-way int16 smopa: each K-step consumes a pair of keys.
    let tk_steps = seq_len.div_ceil(2);
    let tk_steps_aligned = tk_steps.div_ceil(4) * 4;
    let tk_aligned = tk_steps_aligned * 2;

    // Partition workspace buffers into Int8 tiles and scale vectors (zero heap allocations):
    let q_t_i8 = q_t_f32.as_mut_ptr() as *mut i8;
    let k_t_i8 = k_t_f32.as_mut_ptr() as *mut i8;
    let q_scale = q_t_f32.as_mut_ptr().add(8192);
    let k_scale = k_t_f32.as_mut_ptr().add(8192);

    // [tk_steps, 32 cols, 2 keys] i16 = up to 256 * 128 = 32 KiB, then 32 per-column scales.
    let v_packed_i16 = v_contig_f32.as_mut_ptr() as *mut i16;
    let v_scale = v_contig_f32.as_mut_ptr().add(8192); // 32 floats

    // [16, seq_stride] i16 = up to 16 KiB, its pair-transpose [tk_steps_aligned, 16 rows, 2 keys] i16 = up to
    // 16 KiB, then 16 row scales (seq_stride <= 512, tk_steps_aligned <= 256).
    let probs_blk_i16 = p_t_f32.as_mut_ptr() as *mut i16;
    let p_t_i16 = (p_t_f32.as_mut_ptr() as *mut i8).add(16384) as *mut i16;
    let p_scale = p_t_f32.as_mut_ptr().add(8192); // 16 floats
    debug_assert!(seq_stride <= 512 && tk_steps_aligned <= 256);

    debug_assert!(heads.end <= HEADS);
    for h in heads {
        let h_off = h * HDIM;
        pack_qkv_head_int8_neon(
            qkv_buf,
            seq_len,
            seq_stride,
            tk_steps,
            h_off,
            // Fold log2(e) into the K scale so the softmax evaluates exp2 (scores in log2 units).
            inv_sqrt_d * std::f32::consts::LOG2_E,
            q_t_i8,
            k_t_i8,
            q_scale,
            k_scale,
            v_packed_i16,
            v_scale,
        );

        for qb in 0..m_blocks {
            let q_base = qb * 16;
            let valid_q = (seq_len - q_base).min(16);
            let pq_blk = q_t_i8.add(qb * 512);

            // 1. Compute scores_blk[0..valid_q, 0..seq_stride] = Q_blk @ K_T using 4-tile Int8 smopa (8 K-steps!)
            let pk0_init = k_t_i8 as *const i8;
            let pk1_init = pk0_init.add(512);
            let pk2_init = pk0_init.add(1024);
            let pk3_init = pk0_init.add(1536);
            let pout_init = scores_blk.as_mut_ptr();

            core::arch::asm!(
                ".arch_extension sme",
                "smstart",
                "ptrue p0.b",
                "ptrue p1.s",
                "5:",
                "zero {{za}}",
                "mov {d_cnt}, #8",
                "mov {pq}, {pq_blk}",
                "2:",
                "ld1b {{z0.b}}, p0/z, [{pq}]",
                "ld1b {{z1.b}}, p0/z, [{k0}]",
                "ld1b {{z2.b}}, p0/z, [{k1}]",
                "ld1b {{z3.b}}, p0/z, [{k2}]",
                "ld1b {{z4.b}}, p0/z, [{k3}]",
                "add {pq}, {pq}, #64",
                "add {k0}, {k0}, #64",
                "add {k1}, {k1}, #64",
                "add {k2}, {k2}, #64",
                "add {k3}, {k3}, #64",
                "smopa za0.s, p0/m, p0/m, z0.b, z1.b",
                "smopa za1.s, p0/m, p0/m, z0.b, z2.b",
                "smopa za2.s, p0/m, p0/m, z0.b, z3.b",
                "smopa za3.s, p0/m, p0/m, z0.b, z4.b",
                "subs {d_cnt}, {d_cnt}, #1",
                "b.ne 2b",

                // Raw i32 score rows straight from ZA (streaming-SVE vector ops cost ~3 cycles each on M4);
                // the K/Q scale dequantization is fused into the NEON softmax instead.
                "mov {po}, {pout_col}",
                "mov w12, #0",
                "3:",
                "st1w {{za0h.s[w12, 0]}}, p1, [{po}]",
                "add {pqs_tmp}, {po}, #64",
                "st1w {{za1h.s[w12, 0]}}, p1, [{pqs_tmp}]",
                "add {pqs_tmp}, {po}, #128",
                "st1w {{za2h.s[w12, 0]}}, p1, [{pqs_tmp}]",
                "add {pqs_tmp}, {po}, #192",
                "st1w {{za3h.s[w12, 0]}}, p1, [{pqs_tmp}]",
                "add {po}, {po}, {s_bytes}",
                "add w12, w12, #1",
                "cmp w12, {valid_q:w}",
                "b.lo 3b",

                "mov {k0}, {k3}",
                "add {k1}, {k0}, #512",
                "add {k2}, {k1}, #512",
                "add {k3}, {k2}, #512",
                "add {pout_col}, {pout_col}, #256",
                "subs {ks_cnt}, {ks_cnt}, #1",
                "b.ne 5b",
                "smstop",
                pq_blk = inout(reg) pq_blk => _,
                k0 = inout(reg) pk0_init => _,
                k1 = inout(reg) pk1_init => _,
                k2 = inout(reg) pk2_init => _,
                k3 = inout(reg) pk3_init => _,
                pout_col = inout(reg) pout_init => _,
                s_bytes = inout(reg) seq_stride * 4 => _,
                ks_cnt = inout(reg) k_super => _,
                valid_q = inout(reg) valid_q => _,
                d_cnt = lateout(reg) _,
                pq = lateout(reg) _,
                pqs_tmp = lateout(reg) _,
                po = lateout(reg) _,
                out("w12") _,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                options(nostack)
            );

            // 2. Two-Pass Exact-Range Int16 Softmax (raw i32 `scores_blk` -> `probs_blk_i16` + `p_scale`)
            if let Some(ranges) = seg_ranges {
                for r in 0..valid_q {
                    let (s_start, s_end) = ranges[q_base + r];
                    let s_row = &mut scores_blk[r * seq_stride..r * seq_stride + seq_len];
                    let p_row = std::slice::from_raw_parts_mut(
                        probs_blk_i16.add(r * seq_stride),
                        tk_aligned,
                    );
                    p_row[..s_start].fill(0);
                    let ks = std::slice::from_raw_parts(k_scale.add(s_start), s_end - s_start);
                    *p_scale.add(r) = softmax_quantize_i16_from_acc_neon(
                        &mut s_row[s_start..s_end],
                        ks,
                        *q_scale.add(q_base + r),
                        &mut p_row[s_start..s_end],
                    );
                    p_row[s_end..tk_aligned].fill(0);
                }
            } else {
                for r in 0..valid_q {
                    let s_row = &mut scores_blk[r * seq_stride..r * seq_stride + seq_len];
                    let p_row = std::slice::from_raw_parts_mut(
                        probs_blk_i16.add(r * seq_stride),
                        tk_aligned,
                    );
                    let ks = std::slice::from_raw_parts(k_scale as *const f32, seq_len);
                    *p_scale.add(r) = softmax_quantize_i16_from_acc_neon(
                        s_row,
                        ks,
                        *q_scale.add(q_base + r),
                        &mut p_row[..seq_len],
                    );
                    for v in &mut p_row[seq_len..tk_aligned] {
                        *v = 0;
                    }
                }
            }

            // 3. Transpose 4-byte groups (2 keys of i16) of `probs_blk_i16` into `p_t_i16`
            transpose_probs_16_i8_neon(
                probs_blk_i16 as *const i8,
                seq_stride * 2,
                valid_q,
                tk_steps_aligned,
                p_t_i16 as *mut i8,
            );

            // 4. ctx_blk[0..valid_q, 0..32] = P_T_i16 @ V_packed_i16 with 2-way int16 smopa (za0, za1), `tk_steps` key pairs
            let pctx_out = attn_ctx.add(q_base * HIDDEN + h_off);
            core::arch::asm!(
                ".arch_extension sme2",
                "smstart",
                "ptrue p0.b",
                "ptrue p1.s",
                "zero {{za}}",
                "mov {tk_cnt}, {tk_steps}",
                "2:",
                "ld1b {{z0.b}}, p0/z, [{pp}]",
                "ld1b {{z1.b}}, p0/z, [{pv}]",
                "ld1b {{z2.b}}, p0/z, [{pv}, #1, mul vl]",
                "add {pp}, {pp}, #64",
                "add {pv}, {pv}, #128",
                "smopa za0.s, p0/m, p0/m, z0.h, z1.h",
                "smopa za1.s, p0/m, p0/m, z0.h, z2.h",
                "subs {tk_cnt}, {tk_cnt}, #1",
                "b.ne 2b",

                "ld1w {{z16.s}}, p1/z, [{pvs}]",
                "ld1w {{z17.s}}, p1/z, [{pvs}, #1, mul vl]",
                "mov w12, #0",
                "3:",
                "mova z0.s, p1/m, za0h.s[w12, 0]",
                "mova z1.s, p1/m, za1h.s[w12, 0]",
                "scvtf z0.s, p1/m, z0.s",
                "scvtf z1.s, p1/m, z1.s",
                "ld1rw {{z4.s}}, p1/z, [{pps}]",
                "add {pps}, {pps}, #4",
                "fmul z5.s, z16.s, z4.s",
                "fmul z6.s, z17.s, z4.s",
                "fmul z0.s, z0.s, z5.s",
                "fmul z1.s, z1.s, z6.s",
                "st1w {{z0.s}}, p1, [{po}]",
                "st1w {{z1.s}}, p1, [{po}, #1, mul vl]",
                "add {po}, {po}, #1536",
                "add w12, w12, #1",
                "cmp w12, {valid_q:w}",
                "b.lo 3b",
                "smstop",
                pp = inout(reg) p_t_i16 => _,
                pv = inout(reg) v_packed_i16 => _,
                pvs = inout(reg) v_scale => _,
                pps = inout(reg) p_scale => _,
                tk_steps = inout(reg) tk_steps => _,
                po = inout(reg) pctx_out => _,
                valid_q = inout(reg) valid_q => _,
                tk_cnt = lateout(reg) _,
                out("w12") _,
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
                out("v8") _, out("v9") _, out("v10") _, out("v11") _,
                out("v12") _, out("v13") _, out("v14") _, out("v15") _,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _,
                out("v24") _, out("v25") _, out("v26") _, out("v27") _,
                out("v28") _, out("v29") _, out("v30") _, out("v31") _,
                options(nostack)
            );
        }
    }
}

/// Two-Pass Exact-Range Int8 Softmax:
/// Uses $\max_k \exp(s_k - m_r) \equiv 1.0$ to quantize unnormalized exponentials directly to `i8` in `[0, 127]`
/// during Pass 2 while accumulating the exact integer sum $S_r = \sum_k q_k$, returning `p_scale = 1.0 / S_r`.
///
/// # Safety
/// Requires NEON (always present on AArch64); `out_*` must be at least as long as the input row.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn softmax_quantize_i8_neon(scores: &[f32], out_i8: &mut [i8]) -> f32 {
    let len = scores.len();
    if len == 0 {
        return 0.0;
    }
    let ptr = scores.as_ptr();

    let mut vmax0 = vdupq_n_f32(f32::NEG_INFINITY);
    let mut vmax1 = vdupq_n_f32(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 16 <= len {
        let m0 = vmaxq_f32(vld1q_f32(ptr.add(i)), vld1q_f32(ptr.add(i + 4)));
        let m1 = vmaxq_f32(vld1q_f32(ptr.add(i + 8)), vld1q_f32(ptr.add(i + 12)));
        vmax0 = vmaxq_f32(vmax0, m0);
        vmax1 = vmaxq_f32(vmax1, m1);
        i += 16;
    }
    let mut vmax_v = vmaxq_f32(vmax0, vmax1);
    while i + 4 <= len {
        vmax_v = vmaxq_f32(vmax_v, vld1q_f32(ptr.add(i)));
        i += 4;
    }
    let mut max_val = vmaxvq_f32(vmax_v);
    while i < len {
        if *ptr.add(i) > max_val {
            max_val = *ptr.add(i);
        }
        i += 1;
    }

    softmax_exp2_quantize_pass_neon(scores, max_val, out_i8)
}

/// Pass 2 of the exact-range Int8 softmax: `q = round(127 * 2^(s - max))` (scores in log2 units),
/// returning `1 / sum(q)`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn softmax_exp2_quantize_pass_neon(scores: &[f32], max_val: f32, out_i8: &mut [i8]) -> f32 {
    let len = scores.len();
    let ptr = scores.as_ptr();
    let q_ptr = out_i8.as_mut_ptr();
    let vmax = vdupq_n_f32(max_val);
    let mut vsum0 = vdupq_n_s32(0);
    let mut vsum1 = vdupq_n_s32(0);
    let mut i = 0;
    while i + 16 <= len {
        let q0 = exp2_q127_neon(vsubq_f32(vld1q_f32(ptr.add(i)), vmax));
        let q1 = exp2_q127_neon(vsubq_f32(vld1q_f32(ptr.add(i + 4)), vmax));
        let q2 = exp2_q127_neon(vsubq_f32(vld1q_f32(ptr.add(i + 8)), vmax));
        let q3 = exp2_q127_neon(vsubq_f32(vld1q_f32(ptr.add(i + 12)), vmax));

        vsum0 = vaddq_s32(vsum0, vaddq_s32(q0, q1));
        vsum1 = vaddq_s32(vsum1, vaddq_s32(q2, q3));

        let q01 = vqmovn_high_s32(vqmovn_s32(q0), q1);
        let q23 = vqmovn_high_s32(vqmovn_s32(q2), q3);
        vst1q_s8(q_ptr.add(i), vqmovn_high_s16(vqmovn_s16(q01), q23));
        i += 16;
    }
    while i + 4 <= len {
        let q0 = exp2_q127_neon(vsubq_f32(vld1q_f32(ptr.add(i)), vmax));
        vsum0 = vaddq_s32(vsum0, q0);
        let q16 = vqmovn_s32(q0);
        let q8 = vqmovn_s16(vcombine_s16(q16, q16));
        let w = vget_lane_u32(vreinterpret_u32_s8(q8), 0);
        std::ptr::write_unaligned(q_ptr.add(i) as *mut u32, w);
        i += 4;
    }
    let mut isum = vaddvq_s32(vaddq_s32(vsum0, vsum1));
    while i < len {
        // Scores are in log2 units (log2(e) is folded into the K scale).
        let e = ((*ptr.add(i) - max_val).exp2() * 127.0).round() as i32;
        let q = e.clamp(0, 127);
        *q_ptr.add(i) = q as i8;
        isum += q;
        i += 1;
    }
    1.0 / (isum.max(1) as f32)
}

/// Exact-range Int16 softmax over a row of **raw `i32` Q·K accumulators** (stored as `i32` bits in `row`),
/// as written directly from the ZA tiles by the SME kernel. The max pass dequantizes in place,
/// `s[c] = f32(acc[c]) * k_scale[c] * q_scale` (log2 units), on the core-local NEON pipes, then the
/// standard exp2 quantize pass runs on the dequantized row.
///
/// # Safety
/// Requires NEON (always present on AArch64); `out_*` must be at least as long as `row`, and `k_scale` must cover `row.len()` keys.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn softmax_quantize_i16_from_acc_neon(
    row: &mut [f32],
    k_scale: &[f32],
    q_scale: f32,
    out_i16: &mut [i16],
) -> f32 {
    let len = row.len();
    if len == 0 {
        return 0.0;
    }
    debug_assert!(k_scale.len() >= len);
    let p = row.as_mut_ptr();
    let acc = p as *const i32;
    let ks = k_scale.as_ptr();
    let mut vmax0 = vdupq_n_f32(f32::NEG_INFINITY);
    let mut vmax1 = vdupq_n_f32(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 8 <= len {
        let s0 = vmulq_n_f32(
            vmulq_f32(vcvtq_f32_s32(vld1q_s32(acc.add(i))), vld1q_f32(ks.add(i))),
            q_scale,
        );
        let s1 = vmulq_n_f32(
            vmulq_f32(
                vcvtq_f32_s32(vld1q_s32(acc.add(i + 4))),
                vld1q_f32(ks.add(i + 4)),
            ),
            q_scale,
        );
        vst1q_f32(p.add(i), s0);
        vst1q_f32(p.add(i + 4), s1);
        vmax0 = vmaxq_f32(vmax0, s0);
        vmax1 = vmaxq_f32(vmax1, s1);
        i += 8;
    }
    let mut max_val = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
    while i < len {
        let v = (*acc.add(i) as f32) * *ks.add(i) * q_scale;
        *p.add(i) = v;
        if v > max_val {
            max_val = v;
        }
        i += 1;
    }
    softmax_exp2_quantize_pass_i16_neon(row, max_val, out_i16)
}

/// Computes `round(127 * 2^x)` for `x <= 0` (scores already in log2 units), returning `i32` lanes in `[0, 127]`.
/// Uses a degree-3 minimax polynomial for `2^f, f in [0, 1)` (max rel. error 1.4e-4 = 0.018 quantization
/// levels at the 127 scale) and inserts the integer exponent directly into the IEEE-754 exponent bits,
/// replacing the full-precision Cody-Waite `exp` (~40% fewer NEON ops per element).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn exp2_q127_neon(x: float32x4_t) -> int32x4_t {
    exp2_q_neon(x, 127.0, -9.0)
}

/// Computes `round(scale * 2^x)` for `x <= 0`; `x_min` is the point below which the result rounds to 0
/// (`scale * 2^x_min < 0.5`), which also keeps the exponent arithmetic below in the normal range.
/// Degree-3 minimax `2^f` (max rel. error 1.4e-4) plus direct IEEE-754 exponent insertion.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn exp2_q_neon(x: float32x4_t, scale: f32, x_min: f32) -> int32x4_t {
    let x = vmaxq_f32(x, vdupq_n_f32(x_min));
    let n = vcvtmq_s32_f32(x); // floor(x)
    let f = vsubq_f32(x, vcvtq_f32_s32(n)); // f in [0, 1)
    let mut p = vfmaq_f32(
        vdupq_n_f32(0.225_119_37 * scale),
        f,
        vdupq_n_f32(0.078_421_69 * scale),
    );
    p = vfmaq_f32(vdupq_n_f32(0.696_385_3 * scale), f, p);
    p = vfmaq_f32(vdupq_n_f32(0.999_857 * scale), f, p);
    // Multiply by 2^n by adding n to the exponent field (p in [scale, 2*scale), n in [x_min, 0]).
    let scaled = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), vshlq_n_s32(n, 23)));
    vcvtnq_s32_f32(scaled)
}

/// Attention probabilities are quantized to `[0, P_Q16_MAX]` (14 bits) for the int16 `P @ V` product.
/// With int8 `V` (|v| <= 127) the i32 accumulator is bounded by 512 * 16383 * 127 = 1.07e9 < 2^31.
#[cfg(target_arch = "aarch64")]
const P_Q16_MAX: f32 = 16383.0;

/// Int16 variant of the exp2 quantize pass: `q = round(16383 * 2^(s - max))`, returning `1 / sum(q)`.
/// The 7-bit (`[0, 127]`) version lost ~0.0009 cosine similarity on 300-512 token inputs because most
/// probabilities of a diffuse attention row collapse onto a handful of integer levels.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn softmax_exp2_quantize_pass_i16_neon(
    scores: &[f32],
    max_val: f32,
    out_i16: &mut [i16],
) -> f32 {
    let len = scores.len();
    debug_assert!(out_i16.len() >= len);
    let ptr = scores.as_ptr();
    let q_ptr = out_i16.as_mut_ptr();
    let vmax = vdupq_n_f32(max_val);
    let mut vsum0 = vdupq_n_s32(0);
    let mut vsum1 = vdupq_n_s32(0);
    let mut i = 0;
    while i + 8 <= len {
        let q0 = exp2_q_neon(vsubq_f32(vld1q_f32(ptr.add(i)), vmax), P_Q16_MAX, -16.0);
        let q1 = exp2_q_neon(vsubq_f32(vld1q_f32(ptr.add(i + 4)), vmax), P_Q16_MAX, -16.0);
        vsum0 = vaddq_s32(vsum0, q0);
        vsum1 = vaddq_s32(vsum1, q1);
        vst1q_s16(q_ptr.add(i), vqmovn_high_s32(vqmovn_s32(q0), q1));
        i += 8;
    }
    let mut isum = vaddvq_s32(vaddq_s32(vsum0, vsum1));
    while i < len {
        let e = ((*ptr.add(i) - max_val).exp2() * P_Q16_MAX).round() as i32;
        let q = e.clamp(0, P_Q16_MAX as i32);
        *q_ptr.add(i) = q as i16;
        isum += q;
        i += 1;
    }
    1.0 / (isum.max(1) as f32)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_probs_16_i8_neon(
    probs_blk_i8: *const i8,
    seq_stride: usize,
    valid_q: usize,
    tk_steps_aligned: usize,
    p_t_i8: *mut i8,
) {
    let zero = vdupq_n_s32(0);
    let src_i32 = probs_blk_i8 as *const i32;
    let dst_i32 = p_t_i8 as *mut i32;
    let stride_i32 = seq_stride / 4;

    let mut ks = 0usize;
    while ks + 4 <= tk_steps_aligned {
        let dst = dst_i32.add(ks * 16);
        for rg in 0..4 {
            let r0 = rg * 4;
            if r0 >= valid_q {
                vst1q_s32(dst.add(r0), zero);
                vst1q_s32(dst.add(16 + r0), zero);
                vst1q_s32(dst.add(32 + r0), zero);
                vst1q_s32(dst.add(48 + r0), zero);
                continue;
            }
            let v0 = vld1q_s32(src_i32.add(r0 * stride_i32 + ks));
            let v1 = if r0 + 1 < valid_q {
                vld1q_s32(src_i32.add((r0 + 1) * stride_i32 + ks))
            } else {
                zero
            };
            let v2 = if r0 + 2 < valid_q {
                vld1q_s32(src_i32.add((r0 + 2) * stride_i32 + ks))
            } else {
                zero
            };
            let v3 = if r0 + 3 < valid_q {
                vld1q_s32(src_i32.add((r0 + 3) * stride_i32 + ks))
            } else {
                zero
            };

            let t0 = vtrn1q_s32(v0, v1);
            let t1 = vtrn2q_s32(v0, v1);
            let t2 = vtrn1q_s32(v2, v3);
            let t3 = vtrn2q_s32(v2, v3);

            vst1q_s32(
                dst.add(r0),
                vcombine_s32(vget_low_s32(t0), vget_low_s32(t2)),
            );
            vst1q_s32(
                dst.add(16 + r0),
                vcombine_s32(vget_low_s32(t1), vget_low_s32(t3)),
            );
            vst1q_s32(
                dst.add(32 + r0),
                vcombine_s32(vget_high_s32(t0), vget_high_s32(t2)),
            );
            vst1q_s32(
                dst.add(48 + r0),
                vcombine_s32(vget_high_s32(t1), vget_high_s32(t3)),
            );
        }
        ks += 4;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn quantize_head32_row_neon(
    src: *const f32,
    out_scale_mul: f32,
) -> (int32x4_t, int32x4_t, f32) {
    let v0 = vld1q_f32(src);
    let v1 = vld1q_f32(src.add(4));
    let v2 = vld1q_f32(src.add(8));
    let v3 = vld1q_f32(src.add(12));
    let v4 = vld1q_f32(src.add(16));
    let v5 = vld1q_f32(src.add(20));
    let v6 = vld1q_f32(src.add(24));
    let v7 = vld1q_f32(src.add(28));

    let m01 = vmaxq_f32(vabsq_f32(v0), vabsq_f32(v1));
    let m23 = vmaxq_f32(vabsq_f32(v2), vabsq_f32(v3));
    let m45 = vmaxq_f32(vabsq_f32(v4), vabsq_f32(v5));
    let m67 = vmaxq_f32(vabsq_f32(v6), vabsq_f32(v7));
    let m03 = vmaxq_f32(m01, m23);
    let m47 = vmaxq_f32(m45, m67);
    let max_abs = vmaxvq_f32(vmaxq_f32(m03, m47)).max(1e-8);

    let inv_s = vdupq_n_f32(127.0 / max_abs);
    let q0 = vcvtnq_s32_f32(vmulq_f32(v0, inv_s));
    let q1 = vcvtnq_s32_f32(vmulq_f32(v1, inv_s));
    let q2 = vcvtnq_s32_f32(vmulq_f32(v2, inv_s));
    let q3 = vcvtnq_s32_f32(vmulq_f32(v3, inv_s));
    let q4 = vcvtnq_s32_f32(vmulq_f32(v4, inv_s));
    let q5 = vcvtnq_s32_f32(vmulq_f32(v5, inv_s));
    let q6 = vcvtnq_s32_f32(vmulq_f32(v6, inv_s));
    let q7 = vcvtnq_s32_f32(vmulq_f32(v7, inv_s));

    let lo_16a = vqmovn_high_s32(vqmovn_s32(q0), q1);
    let lo_16b = vqmovn_high_s32(vqmovn_s32(q2), q3);
    let lo_i8 = vqmovn_high_s16(vqmovn_s16(lo_16a), lo_16b);

    let hi_16a = vqmovn_high_s32(vqmovn_s32(q4), q5);
    let hi_16b = vqmovn_high_s32(vqmovn_s32(q6), q7);
    let hi_i8 = vqmovn_high_s16(vqmovn_s16(hi_16a), hi_16b);

    (
        vreinterpretq_s32_s8(lo_i8),
        vreinterpretq_s32_s8(hi_i8),
        (max_abs / 127.0) * out_scale_mul,
    )
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn quantize_v_row_neon(
    src: *const f32,
    inv0: float32x4_t,
    inv1: float32x4_t,
    inv2: float32x4_t,
    inv3: float32x4_t,
    inv4: float32x4_t,
    inv5: float32x4_t,
    inv6: float32x4_t,
    inv7: float32x4_t,
) -> (int8x16_t, int8x16_t) {
    let q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src), inv0));
    let q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.add(4)), inv1));
    let q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.add(8)), inv2));
    let q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.add(12)), inv3));
    let q4 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.add(16)), inv4));
    let q5 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.add(20)), inv5));
    let q6 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.add(24)), inv6));
    let q7 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(src.add(28)), inv7));

    let lo_16a = vqmovn_high_s32(vqmovn_s32(q0), q1);
    let lo_16b = vqmovn_high_s32(vqmovn_s32(q2), q3);
    let lo_i8 = vqmovn_high_s16(vqmovn_s16(lo_16a), lo_16b);

    let hi_16a = vqmovn_high_s32(vqmovn_s32(q4), q5);
    let hi_16b = vqmovn_high_s32(vqmovn_s32(q6), q7);
    let hi_i8 = vqmovn_high_s16(vqmovn_s16(hi_16a), hi_16b);

    (lo_i8, hi_i8)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn pack_qkv_head_int8_neon(
    qkv: &[f32],
    seq_len: usize,
    seq_stride: usize,
    tk_steps: usize,
    h_off: usize,
    k_scale_mul: f32,
    q_t_i8: *mut i8,
    k_t_i8: *mut i8,
    q_scale: *mut f32,
    k_scale: *mut f32,
    v_packed_i16: *mut i16,
    v_scale: *mut f32,
) {
    const HIDDEN: usize = 384;
    let m_blocks = seq_len.div_ceil(16);
    let stride = 3 * HIDDEN;
    let zero_i32 = vdupq_n_s32(0);
    let zero_i8 = vdupq_n_s8(0);
    let qkv_ptr = qkv.as_ptr();

    // 1. Find per-column max_abs across `t in 0..seq_len` for V_h (`[seq_len, 32]`) and pack V_h into `[tk_steps, 2, 64]` i8
    let mut vm0 = vdupq_n_f32(1e-8);
    let mut vm1 = vdupq_n_f32(1e-8);
    let mut vm2 = vdupq_n_f32(1e-8);
    let mut vm3 = vdupq_n_f32(1e-8);
    let mut vm4 = vdupq_n_f32(1e-8);
    let mut vm5 = vdupq_n_f32(1e-8);
    let mut vm6 = vdupq_n_f32(1e-8);
    let mut vm7 = vdupq_n_f32(1e-8);

    for t in 0..seq_len {
        let src = qkv_ptr.add(t * stride + 2 * HIDDEN + h_off);
        vm0 = vmaxq_f32(vm0, vabsq_f32(vld1q_f32(src)));
        vm1 = vmaxq_f32(vm1, vabsq_f32(vld1q_f32(src.add(4))));
        vm2 = vmaxq_f32(vm2, vabsq_f32(vld1q_f32(src.add(8))));
        vm3 = vmaxq_f32(vm3, vabsq_f32(vld1q_f32(src.add(12))));
        vm4 = vmaxq_f32(vm4, vabsq_f32(vld1q_f32(src.add(16))));
        vm5 = vmaxq_f32(vm5, vabsq_f32(vld1q_f32(src.add(20))));
        vm6 = vmaxq_f32(vm6, vabsq_f32(vld1q_f32(src.add(24))));
        vm7 = vmaxq_f32(vm7, vabsq_f32(vld1q_f32(src.add(28))));
    }

    let v_inv_127 = vdupq_n_f32(1.0 / 127.0);
    let v127 = vdupq_n_f32(127.0);
    vst1q_f32(v_scale, vmulq_f32(vm0, v_inv_127));
    vst1q_f32(v_scale.add(4), vmulq_f32(vm1, v_inv_127));
    vst1q_f32(v_scale.add(8), vmulq_f32(vm2, v_inv_127));
    vst1q_f32(v_scale.add(12), vmulq_f32(vm3, v_inv_127));
    vst1q_f32(v_scale.add(16), vmulq_f32(vm4, v_inv_127));
    vst1q_f32(v_scale.add(20), vmulq_f32(vm5, v_inv_127));
    vst1q_f32(v_scale.add(24), vmulq_f32(vm6, v_inv_127));
    vst1q_f32(v_scale.add(28), vmulq_f32(vm7, v_inv_127));

    let inv0 = vdivq_f32(v127, vm0);
    let inv1 = vdivq_f32(v127, vm1);
    let inv2 = vdivq_f32(v127, vm2);
    let inv3 = vdivq_f32(v127, vm3);
    let inv4 = vdivq_f32(v127, vm4);
    let inv5 = vdivq_f32(v127, vm5);
    let inv6 = vdivq_f32(v127, vm6);
    let inv7 = vdivq_f32(v127, vm7);

    // Pack V_h as int16 key pairs for 2-way smopa: for pair step `ks` (keys 2ks, 2ks+1), column c occupies
    // 4 bytes `[v(2ks, c), v(2ks+1, c)]`, columns 0..15 in the first 64 bytes and 16..31 in the next 64.
    for ks in 0..tk_steps {
        let t0 = ks * 2;
        let dst = v_packed_i16.add(ks * 64);
        let (r0_lo, r0_hi) = quantize_v_row_neon(
            qkv_ptr.add(t0 * stride + 2 * HIDDEN + h_off),
            inv0,
            inv1,
            inv2,
            inv3,
            inv4,
            inv5,
            inv6,
            inv7,
        );
        let (r1_lo, r1_hi) = if t0 + 1 < seq_len {
            quantize_v_row_neon(
                qkv_ptr.add((t0 + 1) * stride + 2 * HIDDEN + h_off),
                inv0,
                inv1,
                inv2,
                inv3,
                inv4,
                inv5,
                inv6,
                inv7,
            )
        } else {
            (zero_i8, zero_i8)
        };
        let a = [
            vmovl_s8(vget_low_s8(r0_lo)),
            vmovl_high_s8(r0_lo),
            vmovl_s8(vget_low_s8(r0_hi)),
            vmovl_high_s8(r0_hi),
        ];
        let b = [
            vmovl_s8(vget_low_s8(r1_lo)),
            vmovl_high_s8(r1_lo),
            vmovl_s8(vget_low_s8(r1_hi)),
            vmovl_high_s8(r1_hi),
        ];
        for g in 0..4 {
            vst1q_s16(dst.add(g * 16), vzip1q_s16(a[g], b[g]));
            vst1q_s16(dst.add(g * 16 + 8), vzip2q_s16(a[g], b[g]));
        }
    }

    // 2. Quantize & pack Q_h and K_h into `[m_blocks, 8, 16]` i32s
    for mb in 0..m_blocks {
        let t_base = mb * 16;
        let valid = (seq_len - t_base).min(16);
        let q_blk = (q_t_i8 as *mut i32).add(mb * 128);
        let k_blk = (k_t_i8 as *mut i32).add(mb * 128);

        for rg in 0..4 {
            let r0 = rg * 4;
            let (q0_lo, q0_hi, qs0) = if r0 < valid {
                quantize_head32_row_neon(qkv_ptr.add((t_base + r0) * stride + h_off), 1.0)
            } else {
                (zero_i32, zero_i32, 0.0)
            };
            let (q1_lo, q1_hi, qs1) = if r0 + 1 < valid {
                quantize_head32_row_neon(qkv_ptr.add((t_base + r0 + 1) * stride + h_off), 1.0)
            } else {
                (zero_i32, zero_i32, 0.0)
            };
            let (q2_lo, q2_hi, qs2) = if r0 + 2 < valid {
                quantize_head32_row_neon(qkv_ptr.add((t_base + r0 + 2) * stride + h_off), 1.0)
            } else {
                (zero_i32, zero_i32, 0.0)
            };
            let (q3_lo, q3_hi, qs3) = if r0 + 3 < valid {
                quantize_head32_row_neon(qkv_ptr.add((t_base + r0 + 3) * stride + h_off), 1.0)
            } else {
                (zero_i32, zero_i32, 0.0)
            };

            *q_scale.add(t_base + r0) = qs0;
            *q_scale.add(t_base + r0 + 1) = qs1;
            *q_scale.add(t_base + r0 + 2) = qs2;
            *q_scale.add(t_base + r0 + 3) = qs3;

            let qt0 = vtrn1q_s32(q0_lo, q1_lo);
            let qt1 = vtrn2q_s32(q0_lo, q1_lo);
            let qt2 = vtrn1q_s32(q2_lo, q3_lo);
            let qt3 = vtrn2q_s32(q2_lo, q3_lo);
            vst1q_s32(
                q_blk.add(r0),
                vcombine_s32(vget_low_s32(qt0), vget_low_s32(qt2)),
            );
            vst1q_s32(
                q_blk.add(16 + r0),
                vcombine_s32(vget_low_s32(qt1), vget_low_s32(qt3)),
            );
            vst1q_s32(
                q_blk.add(32 + r0),
                vcombine_s32(vget_high_s32(qt0), vget_high_s32(qt2)),
            );
            vst1q_s32(
                q_blk.add(48 + r0),
                vcombine_s32(vget_high_s32(qt1), vget_high_s32(qt3)),
            );

            let qt4 = vtrn1q_s32(q0_hi, q1_hi);
            let qt5 = vtrn2q_s32(q0_hi, q1_hi);
            let qt6 = vtrn1q_s32(q2_hi, q3_hi);
            let qt7 = vtrn2q_s32(q2_hi, q3_hi);
            vst1q_s32(
                q_blk.add(64 + r0),
                vcombine_s32(vget_low_s32(qt4), vget_low_s32(qt6)),
            );
            vst1q_s32(
                q_blk.add(80 + r0),
                vcombine_s32(vget_low_s32(qt5), vget_low_s32(qt7)),
            );
            vst1q_s32(
                q_blk.add(96 + r0),
                vcombine_s32(vget_high_s32(qt4), vget_high_s32(qt6)),
            );
            vst1q_s32(
                q_blk.add(112 + r0),
                vcombine_s32(vget_high_s32(qt5), vget_high_s32(qt7)),
            );

            let (k0_lo, k0_hi, ks0) = if r0 < valid {
                quantize_head32_row_neon(
                    qkv_ptr.add((t_base + r0) * stride + HIDDEN + h_off),
                    k_scale_mul,
                )
            } else {
                (zero_i32, zero_i32, 0.0)
            };
            let (k1_lo, k1_hi, ks1) = if r0 + 1 < valid {
                quantize_head32_row_neon(
                    qkv_ptr.add((t_base + r0 + 1) * stride + HIDDEN + h_off),
                    k_scale_mul,
                )
            } else {
                (zero_i32, zero_i32, 0.0)
            };
            let (k2_lo, k2_hi, ks2) = if r0 + 2 < valid {
                quantize_head32_row_neon(
                    qkv_ptr.add((t_base + r0 + 2) * stride + HIDDEN + h_off),
                    k_scale_mul,
                )
            } else {
                (zero_i32, zero_i32, 0.0)
            };
            let (k3_lo, k3_hi, ks3) = if r0 + 3 < valid {
                quantize_head32_row_neon(
                    qkv_ptr.add((t_base + r0 + 3) * stride + HIDDEN + h_off),
                    k_scale_mul,
                )
            } else {
                (zero_i32, zero_i32, 0.0)
            };

            *k_scale.add(t_base + r0) = ks0;
            *k_scale.add(t_base + r0 + 1) = ks1;
            *k_scale.add(t_base + r0 + 2) = ks2;
            *k_scale.add(t_base + r0 + 3) = ks3;

            let kt0 = vtrn1q_s32(k0_lo, k1_lo);
            let kt1 = vtrn2q_s32(k0_lo, k1_lo);
            let kt2 = vtrn1q_s32(k2_lo, k3_lo);
            let kt3 = vtrn2q_s32(k2_lo, k3_lo);
            vst1q_s32(
                k_blk.add(r0),
                vcombine_s32(vget_low_s32(kt0), vget_low_s32(kt2)),
            );
            vst1q_s32(
                k_blk.add(16 + r0),
                vcombine_s32(vget_low_s32(kt1), vget_low_s32(kt3)),
            );
            vst1q_s32(
                k_blk.add(32 + r0),
                vcombine_s32(vget_high_s32(kt0), vget_high_s32(kt2)),
            );
            vst1q_s32(
                k_blk.add(48 + r0),
                vcombine_s32(vget_high_s32(kt1), vget_high_s32(kt3)),
            );

            let kt4 = vtrn1q_s32(k0_hi, k1_hi);
            let kt5 = vtrn2q_s32(k0_hi, k1_hi);
            let kt6 = vtrn1q_s32(k2_hi, k3_hi);
            let kt7 = vtrn2q_s32(k2_hi, k3_hi);
            vst1q_s32(
                k_blk.add(64 + r0),
                vcombine_s32(vget_low_s32(kt4), vget_low_s32(kt6)),
            );
            vst1q_s32(
                k_blk.add(80 + r0),
                vcombine_s32(vget_low_s32(kt5), vget_low_s32(kt7)),
            );
            vst1q_s32(
                k_blk.add(96 + r0),
                vcombine_s32(vget_high_s32(kt4), vget_high_s32(kt6)),
            );
            vst1q_s32(
                k_blk.add(112 + r0),
                vcombine_s32(vget_high_s32(kt5), vget_high_s32(kt7)),
            );
        }
    }

    for t in (m_blocks * 16)..seq_stride {
        *k_scale.add(t) = 0.0;
    }
}

/// Copies `k_head[0..seq_len, 0..32]` and `v_head[0..seq_len, 0..32]` for head `h_off` from `qkv_buf` (`stride = 1152` or `768`).
#[inline(always)]
pub fn extract_kv_head32(
    qkv_buf: &[f32],
    seq_len: usize,
    row_stride: usize,
    k_col_off: usize,
    v_col_off: usize,
    k_head: &mut [f32],
    v_head: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let src_base = qkv_buf.as_ptr();
        let k_dst = k_head.as_mut_ptr();
        let v_dst = v_head.as_mut_ptr();
        for t in 0..seq_len {
            let row = src_base.add(t * row_stride);
            let ks = row.add(k_col_off);
            let vs = row.add(v_col_off);
            let kd = k_dst.add(t * 32);
            let vd = v_dst.add(t * 32);
            vst1q_f32(kd, vld1q_f32(ks));
            vst1q_f32(kd.add(4), vld1q_f32(ks.add(4)));
            vst1q_f32(kd.add(8), vld1q_f32(ks.add(8)));
            vst1q_f32(kd.add(12), vld1q_f32(ks.add(12)));
            vst1q_f32(kd.add(16), vld1q_f32(ks.add(16)));
            vst1q_f32(kd.add(20), vld1q_f32(ks.add(20)));
            vst1q_f32(kd.add(24), vld1q_f32(ks.add(24)));
            vst1q_f32(kd.add(28), vld1q_f32(ks.add(28)));

            vst1q_f32(vd, vld1q_f32(vs));
            vst1q_f32(vd.add(4), vld1q_f32(vs.add(4)));
            vst1q_f32(vd.add(8), vld1q_f32(vs.add(8)));
            vst1q_f32(vd.add(12), vld1q_f32(vs.add(12)));
            vst1q_f32(vd.add(16), vld1q_f32(vs.add(16)));
            vst1q_f32(vd.add(20), vld1q_f32(vs.add(20)));
            vst1q_f32(vd.add(24), vld1q_f32(vs.add(24)));
            vst1q_f32(vd.add(28), vld1q_f32(vs.add(28)));
        }
        return;
    }
    #[allow(unreachable_code)]
    for t in 0..seq_len {
        let row = &qkv_buf[t * row_stride..(t + 1) * row_stride];
        k_head[t * 32..(t + 1) * 32].copy_from_slice(&row[k_col_off..k_col_off + 32]);
        v_head[t * 32..(t + 1) * 32].copy_from_slice(&row[v_col_off..v_col_off + 32]);
    }
}

/// L2-normalizes a slice in-place.
#[inline]
pub fn l2_normalize_inplace(v: &mut [f32]) {
    let mut sum_sq = 0.0f32;
    for &x in v.iter() {
        sum_sq += x * x;
    }
    if sum_sq > 0.0 {
        let inv_norm = 1.0 / sum_sq.sqrt();
        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// x86_64 AVX2+FMA attention kernels (single-sequence, ragged and `[CLS]` attention).
//
// These must be `#[target_feature]` functions: intrinsics called from a function compiled
// without AVX2 cannot be inlined, so each one becomes an out-of-line call passing vectors
// through memory (this made x86 attention ~an order of magnitude slower than NEON).
// ---------------------------------------------------------------------------------------------

/// Runtime check for the AVX2+FMA attention kernels (`is_x86_feature_detected!` caches).
#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2_fma() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot32_f32_avx2(pa: *const f32, pb: *const f32) -> f32 {
    let mut s0 = _mm256_mul_ps(_mm256_loadu_ps(pa), _mm256_loadu_ps(pb));
    let mut s1 = _mm256_mul_ps(_mm256_loadu_ps(pa.add(8)), _mm256_loadu_ps(pb.add(8)));
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(16)), _mm256_loadu_ps(pb.add(16)), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(24)), _mm256_loadu_ps(pb.add(24)), s1);
    hsum256_ps(_mm256_add_ps(s0, s1))
}

/// Partial dot product of a 32-float key with `q` held in 4 registers (8 lanes left to reduce).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dot32x4_avx2(q0: __m256, q1: __m256, q2: __m256, q3: __m256, kp: *const f32) -> __m256 {
    let a = _mm256_fmadd_ps(
        q1,
        _mm256_loadu_ps(kp.add(8)),
        _mm256_mul_ps(q0, _mm256_loadu_ps(kp)),
    );
    let b = _mm256_fmadd_ps(
        q3,
        _mm256_loadu_ps(kp.add(24)),
        _mm256_mul_ps(q2, _mm256_loadu_ps(kp.add(16))),
    );
    _mm256_add_ps(a, b)
}

/// `scores[tk] = dot(q * scale, k[tk])` for `seq_len` keys of 32 floats: `q` lives in 4 registers,
/// 4 keys per iteration, reduced with a `hadd` tree into one 4-wide store.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_scores_head32_avx2(
    q: *const f32,
    k: *const f32,
    scale: f32,
    sp: *mut f32,
    seq_len: usize,
) {
    let vs = _mm256_set1_ps(scale);
    let q0 = _mm256_mul_ps(_mm256_loadu_ps(q), vs);
    let q1 = _mm256_mul_ps(_mm256_loadu_ps(q.add(8)), vs);
    let q2 = _mm256_mul_ps(_mm256_loadu_ps(q.add(16)), vs);
    let q3 = _mm256_mul_ps(_mm256_loadu_ps(q.add(24)), vs);
    let mut tk = 0usize;
    while tk + 4 <= seq_len {
        let k0 = k.add(tk * 32);
        let (s0, s1, s2, s3) = (
            dot32x4_avx2(q0, q1, q2, q3, k0),
            dot32x4_avx2(q0, q1, q2, q3, k0.add(32)),
            dot32x4_avx2(q0, q1, q2, q3, k0.add(64)),
            dot32x4_avx2(q0, q1, q2, q3, k0.add(96)),
        );
        // Per 128-bit half: [sum4(s0), sum4(s1), sum4(s2), sum4(s3)]; adding the halves finishes the dots.
        let t = _mm256_hadd_ps(_mm256_hadd_ps(s0, s1), _mm256_hadd_ps(s2, s3));
        let r = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
        _mm_storeu_ps(sp.add(tk), r);
        tk += 4;
    }
    while tk < seq_len {
        *sp.add(tk) = hsum256_ps(dot32x4_avx2(q0, q1, q2, q3, k.add(tk * 32)));
        tk += 1;
    }
}

/// In-place softmax, 8 lanes at a time with [`avx2_exp_f32`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_inplace_avx2(scores: &mut [f32]) {
    let len = scores.len();
    let ptr = scores.as_mut_ptr();
    let mut i = 0usize;
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    while i + 8 <= len {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(ptr.add(i)));
        i += 8;
    }
    let mut max_val = hmax256_ps(vmax);
    while i < len {
        max_val = max_val.max(*ptr.add(i));
        i += 1;
    }

    let vm = _mm256_set1_ps(max_val);
    let mut vsum0 = _mm256_setzero_ps();
    let mut vsum1 = _mm256_setzero_ps();
    i = 0;
    while i + 16 <= len {
        let e0 = avx2_exp_f32(_mm256_sub_ps(_mm256_loadu_ps(ptr.add(i)), vm));
        let e1 = avx2_exp_f32(_mm256_sub_ps(_mm256_loadu_ps(ptr.add(i + 8)), vm));
        _mm256_storeu_ps(ptr.add(i), e0);
        _mm256_storeu_ps(ptr.add(i + 8), e1);
        vsum0 = _mm256_add_ps(vsum0, e0);
        vsum1 = _mm256_add_ps(vsum1, e1);
        i += 16;
    }
    while i + 8 <= len {
        let e = avx2_exp_f32(_mm256_sub_ps(_mm256_loadu_ps(ptr.add(i)), vm));
        _mm256_storeu_ps(ptr.add(i), e);
        vsum0 = _mm256_add_ps(vsum0, e);
        i += 8;
    }
    let mut sum = hsum256_ps(_mm256_add_ps(vsum0, vsum1));
    while i < len {
        let e = (*ptr.add(i) - max_val).exp();
        *ptr.add(i) = e;
        sum += e;
        i += 1;
    }

    let inv = 1.0 / sum;
    let vinv = _mm256_set1_ps(inv);
    i = 0;
    while i + 8 <= len {
        _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), vinv));
        i += 8;
    }
    while i < len {
        *ptr.add(i) *= inv;
        i += 1;
    }
}

/// `out[0..32] = sum_t p[t] * v[t]`, two rows per iteration into 8 independent accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn weighted_sum_head32_avx2(p: *const f32, v: *const f32, seq_len: usize, out: *mut f32) {
    let mut a = [_mm256_setzero_ps(); 4];
    let mut b = [_mm256_setzero_ps(); 4];
    let mut t = 0usize;
    while t + 2 <= seq_len {
        let (pa, pb) = (_mm256_set1_ps(*p.add(t)), _mm256_set1_ps(*p.add(t + 1)));
        let (ra, rb) = (v.add(t * 32), v.add(t * 32 + 32));
        for j in 0..4 {
            a[j] = _mm256_fmadd_ps(_mm256_loadu_ps(ra.add(8 * j)), pa, a[j]);
            b[j] = _mm256_fmadd_ps(_mm256_loadu_ps(rb.add(8 * j)), pb, b[j]);
        }
        t += 2;
    }
    if t < seq_len {
        let pa = _mm256_set1_ps(*p.add(t));
        let ra = v.add(t * 32);
        for j in 0..4 {
            a[j] = _mm256_fmadd_ps(_mm256_loadu_ps(ra.add(8 * j)), pa, a[j]);
        }
    }
    for j in 0..4 {
        _mm256_storeu_ps(out.add(8 * j), _mm256_add_ps(a[j], b[j]));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hmax256_ps(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let m128 = _mm_max_ps(lo, hi);
    let shuf = _mm_movehdup_ps(m128);
    let m64 = _mm_max_ps(m128, shuf);
    let shuf2 = _mm_movehl_ps(m64, m64);
    _mm_cvtss_f32(_mm_max_ss(m64, shuf2))
}
