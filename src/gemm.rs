//! Multi-Tier SIMD Int8 Matrix Multiplication (`s8 * s8 -> s32 -> f32`) with fused epilogues.
//!
//! Tiers (selected automatically at initialization, or configurable for benchmarking):
//! 1. `ArmI8mm`: ARMv8.6 `FEAT_I8MM` (`smmla` 2x8x2 matrix multiply-accumulate, 32 MACs/inst)
//! 2. `ArmDotProd`: ARMv8.2 `FEAT_DotProd` (`sdot` 4-way dot product, 16 MACs/inst)
//! 3. `ArmNeon`: ARMv8.0 baseline NEON (`vmull_s8` + `vpadalq_s16`)
//! 4. `X86AvxVnni`: x86_64 AVX-VNNI (`vpdpbusd`, 32 MACs/inst)
//! 5. `X86Avx2`: x86_64 AVX2 (`vpsignb` + `vpmaddubsw` + `vpmaddwd`, saturation-free)
//! 6. `Scalar`: Portable unrolled fallback

use crate::aligned::AlignedBuf;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    ArmSme,
    ArmI8mm,
    ArmDotProd,
    ArmNeon,
    X86AvxVnni,
    X86Avx2,
    Scalar,
}

#[cfg(target_arch = "aarch64")]
fn detect_arm_sme() -> bool {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn sysctlbyname(
                name: *const std::ffi::c_char,
                oldp: *mut std::ffi::c_void,
                oldlenp: *mut usize,
                newp: *const std::ffi::c_void,
                newlen: usize,
            ) -> i32;
        }
        let mut val: i32 = 0;
        let mut size = std::mem::size_of::<i32>();
        let res = unsafe {
            sysctlbyname(
                c"hw.optional.arm.FEAT_SME".as_ptr(),
                &mut val as *mut i32 as *mut _,
                &mut size,
                std::ptr::null(),
                0,
            )
        };
        return res == 0 && val == 1;
    }
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn getauxval(type_: u64) -> u64;
        }
        const AT_HWCAP2: u64 = 26;
        const HWCAP2_SME: u64 = 1 << 23;
        return unsafe { (getauxval(AT_HWCAP2) & HWCAP2_SME) != 0 };
    }
    #[allow(unreachable_code)]
    false
}

impl SimdBackend {
    pub fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            if detect_arm_sme() {
                return SimdBackend::ArmSme;
            }
            if std::arch::is_aarch64_feature_detected!("i8mm") {
                return SimdBackend::ArmI8mm;
            }
            if std::arch::is_aarch64_feature_detected!("dotprod") {
                return SimdBackend::ArmDotProd;
            }
            return SimdBackend::ArmNeon;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                if is_x86_feature_detected!("avxvnni") {
                    return SimdBackend::X86AvxVnni;
                }
                return SimdBackend::X86Avx2;
            }
        }
        #[allow(unreachable_code)]
        SimdBackend::Scalar
    }

    pub fn name(&self) -> &'static str {
        match self {
            SimdBackend::ArmSme => "ARM64 FEAT_SME (smopa 16x64)",
            SimdBackend::ArmI8mm => "ARM64 FEAT_I8MM (smmla 8x8)",
            SimdBackend::ArmDotProd => "ARM64 FEAT_DotProd (sdot 4x8)",
            SimdBackend::ArmNeon => "ARM64 NEON (vmull_s8)",
            SimdBackend::X86AvxVnni => "x86_64 AVX-VNNI (vpdpbusd 4x16)",
            SimdBackend::X86Avx2 => "x86_64 AVX2 (vpmaddubsw 4x16)",
            SimdBackend::Scalar => "Portable Scalar",
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            SimdBackend::ArmSme => 1,
            SimdBackend::ArmI8mm => 2,
            SimdBackend::ArmDotProd => 3,
            SimdBackend::ArmNeon => 4,
            SimdBackend::X86AvxVnni => 5,
            SimdBackend::X86Avx2 => 6,
            SimdBackend::Scalar => 7,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(SimdBackend::ArmSme),
            2 => Some(SimdBackend::ArmI8mm),
            3 => Some(SimdBackend::ArmDotProd),
            4 => Some(SimdBackend::ArmNeon),
            5 => Some(SimdBackend::X86AvxVnni),
            6 => Some(SimdBackend::X86Avx2),
            7 => Some(SimdBackend::Scalar),
            _ => None,
        }
    }
}

/// Pre-packed Int8 weight matrix `W` of logical shape `[N, K]` (out_features x in_features),
/// together with per-output-channel dequantization scales `w_scale[N]`, bias `bias[N]`,
/// and SmoothQuant input scaling `inv_smooth[K]`.
#[derive(Clone)]
pub struct PackedLinearInt8 {
    pub n_out: usize,
    pub k_in: usize,
    pub backend: SimdBackend,
    /// Packed weight bytes in the layout required by `backend`.
    pub packed_w: AlignedBuf<i8>,
    /// Per-output-channel weight scale (`max_abs_row / 127.0`), length `n_out`.
    pub w_scale: AlignedBuf<f32>,
    /// Output bias, length `n_out`.
    pub bias: AlignedBuf<f32>,
    /// SmoothQuant inverse input scale (`1.0 / s_j`), length `k_in`.
    pub inv_smooth: AlignedBuf<f32>,
    /// Column sums `-128 * sum_k(W[n, k])` for unsigned-activation VNNI compensation.
    pub col_sum_comp: AlignedBuf<i32>,
}

impl PackedLinearInt8 {
    pub fn from_raw_row_major(
        n_out: usize,
        k_in: usize,
        w_row_major: &[i8],
        w_scale: &[f32],
        bias: &[f32],
        inv_smooth: &[f32],
        backend: SimdBackend,
    ) -> Self {
        if n_out == 0 {
            return Self {
                n_out: 0,
                k_in,
                backend,
                packed_w: AlignedBuf::new_zeroed(0),
                w_scale: AlignedBuf::new_zeroed(0),
                bias: AlignedBuf::new_zeroed(0),
                inv_smooth: AlignedBuf::from_slice(inv_smooth),
                col_sum_comp: AlignedBuf::new_zeroed(0),
            };
        }
        assert_eq!(w_row_major.len(), n_out * k_in);
        assert_eq!(w_scale.len(), n_out);
        assert_eq!(bias.len(), n_out);
        assert_eq!(inv_smooth.len(), k_in);
        assert_eq!(n_out % 16, 0, "n_out must be a multiple of 16");
        assert_eq!(k_in % 16, 0, "k_in must be a multiple of 16");

        let col_sum_comp = if backend == SimdBackend::X86AvxVnni {
            let mut comp = AlignedBuf::<i32>::new_zeroed(n_out);
            for n in 0..n_out {
                let row = &w_row_major[n * k_in..(n + 1) * k_in];
                let s: i32 = row.iter().map(|&v| v as i32).sum();
                comp[n] = -128 * s;
            }
            comp
        } else {
            AlignedBuf::new_zeroed(0)
        };

        let packed_w = match backend {
            SimdBackend::ArmSme => pack_weights_sme(n_out, k_in, w_row_major),
            SimdBackend::ArmI8mm => pack_weights_i8mm(n_out, k_in, w_row_major),
            SimdBackend::ArmDotProd => pack_weights_sdot(n_out, k_in, w_row_major),
            SimdBackend::X86AvxVnni | SimdBackend::X86Avx2 => {
                pack_weights_x86(n_out, k_in, w_row_major)
            }
            SimdBackend::ArmNeon | SimdBackend::Scalar => AlignedBuf::from_slice(w_row_major),
        };

        Self {
            n_out,
            k_in,
            backend,
            packed_w,
            w_scale: AlignedBuf::from_slice(w_scale),
            bias: AlignedBuf::from_slice(bias),
            inv_smooth: AlignedBuf::from_slice(inv_smooth),
            col_sum_comp,
        }
    }
}

/// Packs `[N, K]` into `[N / 64, K / 4, 4, 64]` for ARM64 `smopa` (`FEAT_SME`).
/// Uses 4x4 NEON `vtrn1q_s32`/`vtrn2q_s32` register transposes on `aarch64`.
pub fn pack_weights_sme(n_out: usize, k_in: usize, w: &[i8]) -> AlignedBuf<i8> {
    assert_eq!(n_out % 64, 0, "SME requires n_out % 64 == 0");
    let mut out = AlignedBuf::<i8>::new_zeroed(n_out * k_in);
    let n_super = n_out / 64;
    let k_steps = k_in / 4;

    #[cfg(target_arch = "aarch64")]
    if k_steps % 4 == 0 {
        unsafe {
            let w_i32 = w.as_ptr() as *const i32;
            let out_i32 = out.as_mut_ptr() as *mut i32;
            for nb in 0..n_super {
                let col_base = nb * 64;
                let nb_dst = out_i32.add(nb * k_steps * 64);
                for t in 0..4 {
                    let tile_col = col_base + t * 16;
                    let mut ks = 0;
                    while ks < k_steps {
                        let dst_ks = nb_dst.add(ks * 64 + t * 16);
                        for rg in 0..4 {
                            let c0 = rg * 4;
                            let r_ptr = w_i32.add((tile_col + c0) * k_steps + ks);
                            let v0 = vld1q_s32(r_ptr);
                            let v1 = vld1q_s32(r_ptr.add(k_steps));
                            let v2 = vld1q_s32(r_ptr.add(2 * k_steps));
                            let v3 = vld1q_s32(r_ptr.add(3 * k_steps));

                            let t0 = vtrn1q_s32(v0, v1);
                            let t1 = vtrn2q_s32(v0, v1);
                            let t2 = vtrn1q_s32(v2, v3);
                            let t3 = vtrn2q_s32(v2, v3);

                            vst1q_s32(
                                dst_ks.add(c0),
                                vcombine_s32(vget_low_s32(t0), vget_low_s32(t2)),
                            );
                            vst1q_s32(
                                dst_ks.add(64 + c0),
                                vcombine_s32(vget_low_s32(t1), vget_low_s32(t3)),
                            );
                            vst1q_s32(
                                dst_ks.add(128 + c0),
                                vcombine_s32(vget_high_s32(t0), vget_high_s32(t2)),
                            );
                            vst1q_s32(
                                dst_ks.add(192 + c0),
                                vcombine_s32(vget_high_s32(t1), vget_high_s32(t3)),
                            );
                        }
                        ks += 4;
                    }
                }
            }
        }
        return out;
    }

    let mut dst_idx = 0;
    for nb in 0..n_super {
        let col_base = nb * 64;
        for ks in 0..k_steps {
            let k_base = ks * 4;
            for t in 0..4 {
                for c in 0..16 {
                    let col = col_base + t * 16 + c;
                    out[dst_idx..dst_idx + 4]
                        .copy_from_slice(&w[col * k_in + k_base..col * k_in + k_base + 4]);
                    dst_idx += 4;
                }
            }
        }
    }
    out
}

/// Packs `[N, K]` into `[N / 8, K / 8, 4, 16]` for ARM64 `smmla` (`i8mm`).
/// Uses 64-bit NEON `vtrn1q_s64`/`vtrn2q_s64` interleaving on `aarch64`.
pub fn pack_weights_i8mm(n_out: usize, k_in: usize, w: &[i8]) -> AlignedBuf<i8> {
    let mut out = AlignedBuf::<i8>::new_zeroed(n_out * k_in);
    let n_panels = n_out / 8;
    let k_steps = k_in / 8;

    #[cfg(target_arch = "aarch64")]
    if k_steps % 2 == 0 {
        unsafe {
            let w_ptr = w.as_ptr();
            let out_ptr = out.as_mut_ptr();
            for nb in 0..n_panels {
                let col_base = nb * 8;
                let panel_dst = out_ptr.add(nb * k_steps * 64);
                let mut ks = 0;
                while ks < k_steps {
                    let kb = ks * 8;
                    let dst0 = panel_dst.add(ks * 64);
                    let dst1 = dst0.add(64);
                    for p in 0..4 {
                        let c0 = col_base + 2 * p;
                        let v0 = vreinterpretq_s64_s8(vld1q_s8(w_ptr.add(c0 * k_in + kb)));
                        let v1 = vreinterpretq_s64_s8(vld1q_s8(w_ptr.add((c0 + 1) * k_in + kb)));
                        let lo = vreinterpretq_s8_s64(vtrn1q_s64(v0, v1));
                        let hi = vreinterpretq_s8_s64(vtrn2q_s64(v0, v1));
                        vst1q_s8(dst0.add(p * 16), lo);
                        vst1q_s8(dst1.add(p * 16), hi);
                    }
                    ks += 2;
                }
            }
        }
        return out;
    }

    let mut dst_idx = 0;
    for nb in 0..n_panels {
        let col_base = nb * 8;
        for ks in 0..k_steps {
            let k_base = ks * 8;
            for p in 0..4 {
                let c0 = col_base + 2 * p;
                let c1 = c0 + 1;
                out[dst_idx..dst_idx + 8]
                    .copy_from_slice(&w[c0 * k_in + k_base..c0 * k_in + k_base + 8]);
                out[dst_idx + 8..dst_idx + 16]
                    .copy_from_slice(&w[c1 * k_in + k_base..c1 * k_in + k_base + 8]);
                dst_idx += 16;
            }
        }
    }
    out
}

/// Packs `[N, K]` into `[N / 8, K / 4, 2, 16]` for ARM64 `sdot` (`dotprod`).
pub fn pack_weights_sdot(n_out: usize, k_in: usize, w: &[i8]) -> AlignedBuf<i8> {
    let mut out = AlignedBuf::<i8>::new_zeroed(n_out * k_in);
    let n_panels = n_out / 8;
    let k_steps = k_in / 4;

    #[cfg(target_arch = "aarch64")]
    if k_steps % 4 == 0 {
        unsafe {
            let w_i32 = w.as_ptr() as *const i32;
            let out_i32 = out.as_mut_ptr() as *mut i32;
            for nb in 0..n_panels {
                let col_base = nb * 8;
                let panel_dst = out_i32.add(nb * k_steps * 8);
                let mut ks = 0;
                while ks < k_steps {
                    let dst_ks = panel_dst.add(ks * 8);
                    for g in 0..2 {
                        let c0 = col_base + g * 4;
                        let r_ptr = w_i32.add(c0 * k_steps + ks);
                        let v0 = vld1q_s32(r_ptr);
                        let v1 = vld1q_s32(r_ptr.add(k_steps));
                        let v2 = vld1q_s32(r_ptr.add(2 * k_steps));
                        let v3 = vld1q_s32(r_ptr.add(3 * k_steps));

                        let t0 = vtrn1q_s32(v0, v1);
                        let t1 = vtrn2q_s32(v0, v1);
                        let t2 = vtrn1q_s32(v2, v3);
                        let t3 = vtrn2q_s32(v2, v3);

                        vst1q_s32(
                            dst_ks.add(g * 4),
                            vcombine_s32(vget_low_s32(t0), vget_low_s32(t2)),
                        );
                        vst1q_s32(
                            dst_ks.add(8 + g * 4),
                            vcombine_s32(vget_low_s32(t1), vget_low_s32(t3)),
                        );
                        vst1q_s32(
                            dst_ks.add(16 + g * 4),
                            vcombine_s32(vget_high_s32(t0), vget_high_s32(t2)),
                        );
                        vst1q_s32(
                            dst_ks.add(24 + g * 4),
                            vcombine_s32(vget_high_s32(t1), vget_high_s32(t3)),
                        );
                    }
                    ks += 4;
                }
            }
        }
        return out;
    }

    let mut dst_idx = 0;
    for nb in 0..n_panels {
        let col_base = nb * 8;
        for ks in 0..k_steps {
            let k_base = ks * 4;
            for g in 0..2 {
                for c in 0..4 {
                    let col = col_base + g * 4 + c;
                    out[dst_idx..dst_idx + 4]
                        .copy_from_slice(&w[col * k_in + k_base..col * k_in + k_base + 4]);
                    dst_idx += 4;
                }
            }
        }
    }
    out
}

/// Packs `[N, K]` into `[N / 16, K / 4, 2, 32]` for x86_64 AVX2 / AVX-VNNI.
/// Within each 16-column panel `nb` and 4-element `k` step, stores 2 32-byte YMM vectors:
/// `g in 0..2`: `[W[nb*16 + 8*g + c, k..k+4] for c in 0..8]`.
pub fn pack_weights_x86(n_out: usize, k_in: usize, w: &[i8]) -> AlignedBuf<i8> {
    let mut out = AlignedBuf::<i8>::new_zeroed(n_out * k_in);
    let n_panels = n_out / 16;
    let k_steps = k_in / 4;
    let mut dst_idx = 0;
    for nb in 0..n_panels {
        let col_base = nb * 16;
        for ks in 0..k_steps {
            let k_base = ks * 4;
            for g in 0..2 {
                for c in 0..8 {
                    let col = col_base + g * 8 + c;
                    out[dst_idx..dst_idx + 4]
                        .copy_from_slice(&w[col * k_in + k_base..col * k_in + k_base + 4]);
                    dst_idx += 4;
                }
            }
        }
    }
    out
}

/// Packs activations `A` (`[M, K]` row-major) into the tile order expected by `ArmI8mm`:
/// - Full 8-row blocks (`m = 0, 8, 16, ...`): `[K / 8, 4_row_pairs, 16_bytes]`
/// - Optional 4-row block (`(M % 8) >= 4`): `[K / 8, 2_row_pairs, 16_bytes]`
/// - Optional 2-row block (`(M % 4) >= 2`): `[K / 8, 16_bytes]`
/// - Optional 1-row tail (`(M % 2) == 1`): `[K / 8, 16_bytes]` (row duplicated in low/high 8 bytes)
#[inline]
pub fn pack_activations_i8mm(m: usize, k: usize, a: &[i8], packed_a: &mut [i8]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        pack_activations_i8mm_neon(m, k, a, packed_a);
        return;
    }
    #[allow(unreachable_code)]
    {
        let k_steps = k / 8;
        let mut dst = 0;
        let mut r = 0;
        while r + 8 <= m {
            for ks in 0..k_steps {
                let kb = ks * 8;
                for p in 0..4 {
                    let r0 = r + 2 * p;
                    let r1 = r0 + 1;
                    packed_a[dst..dst + 8].copy_from_slice(&a[r0 * k + kb..r0 * k + kb + 8]);
                    packed_a[dst + 8..dst + 16].copy_from_slice(&a[r1 * k + kb..r1 * k + kb + 8]);
                    dst += 16;
                }
            }
            r += 8;
        }
        if r + 4 <= m {
            for ks in 0..k_steps {
                let kb = ks * 8;
                for p in 0..2 {
                    let r0 = r + 2 * p;
                    let r1 = r0 + 1;
                    packed_a[dst..dst + 8].copy_from_slice(&a[r0 * k + kb..r0 * k + kb + 8]);
                    packed_a[dst + 8..dst + 16].copy_from_slice(&a[r1 * k + kb..r1 * k + kb + 8]);
                    dst += 16;
                }
            }
            r += 4;
        }
        if r + 2 <= m {
            for ks in 0..k_steps {
                let kb = ks * 8;
                packed_a[dst..dst + 8].copy_from_slice(&a[r * k + kb..r * k + kb + 8]);
                packed_a[dst + 8..dst + 16]
                    .copy_from_slice(&a[(r + 1) * k + kb..(r + 1) * k + kb + 8]);
                dst += 16;
            }
            r += 2;
        }
        if r < m {
            for ks in 0..k_steps {
                let kb = ks * 8;
                packed_a[dst..dst + 8].copy_from_slice(&a[r * k + kb..r * k + kb + 8]);
                packed_a[dst + 8..dst + 16].copy_from_slice(&a[r * k + kb..r * k + kb + 8]);
                dst += 16;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pack_activations_i8mm_neon(m: usize, k: usize, a: &[i8], packed_a: &mut [i8]) {
    let k_steps = k / 8;
    let a_ptr = a.as_ptr();
    let mut dst_ptr = packed_a.as_mut_ptr();
    let mut r = 0;

    while r + 8 <= m {
        let r0 = a_ptr.add(r * k);
        let r1 = a_ptr.add((r + 1) * k);
        let r2 = a_ptr.add((r + 2) * k);
        let r3 = a_ptr.add((r + 3) * k);
        let r4 = a_ptr.add((r + 4) * k);
        let r5 = a_ptr.add((r + 5) * k);
        let r6 = a_ptr.add((r + 6) * k);
        let r7 = a_ptr.add((r + 7) * k);
        for ks in 0..k_steps {
            let kb = ks * 8;
            vst1q_s8(
                dst_ptr,
                vcombine_s8(vld1_s8(r0.add(kb)), vld1_s8(r1.add(kb))),
            );
            vst1q_s8(
                dst_ptr.add(16),
                vcombine_s8(vld1_s8(r2.add(kb)), vld1_s8(r3.add(kb))),
            );
            vst1q_s8(
                dst_ptr.add(32),
                vcombine_s8(vld1_s8(r4.add(kb)), vld1_s8(r5.add(kb))),
            );
            vst1q_s8(
                dst_ptr.add(48),
                vcombine_s8(vld1_s8(r6.add(kb)), vld1_s8(r7.add(kb))),
            );
            dst_ptr = dst_ptr.add(64);
        }
        r += 8;
    }

    if r + 4 <= m {
        let r0 = a_ptr.add(r * k);
        let r1 = a_ptr.add((r + 1) * k);
        let r2 = a_ptr.add((r + 2) * k);
        let r3 = a_ptr.add((r + 3) * k);
        for ks in 0..k_steps {
            let kb = ks * 8;
            vst1q_s8(
                dst_ptr,
                vcombine_s8(vld1_s8(r0.add(kb)), vld1_s8(r1.add(kb))),
            );
            vst1q_s8(
                dst_ptr.add(16),
                vcombine_s8(vld1_s8(r2.add(kb)), vld1_s8(r3.add(kb))),
            );
            dst_ptr = dst_ptr.add(32);
        }
        r += 4;
    }

    if r + 2 <= m {
        let r0 = a_ptr.add(r * k);
        let r1 = a_ptr.add((r + 1) * k);
        for ks in 0..k_steps {
            let kb = ks * 8;
            vst1q_s8(
                dst_ptr,
                vcombine_s8(vld1_s8(r0.add(kb)), vld1_s8(r1.add(kb))),
            );
            dst_ptr = dst_ptr.add(16);
        }
        r += 2;
    }

    if r < m {
        let r0 = a_ptr.add(r * k);
        for ks in 0..k_steps {
            let kb = ks * 8;
            let v = vld1_s8(r0.add(kb));
            vst1q_s8(dst_ptr, vcombine_s8(v, v));
            dst_ptr = dst_ptr.add(16);
        }
    }
}

/// Computes `out[m, n] = (sum_k a_q[m, k] * w_q[n, k]) * (a_scale[m] * w_scale[n]) + bias[n] (+ residual[m, n])`
/// using the optimal SIMD tier on the current CPU.
#[inline]
pub fn gemm_int8_fused(
    m: usize,
    a_q: &[i8],
    a_packed_scratch: &mut [i8],
    a_scale: &[f32],
    weights: &PackedLinearInt8,
    residual: Option<&[f32]>,
    out: &mut [f32],
) {
    if m == 0 {
        return;
    }
    match weights.backend {
        #[cfg(target_arch = "aarch64")]
        SimdBackend::ArmSme => unsafe {
            gemm_sme_fused(m, a_q, a_packed_scratch, a_scale, weights, residual, out);
        },
        #[cfg(target_arch = "aarch64")]
        SimdBackend::ArmI8mm => unsafe {
            gemm_i8mm_fused(m, a_q, a_packed_scratch, a_scale, weights, residual, out);
        },
        #[cfg(target_arch = "aarch64")]
        SimdBackend::ArmDotProd => unsafe {
            gemm_sdot_fused(m, a_q, a_scale, weights, residual, out);
        },
        #[cfg(target_arch = "aarch64")]
        SimdBackend::ArmNeon => unsafe {
            gemm_neon_fused(m, a_q, a_scale, weights, residual, out);
        },
        #[cfg(target_arch = "x86_64")]
        SimdBackend::X86AvxVnni => unsafe {
            gemm_x86_avx_vnni_fused(m, a_q, a_packed_scratch, a_scale, weights, residual, out);
        },
        #[cfg(target_arch = "x86_64")]
        SimdBackend::X86Avx2 => unsafe {
            gemm_x86_avx2_fused(m, a_q, a_scale, weights, residual, out);
        },
        _ => {
            gemm_scalar_fused(m, a_q, a_scale, weights, residual, out);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pack_activations_sme_neon(m: usize, k_in: usize, a: &[i8], out: &mut [i8]) {
    let m_blocks = m.div_ceil(16);
    let k_steps = k_in / 4;
    let a_i32 = a.as_ptr() as *const i32;
    let out_i32 = out.as_mut_ptr() as *mut i32;
    let zero_v = vdupq_n_s32(0);

    for rb in 0..m_blocks {
        let r_base = rb * 16;
        let valid_r = (m - r_base).min(16);
        let block_dst = out_i32.add(rb * k_steps * 16);

        if valid_r == 16 {
            let mut ks = 0;
            while ks + 4 <= k_steps {
                let dst_k = block_dst.add(ks * 16);
                for rg in 0..4 {
                    let row0 = rg * 4;
                    let base_r = (r_base + row0) * k_steps + ks;
                    let v0 = vld1q_s32(a_i32.add(base_r));
                    let v1 = vld1q_s32(a_i32.add(base_r + k_steps));
                    let v2 = vld1q_s32(a_i32.add(base_r + 2 * k_steps));
                    let v3 = vld1q_s32(a_i32.add(base_r + 3 * k_steps));

                    let t0 = vtrn1q_s32(v0, v1);
                    let t1 = vtrn2q_s32(v0, v1);
                    let t2 = vtrn1q_s32(v2, v3);
                    let t3 = vtrn2q_s32(v2, v3);

                    vst1q_s32(
                        dst_k.add(row0),
                        vcombine_s32(vget_low_s32(t0), vget_low_s32(t2)),
                    );
                    vst1q_f32_as_s32(
                        dst_k.add(16 + row0),
                        vcombine_s32(vget_low_s32(t1), vget_low_s32(t3)),
                    );
                    vst1q_s32(
                        dst_k.add(32 + row0),
                        vcombine_s32(vget_high_s32(t0), vget_high_s32(t2)),
                    );
                    vst1q_s32(
                        dst_k.add(48 + row0),
                        vcombine_s32(vget_high_s32(t1), vget_high_s32(t3)),
                    );
                }
                ks += 4;
            }
        } else {
            let mut ks = 0;
            while ks + 4 <= k_steps {
                let dst_k = block_dst.add(ks * 16);
                for rg in 0..4 {
                    let row0 = rg * 4;
                    if row0 >= valid_r {
                        vst1q_s32(dst_k.add(row0), zero_v);
                        vst1q_s32(dst_k.add(16 + row0), zero_v);
                        vst1q_s32(dst_k.add(32 + row0), zero_v);
                        vst1q_s32(dst_k.add(48 + row0), zero_v);
                        continue;
                    }
                    let v0 = vld1q_s32(a_i32.add((r_base + row0) * k_steps + ks));
                    let v1 = if row0 + 1 < valid_r {
                        vld1q_s32(a_i32.add((r_base + row0 + 1) * k_steps + ks))
                    } else {
                        zero_v
                    };
                    let v2 = if row0 + 2 < valid_r {
                        vld1q_s32(a_i32.add((r_base + row0 + 2) * k_steps + ks))
                    } else {
                        zero_v
                    };
                    let v3 = if row0 + 3 < valid_r {
                        vld1q_s32(a_i32.add((r_base + row0 + 3) * k_steps + ks))
                    } else {
                        zero_v
                    };

                    let t0 = vtrn1q_s32(v0, v1);
                    let t1 = vtrn2q_s32(v0, v1);
                    let t2 = vtrn1q_s32(v2, v3);
                    let t3 = vtrn2q_s32(v2, v3);

                    vst1q_s32(
                        dst_k.add(row0),
                        vcombine_s32(vget_low_s32(t0), vget_low_s32(t2)),
                    );
                    vst1q_s32(
                        dst_k.add(16 + row0),
                        vcombine_s32(vget_low_s32(t1), vget_low_s32(t3)),
                    );
                    vst1q_s32(
                        dst_k.add(32 + row0),
                        vcombine_s32(vget_high_s32(t0), vget_high_s32(t2)),
                    );
                    vst1q_s32(
                        dst_k.add(48 + row0),
                        vcombine_s32(vget_high_s32(t1), vget_high_s32(t3)),
                    );
                }
                ks += 4;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vst1q_f32_as_s32(ptr: *mut i32, v: int32x4_t) {
    vst1q_s32(ptr, v);
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe extern "C" fn gemm_sme_kernel(
    m_blocks: usize,
    m: usize,
    n_out: usize,
    k_steps: usize,
    n_super: usize,
    a_packed: *const i8,
    b_base: *const i8,
    ws_base: *const f32,
    bias_base: *const f32,
    a_scale: *const f32,
    res_ptr: *const f32,
    out_base: *mut f32,
) {
    for rb in 0..m_blocks {
        let r_base = rb * 16;
        let valid_r = (m - r_base).min(16);
        let pa_block = a_packed.add(rb * k_steps * 64);
        let pas_base = a_scale.add(r_base);
        let pout_rb = out_base.add(r_base * n_out);
        let pres_rb = if res_ptr.is_null() {
            std::ptr::null()
        } else {
            res_ptr.add(r_base * n_out)
        };

        core::arch::asm!(
            ".arch_extension sme",
            "smstart",
            "ptrue p0.b",
            "ptrue p1.s",
            "5:",
            "zero {{za}}",
            "mov {ks_tmp}, {k_steps}",
            "mov {pa_tmp}, {pa_block}",
            "2:",
            "ld1b {{z0.b}}, p0/z, [{pa_tmp}]",
            "ld1b {{z1.b}}, p0/z, [{pb_cur}]",
            "ld1b {{z2.b}}, p0/z, [{pb_cur}, #1, mul vl]",
            "ld1b {{z3.b}}, p0/z, [{pb_cur}, #2, mul vl]",
            "ld1b {{z4.b}}, p0/z, [{pb_cur}, #3, mul vl]",
            "add {pa_tmp}, {pa_tmp}, #64",
            "add {pb_cur}, {pb_cur}, #256",
            "smopa za0.s, p0/m, p0/m, z0.b, z1.b",
            "smopa za1.s, p0/m, p0/m, z0.b, z2.b",
            "smopa za2.s, p0/m, p0/m, z0.b, z3.b",
            "smopa za3.s, p0/m, p0/m, z0.b, z4.b",
            "subs {ks_tmp}, {ks_tmp}, #1",
            "b.ne 2b",

            "ld1w {{z16.s}}, p1/z, [{pws_cur}]",
            "ld1w {{z17.s}}, p1/z, [{pws_cur}, #1, mul vl]",
            "ld1w {{z18.s}}, p1/z, [{pws_cur}, #2, mul vl]",
            "ld1w {{z19.s}}, p1/z, [{pws_cur}, #3, mul vl]",
            "ld1w {{z20.s}}, p1/z, [{pbias_cur}]",
            "ld1w {{z21.s}}, p1/z, [{pbias_cur}, #1, mul vl]",
            "ld1w {{z22.s}}, p1/z, [{pbias_cur}, #2, mul vl]",
            "ld1w {{z23.s}}, p1/z, [{pbias_cur}, #3, mul vl]",
            "add {pws_cur}, {pws_cur}, #256",
            "add {pbias_cur}, {pbias_cur}, #256",

            "mov {pas_tmp}, {pas_base}",
            "mov {pout_tmp}, {pout_col}",
            "mov {pres_tmp}, {pres_col}",
            "mov w12, #0",
            "3:",
            "mova z0.s, p1/m, za0h.s[w12, 0]",
            "mova z1.s, p1/m, za1h.s[w12, 0]",
            "mova z2.s, p1/m, za2h.s[w12, 0]",
            "mova z3.s, p1/m, za3h.s[w12, 0]",
            "scvtf z0.s, p1/m, z0.s",
            "scvtf z1.s, p1/m, z1.s",
            "scvtf z2.s, p1/m, z2.s",
            "scvtf z3.s, p1/m, z3.s",
            "ld1rw {{z4.s}}, p1/z, [{pas_tmp}]",
            "add {pas_tmp}, {pas_tmp}, #4",
            "fmul z5.s, z16.s, z4.s",
            "fmul z6.s, z17.s, z4.s",
            "fmul z7.s, z18.s, z4.s",
            "fmul z8.s, z19.s, z4.s",
            "mov z24.d, z20.d",
            "mov z25.d, z21.d",
            "mov z26.d, z22.d",
            "mov z27.d, z23.d",
            "cbz {pres_tmp}, 4f",
            "ld1w {{z28.s}}, p1/z, [{pres_tmp}]",
            "ld1w {{z29.s}}, p1/z, [{pres_tmp}, #1, mul vl]",
            "ld1w {{z30.s}}, p1/z, [{pres_tmp}, #2, mul vl]",
            "ld1w {{z31.s}}, p1/z, [{pres_tmp}, #3, mul vl]",
            "fadd z24.s, p1/m, z24.s, z28.s",
            "fadd z25.s, p1/m, z25.s, z29.s",
            "fadd z26.s, p1/m, z26.s, z30.s",
            "fadd z27.s, p1/m, z27.s, z31.s",
            "add {pres_tmp}, {pres_tmp}, {n_bytes}",
            "4:",
            "fmla z24.s, p1/m, z0.s, z5.s",
            "fmla z25.s, p1/m, z1.s, z6.s",
            "fmla z26.s, p1/m, z2.s, z7.s",
            "fmla z27.s, p1/m, z3.s, z8.s",
            "st1w {{z24.s}}, p1, [{pout_tmp}]",
            "st1w {{z25.s}}, p1, [{pout_tmp}, #1, mul vl]",
            "st1w {{z26.s}}, p1, [{pout_tmp}, #2, mul vl]",
            "st1w {{z27.s}}, p1, [{pout_tmp}, #3, mul vl]",
            "add {pout_tmp}, {pout_tmp}, {n_bytes}",
            "add w12, w12, #1",
            "cmp w12, {valid_r:w}",
            "b.lo 3b",

            "add {pout_col}, {pout_col}, #256",
            "cbz {pres_col}, 6f",
            "add {pres_col}, {pres_col}, #256",
            "6:",
            "subs {nb_cnt}, {nb_cnt}, #1",
            "b.ne 5b",
            "smstop",
            pa_block = inout(reg) pa_block => _,
            k_steps = inout(reg) k_steps => _,
            pas_base = inout(reg) pas_base => _,
            n_bytes = inout(reg) n_out * 4 => _,
            valid_r = inout(reg) valid_r => _,
            nb_cnt = inout(reg) n_super => _,
            pb_cur = inout(reg) b_base => _,
            pws_cur = inout(reg) ws_base => _,
            pbias_cur = inout(reg) bias_base => _,
            pout_col = inout(reg) pout_rb => _,
            pres_col = inout(reg) pres_rb => _,
            ks_tmp = lateout(reg) _,
            pa_tmp = lateout(reg) _,
            pas_tmp = lateout(reg) _,
            pout_tmp = lateout(reg) _,
            pres_tmp = lateout(reg) _,
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

/// Packs `m x k_in` row-major Int8 activations into the SME `smopa` A-operand layout
/// (`[m.div_ceil(16), k_in / 4, 16 rows, 4 k]`). `out` needs `m.div_ceil(16) * 16 * k_in` bytes.
#[cfg(target_arch = "aarch64")]
pub fn sme_pack_activations(m: usize, k_in: usize, a: &[i8], out: &mut [i8]) {
    assert!(a.len() >= m * k_in && out.len() >= m.div_ceil(16) * 16 * k_in);
    unsafe { pack_activations_sme_neon(m, k_in, a, out) }
}

/// Runs the SME GEMM epilogue-fused kernel for the 64-column super-blocks `[sb_start, sb_start + sb_count)`
/// of `weights` on pre-packed activations (`sme_pack_activations`), writing those columns of the
/// row-major `[m, n_out]` output through a raw pointer, so that several threads can each fill a
/// disjoint column range of one shared output buffer (intra-op parallelism across SME units).
///
/// # Safety
/// `out` (and `residual`, if non-null) must be valid for `m * n_out` f32s; no other thread may access
/// the written column range concurrently. `weights.backend` must be `SimdBackend::ArmSme`.
#[cfg(target_arch = "aarch64")]
pub unsafe fn sme_gemm_packed_superblocks(
    m: usize,
    a_packed: &[i8],
    a_scale: &[f32],
    weights: &PackedLinearInt8,
    residual: *const f32,
    out: *mut f32,
    sb_start: usize,
    sb_count: usize,
) {
    debug_assert_eq!(weights.backend, SimdBackend::ArmSme);
    let n_out = weights.n_out;
    let k_steps = weights.k_in / 4;
    if m == 0 || sb_count == 0 {
        return;
    }
    debug_assert!((sb_start + sb_count) * 64 <= n_out);
    let col0 = sb_start * 64;
    gemm_sme_kernel(
        m.div_ceil(16),
        m,
        n_out,
        k_steps,
        sb_count,
        a_packed.as_ptr(),
        weights.packed_w.as_ptr().add(sb_start * k_steps * 256),
        weights.w_scale.as_ptr().add(col0),
        weights.bias.as_ptr().add(col0),
        a_scale.as_ptr(),
        if residual.is_null() {
            residual
        } else {
            residual.add(col0)
        },
        out.add(col0),
    );
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn gemm_sme_fused(
    m: usize,
    a_q: &[i8],
    a_packed: &mut [i8],
    a_scale: &[f32],
    weights: &PackedLinearInt8,
    residual: Option<&[f32]>,
    out: &mut [f32],
) {
    let n_out = weights.n_out;
    let k_in = weights.k_in;
    let k_steps = k_in / 4;
    let n_super = n_out / 64;
    let m_blocks = m.div_ceil(16);

    pack_activations_sme_neon(m, k_in, a_q, a_packed);

    let res_ptr = residual.map(|r| r.as_ptr()).unwrap_or(std::ptr::null());
    gemm_sme_kernel(
        m_blocks,
        m,
        n_out,
        k_steps,
        n_super,
        a_packed.as_ptr(),
        weights.packed_w.as_ptr(),
        weights.w_scale.as_ptr(),
        weights.bias.as_ptr(),
        a_scale.as_ptr(),
        res_ptr,
        out.as_mut_ptr(),
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
#[inline]
unsafe fn smmla(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    core::arch::asm!(
        "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
        acc = inout(vreg) acc,
        a = in(vreg) a,
        b = in(vreg) b,
        options(pure, nomem, nostack)
    );
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn smmla_8x8_kloop(
    pa: *const i8,
    pb: *const i8,
    k_steps: usize,
    c00: &mut int32x4_t,
    c01: &mut int32x4_t,
    c02: &mut int32x4_t,
    c03: &mut int32x4_t,
    c10: &mut int32x4_t,
    c11: &mut int32x4_t,
    c12: &mut int32x4_t,
    c13: &mut int32x4_t,
    c20: &mut int32x4_t,
    c21: &mut int32x4_t,
    c22: &mut int32x4_t,
    c23: &mut int32x4_t,
    c30: &mut int32x4_t,
    c31: &mut int32x4_t,
    c32: &mut int32x4_t,
    c33: &mut int32x4_t,
) {
    let mut a0: int8x16_t;
    let mut a1: int8x16_t;
    let mut a2: int8x16_t;
    let mut a3: int8x16_t;
    let mut b0: int8x16_t;
    let mut b1: int8x16_t;
    let mut b2: int8x16_t;
    let mut b3: int8x16_t;

    core::arch::asm!(
        "2:",
        "ldp {a0:q}, {a1:q}, [{pa}], #32",
        "ldp {b0:q}, {b1:q}, [{pb}], #32",
        "ldp {a2:q}, {a3:q}, [{pa}], #32",
        "smmla {c00:v}.4s, {a0:v}.16b, {b0:v}.16b",
        "smmla {c01:v}.4s, {a0:v}.16b, {b1:v}.16b",
        "ldp {b2:q}, {b3:q}, [{pb}], #32",
        "smmla {c10:v}.4s, {a1:v}.16b, {b0:v}.16b",
        "smmla {c11:v}.4s, {a1:v}.16b, {b1:v}.16b",
        "smmla {c02:v}.4s, {a0:v}.16b, {b2:v}.16b",
        "smmla {c03:v}.4s, {a0:v}.16b, {b3:v}.16b",
        "smmla {c12:v}.4s, {a1:v}.16b, {b2:v}.16b",
        "smmla {c13:v}.4s, {a1:v}.16b, {b3:v}.16b",
        "smmla {c20:v}.4s, {a2:v}.16b, {b0:v}.16b",
        "smmla {c21:v}.4s, {a2:v}.16b, {b1:v}.16b",
        "smmla {c22:v}.4s, {a2:v}.16b, {b2:v}.16b",
        "smmla {c23:v}.4s, {a2:v}.16b, {b3:v}.16b",
        "smmla {c30:v}.4s, {a3:v}.16b, {b0:v}.16b",
        "smmla {c31:v}.4s, {a3:v}.16b, {b1:v}.16b",
        "smmla {c32:v}.4s, {a3:v}.16b, {b2:v}.16b",
        "smmla {c33:v}.4s, {a3:v}.16b, {b3:v}.16b",
        "subs {ks}, {ks}, #1",
        "b.ne 2b",
        pa = inout(reg) pa => _,
        pb = inout(reg) pb => _,
        ks = inout(reg) k_steps => _,
        a0 = out(vreg) a0,
        a1 = out(vreg) a1,
        a2 = out(vreg) a2,
        a3 = out(vreg) a3,
        b0 = out(vreg) b0,
        b1 = out(vreg) b1,
        b2 = out(vreg) b2,
        b3 = out(vreg) b3,
        c00 = inout(vreg) *c00,
        c01 = inout(vreg) *c01,
        c02 = inout(vreg) *c02,
        c03 = inout(vreg) *c03,
        c10 = inout(vreg) *c10,
        c11 = inout(vreg) *c11,
        c12 = inout(vreg) *c12,
        c13 = inout(vreg) *c13,
        c20 = inout(vreg) *c20,
        c21 = inout(vreg) *c21,
        c22 = inout(vreg) *c22,
        c23 = inout(vreg) *c23,
        c30 = inout(vreg) *c30,
        c31 = inout(vreg) *c31,
        c32 = inout(vreg) *c32,
        c33 = inout(vreg) *c33,
        options(nostack, readonly)
    );
    let _ = (a0, a1, a2, a3, b0, b1, b2, b3);
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,i8mm")]
unsafe fn gemm_i8mm_fused(
    m: usize,
    a_q: &[i8],
    a_packed: &mut [i8],
    a_scale: &[f32],
    weights: &PackedLinearInt8,
    residual: Option<&[f32]>,
    out: &mut [f32],
) {
    let n_out = weights.n_out;
    let k_in = weights.k_in;
    let k_steps = k_in / 8;
    let n_panels = n_out / 8;

    pack_activations_i8mm_neon(m, k_in, a_q, a_packed);

    let b_base = weights.packed_w.as_ptr();
    let ws_base = weights.w_scale.as_ptr();
    let bias_base = weights.bias.as_ptr();
    let out_base = out.as_mut_ptr();
    let res_base = residual.map(|r| r.as_ptr());

    // Fast path for M == 1 (e.g. Layer 11 CLS-only pruning! Processes 16 columns = 2 panels per step)
    if m == 1 {
        let s_row = vdupq_n_f32(a_scale[0]);
        let mut nb = 0;
        while nb + 2 <= n_panels {
            let col0 = nb * 8;
            let col1 = (nb + 1) * 8;
            let mut pa = a_packed.as_ptr();
            let mut pb0 = b_base.add(nb * k_steps * 64);
            let mut pb1 = b_base.add((nb + 1) * k_steps * 64);

            let mut c0 = vdupq_n_s32(0);
            let mut c1 = vdupq_n_s32(0);
            let mut c2 = vdupq_n_s32(0);
            let mut c3 = vdupq_n_s32(0);
            let mut c4 = vdupq_n_s32(0);
            let mut c5 = vdupq_n_s32(0);
            let mut c6 = vdupq_n_s32(0);
            let mut c7 = vdupq_n_s32(0);

            for _ in 0..k_steps {
                let va = vld1q_s8(pa);
                pa = pa.add(16);
                let b0 = vld1q_s8(pb0);
                let b1 = vld1q_s8(pb0.add(16));
                let b2 = vld1q_s8(pb0.add(32));
                let b3 = vld1q_s8(pb0.add(48));
                pb0 = pb0.add(64);
                let b4 = vld1q_s8(pb1);
                let b5 = vld1q_s8(pb1.add(16));
                let b6 = vld1q_s8(pb1.add(32));
                let b7 = vld1q_s8(pb1.add(48));
                pb1 = pb1.add(64);

                c0 = smmla(c0, va, b0);
                c1 = smmla(c1, va, b1);
                c2 = smmla(c2, va, b2);
                c3 = smmla(c3, va, b3);
                c4 = smmla(c4, va, b4);
                c5 = smmla(c5, va, b5);
                c6 = smmla(c6, va, b6);
                c7 = smmla(c7, va, b7);
            }

            let acc_0_3 = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c0), vget_low_s32(c1)));
            let acc_4_7 = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c2), vget_low_s32(c3)));
            let acc_8_11 = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c4), vget_low_s32(c5)));
            let acc_12_15 = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c6), vget_low_s32(c7)));

            let sc0 = vmulq_f32(s_row, vld1q_f32(ws_base.add(col0)));
            let sc1 = vmulq_f32(s_row, vld1q_f32(ws_base.add(col0 + 4)));
            let sc2 = vmulq_f32(s_row, vld1q_f32(ws_base.add(col1)));
            let sc3 = vmulq_f32(s_row, vld1q_f32(ws_base.add(col1 + 4)));

            let mut bias0 = vld1q_f32(bias_base.add(col0));
            let mut bias1 = vld1q_f32(bias_base.add(col0 + 4));
            let mut bias2 = vld1q_f32(bias_base.add(col1));
            let mut bias3 = vld1q_f32(bias_base.add(col1 + 4));

            if let Some(rp) = res_base {
                bias0 = vaddq_f32(bias0, vld1q_f32(rp.add(col0)));
                bias1 = vaddq_f32(bias1, vld1q_f32(rp.add(col0 + 4)));
                bias2 = vaddq_f32(bias2, vld1q_f32(rp.add(col1)));
                bias3 = vaddq_f32(bias3, vld1q_f32(rp.add(col1 + 4)));
            }

            vst1q_f32(out_base.add(col0), vfmaq_f32(bias0, acc_0_3, sc0));
            vst1q_f32(out_base.add(col0 + 4), vfmaq_f32(bias1, acc_4_7, sc1));
            vst1q_f32(out_base.add(col1), vfmaq_f32(bias2, acc_8_11, sc2));
            vst1q_f32(out_base.add(col1 + 4), vfmaq_f32(bias3, acc_12_15, sc3));

            nb += 2;
        }
        return;
    }

    // General M >= 2 path: Outer loop over `nb` (keeps each 3-12 KB panel of B hot in L1D cache!),
    // inner loop over 8-row tiles, 4-row tile, 2-row tile, and 1-row tail.
    for nb in 0..n_panels {
        let col = nb * 8;
        let pb_panel = b_base.add(nb * k_steps * 64);
        let ws_lo = vld1q_f32(ws_base.add(col));
        let ws_hi = vld1q_f32(ws_base.add(col + 4));
        let bias_lo = vld1q_f32(bias_base.add(col));
        let bias_hi = vld1q_f32(bias_base.add(col + 4));

        let mut r = 0usize;
        let mut a_tile_offset = 0usize;

        while r + 8 <= m {
            let pa = a_packed.as_ptr().add(a_tile_offset);
            let pb = pb_panel;

            let mut c00 = vdupq_n_s32(0);
            let mut c01 = vdupq_n_s32(0);
            let mut c02 = vdupq_n_s32(0);
            let mut c03 = vdupq_n_s32(0);
            let mut c10 = vdupq_n_s32(0);
            let mut c11 = vdupq_n_s32(0);
            let mut c12 = vdupq_n_s32(0);
            let mut c13 = vdupq_n_s32(0);
            let mut c20 = vdupq_n_s32(0);
            let mut c21 = vdupq_n_s32(0);
            let mut c22 = vdupq_n_s32(0);
            let mut c23 = vdupq_n_s32(0);
            let mut c30 = vdupq_n_s32(0);
            let mut c31 = vdupq_n_s32(0);
            let mut c32 = vdupq_n_s32(0);
            let mut c33 = vdupq_n_s32(0);

            smmla_8x8_kloop(
                pa, pb, k_steps, &mut c00, &mut c01, &mut c02, &mut c03, &mut c10, &mut c11,
                &mut c12, &mut c13, &mut c20, &mut c21, &mut c22, &mut c23, &mut c30, &mut c31,
                &mut c32, &mut c33,
            );

            write_pair_i8mm(
                r, col, n_out, c00, c01, c02, c03, a_scale, ws_lo, ws_hi, bias_lo, bias_hi,
                res_base, out_base,
            );
            write_pair_i8mm(
                r + 2,
                col,
                n_out,
                c10,
                c11,
                c12,
                c13,
                a_scale,
                ws_lo,
                ws_hi,
                bias_lo,
                bias_hi,
                res_base,
                out_base,
            );
            write_pair_i8mm(
                r + 4,
                col,
                n_out,
                c20,
                c21,
                c22,
                c23,
                a_scale,
                ws_lo,
                ws_hi,
                bias_lo,
                bias_hi,
                res_base,
                out_base,
            );
            write_pair_i8mm(
                r + 6,
                col,
                n_out,
                c30,
                c31,
                c32,
                c33,
                a_scale,
                ws_lo,
                ws_hi,
                bias_lo,
                bias_hi,
                res_base,
                out_base,
            );
            a_tile_offset += k_steps * 64;
            r += 8;
        }

        if r + 4 <= m {
            let mut pa = a_packed.as_ptr().add(a_tile_offset);
            let mut pb = pb_panel;

            let mut c00 = vdupq_n_s32(0);
            let mut c01 = vdupq_n_s32(0);
            let mut c02 = vdupq_n_s32(0);
            let mut c03 = vdupq_n_s32(0);
            let mut c10 = vdupq_n_s32(0);
            let mut c11 = vdupq_n_s32(0);
            let mut c12 = vdupq_n_s32(0);
            let mut c13 = vdupq_n_s32(0);

            for _ in 0..k_steps {
                let a0 = vld1q_s8(pa);
                let a1 = vld1q_s8(pa.add(16));
                pa = pa.add(32);

                let b0 = vld1q_s8(pb);
                let b1 = vld1q_s8(pb.add(16));
                let b2 = vld1q_s8(pb.add(32));
                let b3 = vld1q_s8(pb.add(48));
                pb = pb.add(64);

                c00 = smmla(c00, a0, b0);
                c01 = smmla(c01, a0, b1);
                c02 = smmla(c02, a0, b2);
                c03 = smmla(c03, a0, b3);

                c10 = smmla(c10, a1, b0);
                c11 = smmla(c11, a1, b1);
                c12 = smmla(c12, a1, b2);
                c13 = smmla(c13, a1, b3);
            }

            write_pair_i8mm(
                r, col, n_out, c00, c01, c02, c03, a_scale, ws_lo, ws_hi, bias_lo, bias_hi,
                res_base, out_base,
            );
            write_pair_i8mm(
                r + 2,
                col,
                n_out,
                c10,
                c11,
                c12,
                c13,
                a_scale,
                ws_lo,
                ws_hi,
                bias_lo,
                bias_hi,
                res_base,
                out_base,
            );
            a_tile_offset += k_steps * 32;
            r += 4;
        }

        if r + 2 <= m {
            let mut pa = a_packed.as_ptr().add(a_tile_offset);
            let mut pb = pb_panel;

            let mut c00 = vdupq_n_s32(0);
            let mut c01 = vdupq_n_s32(0);
            let mut c02 = vdupq_n_s32(0);
            let mut c03 = vdupq_n_s32(0);

            for _ in 0..k_steps {
                let a0 = vld1q_s8(pa);
                pa = pa.add(16);
                let b0 = vld1q_s8(pb);
                let b1 = vld1q_s8(pb.add(16));
                let b2 = vld1q_s8(pb.add(32));
                let b3 = vld1q_s8(pb.add(48));
                pb = pb.add(64);

                c00 = smmla(c00, a0, b0);
                c01 = smmla(c01, a0, b1);
                c02 = smmla(c02, a0, b2);
                c03 = smmla(c03, a0, b3);
            }

            write_pair_i8mm(
                r, col, n_out, c00, c01, c02, c03, a_scale, ws_lo, ws_hi, bias_lo, bias_hi,
                res_base, out_base,
            );
            a_tile_offset += k_steps * 16;
            r += 2;
        }

        if r < m {
            let mut pa = a_packed.as_ptr().add(a_tile_offset);
            let mut pb = pb_panel;
            let s_r0 = vdupq_n_f32(a_scale[r]);
            let row_offset = r * n_out;

            let mut c00 = vdupq_n_s32(0);
            let mut c01 = vdupq_n_s32(0);
            let mut c02 = vdupq_n_s32(0);
            let mut c03 = vdupq_n_s32(0);

            for _ in 0..k_steps {
                let a0 = vld1q_s8(pa);
                pa = pa.add(16);
                let b0 = vld1q_s8(pb);
                let b1 = vld1q_s8(pb.add(16));
                let b2 = vld1q_s8(pb.add(32));
                let b3 = vld1q_s8(pb.add(48));
                pb = pb.add(64);

                c00 = smmla(c00, a0, b0);
                c01 = smmla(c01, a0, b1);
                c02 = smmla(c02, a0, b2);
                c03 = smmla(c03, a0, b3);
            }

            let ws_r_lo = vmulq_f32(s_r0, ws_lo);
            let ws_r_hi = vmulq_f32(s_r0, ws_hi);
            let mut b_lo = bias_lo;
            let mut b_hi = bias_hi;
            if let Some(rp) = res_base {
                b_lo = vaddq_f32(b_lo, vld1q_f32(rp.add(row_offset + col)));
                b_hi = vaddq_f32(b_hi, vld1q_f32(rp.add(row_offset + col + 4)));
            }
            let r0_lo = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c00), vget_low_s32(c01)));
            let r0_hi = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c02), vget_low_s32(c03)));
            vst1q_f32(
                out_base.add(row_offset + col),
                vfmaq_f32(b_lo, r0_lo, ws_r_lo),
            );
            vst1q_f32(
                out_base.add(row_offset + col + 4),
                vfmaq_f32(b_hi, r0_hi, ws_r_hi),
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn write_pair_i8mm(
    r: usize,
    col: usize,
    n_out: usize,
    c0: int32x4_t,
    c1: int32x4_t,
    c2: int32x4_t,
    c3: int32x4_t,
    a_scale: &[f32],
    ws_lo: float32x4_t,
    ws_hi: float32x4_t,
    bias_lo: float32x4_t,
    bias_hi: float32x4_t,
    res_base: Option<*const f32>,
    out_base: *mut f32,
) {
    let s0 = vdupq_n_f32(*a_scale.get_unchecked(r));
    let s1 = vdupq_n_f32(*a_scale.get_unchecked(r + 1));

    let r0_lo = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c0), vget_low_s32(c1)));
    let r0_hi = vcvtq_f32_s32(vcombine_s32(vget_low_s32(c2), vget_low_s32(c3)));
    let r1_lo = vcvtq_f32_s32(vcombine_s32(vget_high_s32(c0), vget_high_s32(c1)));
    let r1_hi = vcvtq_f32_s32(vcombine_s32(vget_high_s32(c2), vget_high_s32(c3)));

    let off0 = r * n_out + col;
    let off1 = (r + 1) * n_out + col;

    let (b0_lo, b0_hi, b1_lo, b1_hi) = if let Some(rp) = res_base {
        (
            vaddq_f32(bias_lo, vld1q_f32(rp.add(off0))),
            vaddq_f32(bias_hi, vld1q_f32(rp.add(off0 + 4))),
            vaddq_f32(bias_lo, vld1q_f32(rp.add(off1))),
            vaddq_f32(bias_hi, vld1q_f32(rp.add(off1 + 4))),
        )
    } else {
        (bias_lo, bias_hi, bias_lo, bias_hi)
    };

    vst1q_f32(
        out_base.add(off0),
        vfmaq_f32(b0_lo, r0_lo, vmulq_f32(s0, ws_lo)),
    );
    vst1q_f32(
        out_base.add(off0 + 4),
        vfmaq_f32(b0_hi, r0_hi, vmulq_f32(s0, ws_hi)),
    );
    vst1q_f32(
        out_base.add(off1),
        vfmaq_f32(b1_lo, r1_lo, vmulq_f32(s1, ws_lo)),
    );
    vst1q_f32(
        out_base.add(off1 + 4),
        vfmaq_f32(b1_hi, r1_hi, vmulq_f32(s1, ws_hi)),
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn sdot_vec(mut acc: int32x4_t, b: int8x16_t, a: int8x16_t) -> int32x4_t {
    core::arch::asm!(
        "sdot {acc:v}.4s, {b:v}.16b, {a:v}.16b",
        acc = inout(vreg) acc,
        b = in(vreg) b,
        a = in(vreg) a,
        options(pure, nomem, nostack)
    );
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon,dotprod")]
unsafe fn gemm_sdot_fused(
    m: usize,
    a_q: &[i8],
    a_scale: &[f32],
    weights: &PackedLinearInt8,
    residual: Option<&[f32]>,
    out: &mut [f32],
) {
    let n_out = weights.n_out;
    let k_in = weights.k_in;
    let k_steps = k_in / 4;
    let n_panels = n_out / 8;

    let a_base = a_q.as_ptr();
    let b_base = weights.packed_w.as_ptr();
    let ws_base = weights.w_scale.as_ptr();
    let bias_base = weights.bias.as_ptr();
    let out_base = out.as_mut_ptr();
    let res_base = residual.map(|r| r.as_ptr());

    let mut r = 0;
    while r + 4 <= m {
        let a_row0 = a_base.add(r * k_in) as *const i32;
        let a_row1 = a_base.add((r + 1) * k_in) as *const i32;
        let a_row2 = a_base.add((r + 2) * k_in) as *const i32;
        let a_row3 = a_base.add((r + 3) * k_in) as *const i32;
        let s0 = vdupq_n_f32(a_scale[r]);
        let s1 = vdupq_n_f32(a_scale[r + 1]);
        let s2 = vdupq_n_f32(a_scale[r + 2]);
        let s3 = vdupq_n_f32(a_scale[r + 3]);

        for nb in 0..n_panels {
            let col = nb * 8;
            let mut pb = b_base.add(nb * k_steps * 32);
            let mut c00 = vdupq_n_s32(0);
            let mut c01 = vdupq_n_s32(0);
            let mut c10 = vdupq_n_s32(0);
            let mut c11 = vdupq_n_s32(0);
            let mut c20 = vdupq_n_s32(0);
            let mut c21 = vdupq_n_s32(0);
            let mut c30 = vdupq_n_s32(0);
            let mut c31 = vdupq_n_s32(0);

            for ks in 0..k_steps {
                let b0 = vld1q_s8(pb);
                let b1 = vld1q_s8(pb.add(16));
                pb = pb.add(32);

                let va0 = vreinterpretq_s8_s32(vdupq_n_s32(*a_row0.add(ks)));
                let va1 = vreinterpretq_s8_s32(vdupq_n_s32(*a_row1.add(ks)));
                let va2 = vreinterpretq_s8_s32(vdupq_n_s32(*a_row2.add(ks)));
                let va3 = vreinterpretq_s8_s32(vdupq_n_s32(*a_row3.add(ks)));

                c00 = sdot_vec(c00, b0, va0);
                c01 = sdot_vec(c01, b1, va0);
                c10 = sdot_vec(c10, b0, va1);
                c11 = sdot_vec(c11, b1, va1);
                c20 = sdot_vec(c20, b0, va2);
                c21 = sdot_vec(c21, b1, va2);
                c30 = sdot_vec(c30, b0, va3);
                c31 = sdot_vec(c31, b1, va3);
            }

            let ws_lo = vld1q_f32(ws_base.add(col));
            let ws_hi = vld1q_f32(ws_base.add(col + 4));
            let bias_lo = vld1q_f32(bias_base.add(col));
            let bias_hi = vld1q_f32(bias_base.add(col + 4));

            for (dr, (c_lo, c_hi, sr)) in [
                (0, (c00, c01, s0)),
                (1, (c10, c11, s1)),
                (2, (c20, c21, s2)),
                (3, (c30, c31, s3)),
            ] {
                let off = (r + dr) * n_out + col;
                let mut b_lo = bias_lo;
                let mut b_hi = bias_hi;
                if let Some(rp) = res_base {
                    b_lo = vaddq_f32(b_lo, vld1q_f32(rp.add(off)));
                    b_hi = vaddq_f32(b_hi, vld1q_f32(rp.add(off + 4)));
                }
                vst1q_f32(
                    out_base.add(off),
                    vfmaq_f32(b_lo, vcvtq_f32_s32(c_lo), vmulq_f32(sr, ws_lo)),
                );
                vst1q_f32(
                    out_base.add(off + 4),
                    vfmaq_f32(b_hi, vcvtq_f32_s32(c_hi), vmulq_f32(sr, ws_hi)),
                );
            }
        }
        r += 4;
    }

    while r < m {
        let a_row = a_base.add(r * k_in) as *const i32;
        let s_row = vdupq_n_f32(a_scale[r]);
        let row_off = r * n_out;

        for nb in 0..n_panels {
            let col = nb * 8;
            let mut pb = b_base.add(nb * k_steps * 32);
            let mut c0 = vdupq_n_s32(0);
            let mut c1 = vdupq_n_s32(0);

            for ks in 0..k_steps {
                let va = vreinterpretq_s8_s32(vdupq_n_s32(*a_row.add(ks)));
                let b0 = vld1q_s8(pb);
                let b1 = vld1q_s8(pb.add(16));
                pb = pb.add(32);
                c0 = sdot_vec(c0, b0, va);
                c1 = sdot_vec(c1, b1, va);
            }

            let ws_lo = vmulq_f32(s_row, vld1q_f32(ws_base.add(col)));
            let ws_hi = vmulq_f32(s_row, vld1q_f32(ws_base.add(col + 4)));
            let mut b_lo = vld1q_f32(bias_base.add(col));
            let mut b_hi = vld1q_f32(bias_base.add(col + 4));
            if let Some(rp) = res_base {
                b_lo = vaddq_f32(b_lo, vld1q_f32(rp.add(row_off + col)));
                b_hi = vaddq_f32(b_hi, vld1q_f32(rp.add(row_off + col + 4)));
            }
            vst1q_f32(
                out_base.add(row_off + col),
                vfmaq_f32(b_lo, vcvtq_f32_s32(c0), ws_lo),
            );
            vst1q_f32(
                out_base.add(row_off + col + 4),
                vfmaq_f32(b_hi, vcvtq_f32_s32(c1), ws_hi),
            );
        }
        r += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gemm_neon_fused(
    m: usize,
    a_q: &[i8],
    a_scale: &[f32],
    weights: &PackedLinearInt8,
    residual: Option<&[f32]>,
    out: &mut [f32],
) {
    let n_out = weights.n_out;
    let k_in = weights.k_in;
    let a_base = a_q.as_ptr();
    let w_base = weights.packed_w.as_ptr();

    for r in 0..m {
        let a_row = a_base.add(r * k_in);
        let s_r = a_scale[r];
        let row_off = r * n_out;
        for n in 0..n_out {
            let w_row = w_base.add(n * k_in);
            let mut acc = vdupq_n_s32(0);
            let mut k = 0;
            while k < k_in {
                let va = vld1q_s8(a_row.add(k));
                let vw = vld1q_s8(w_row.add(k));
                let p_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vw));
                let p_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vw));
                let sum16 = vaddq_s16(p_lo, p_hi);
                acc = vpadalq_s16(acc, sum16);
                k += 16;
            }
            let dot = vaddvq_s32(acc);
            let mut b_val = weights.bias[n];
            if let Some(res) = residual {
                b_val += res[row_off + n];
            }
            out[row_off + n] = (dot as f32).mul_add(s_r * weights.w_scale[n], b_val);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_x86_avx2_fused(
    m: usize,
    a_q: &[i8],
    a_scale: &[f32],
    weights: &PackedLinearInt8,
    residual: Option<&[f32]>,
    out: &mut [f32],
) {
    let n_out = weights.n_out;
    let k_in = weights.k_in;
    let k_steps = k_in / 4;
    let n_panels = n_out / 16;
    let ones_16 = _mm256_set1_epi16(1);

    let a_base = a_q.as_ptr();
    let b_base = weights.packed_w.as_ptr();
    let ws_base = weights.w_scale.as_ptr();
    let bias_base = weights.bias.as_ptr();
    let out_base = out.as_mut_ptr();
    let res_base = residual.map(|r| r.as_ptr());

    for nb in 0..n_panels {
        let col = nb * 16;
        let pb_panel = b_base.add(nb * k_steps * 64);
        let ws0 = _mm256_loadu_ps(ws_base.add(col));
        let ws1 = _mm256_loadu_ps(ws_base.add(col + 8));
        let bias0 = _mm256_loadu_ps(bias_base.add(col));
        let bias1 = _mm256_loadu_ps(bias_base.add(col + 8));

        let mut r = 0;
        while r + 4 <= m {
            let a_row0 = a_base.add(r * k_in) as *const i32;
            let a_row1 = a_base.add((r + 1) * k_in) as *const i32;
            let a_row2 = a_base.add((r + 2) * k_in) as *const i32;
            let a_row3 = a_base.add((r + 3) * k_in) as *const i32;
            let mut pb = pb_panel;

            let mut c00 = _mm256_setzero_si256();
            let mut c01 = _mm256_setzero_si256();
            let mut c10 = _mm256_setzero_si256();
            let mut c11 = _mm256_setzero_si256();
            let mut c20 = _mm256_setzero_si256();
            let mut c21 = _mm256_setzero_si256();
            let mut c30 = _mm256_setzero_si256();
            let mut c31 = _mm256_setzero_si256();

            for ks in 0..k_steps {
                let b0 = _mm256_loadu_si256(pb as *const __m256i);
                let b1 = _mm256_loadu_si256(pb.add(32) as *const __m256i);
                pb = pb.add(64);

                let va0 = _mm256_set1_epi32(*a_row0.add(ks));
                let va0_abs = _mm256_abs_epi8(va0);
                c00 = _mm256_add_epi32(
                    c00,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(va0_abs, _mm256_sign_epi8(b0, va0)),
                        ones_16,
                    ),
                );
                c01 = _mm256_add_epi32(
                    c01,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(va0_abs, _mm256_sign_epi8(b1, va0)),
                        ones_16,
                    ),
                );

                let va1 = _mm256_set1_epi32(*a_row1.add(ks));
                let va1_abs = _mm256_abs_epi8(va1);
                c10 = _mm256_add_epi32(
                    c10,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(va1_abs, _mm256_sign_epi8(b0, va1)),
                        ones_16,
                    ),
                );
                c11 = _mm256_add_epi32(
                    c11,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(va1_abs, _mm256_sign_epi8(b1, va1)),
                        ones_16,
                    ),
                );

                let va2 = _mm256_set1_epi32(*a_row2.add(ks));
                let va2_abs = _mm256_abs_epi8(va2);
                c20 = _mm256_add_epi32(
                    c20,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(va2_abs, _mm256_sign_epi8(b0, va2)),
                        ones_16,
                    ),
                );
                c21 = _mm256_add_epi32(
                    c21,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(va2_abs, _mm256_sign_epi8(b1, va2)),
                        ones_16,
                    ),
                );

                let va3 = _mm256_set1_epi32(*a_row3.add(ks));
                let va3_abs = _mm256_abs_epi8(va3);
                c30 = _mm256_add_epi32(
                    c30,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(va3_abs, _mm256_sign_epi8(b0, va3)),
                        ones_16,
                    ),
                );
                c31 = _mm256_add_epi32(
                    c31,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(va3_abs, _mm256_sign_epi8(b1, va3)),
                        ones_16,
                    ),
                );
            }

            for (dr, (c_lo, c_hi)) in [
                (0, (c00, c01)),
                (1, (c10, c11)),
                (2, (c20, c21)),
                (3, (c30, c31)),
            ] {
                let s_row = _mm256_set1_ps(a_scale[r + dr]);
                let row_off = (r + dr) * n_out + col;
                let mut b0_f = bias0;
                let mut b1_f = bias1;
                if let Some(rp) = res_base {
                    b0_f = _mm256_add_ps(b0_f, _mm256_loadu_ps(rp.add(row_off)));
                    b1_f = _mm256_add_ps(b1_f, _mm256_loadu_ps(rp.add(row_off + 8)));
                }
                _mm256_storeu_ps(
                    out_base.add(row_off),
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(c_lo), _mm256_mul_ps(s_row, ws0), b0_f),
                );
                _mm256_storeu_ps(
                    out_base.add(row_off + 8),
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(c_hi), _mm256_mul_ps(s_row, ws1), b1_f),
                );
            }
            r += 4;
        }

        while r < m {
            let a_row = a_base.add(r * k_in) as *const i32;
            let s_row = _mm256_set1_ps(a_scale[r]);
            let row_off = r * n_out + col;
            let mut pb = pb_panel;
            let mut c0 = _mm256_setzero_si256();
            let mut c1 = _mm256_setzero_si256();

            for ks in 0..k_steps {
                let va = _mm256_set1_epi32(*a_row.add(ks));
                let va_abs = _mm256_abs_epi8(va);
                let b0 = _mm256_loadu_si256(pb as *const __m256i);
                let b1 = _mm256_loadu_si256(pb.add(32) as *const __m256i);
                pb = pb.add(64);

                let p0 = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(va_abs, _mm256_sign_epi8(b0, va)),
                    ones_16,
                );
                let p1 = _mm256_madd_epi16(
                    _mm256_maddubs_epi16(va_abs, _mm256_sign_epi8(b1, va)),
                    ones_16,
                );
                c0 = _mm256_add_epi32(c0, p0);
                c1 = _mm256_add_epi32(c1, p1);
            }

            let mut b0_f = bias0;
            let mut b1_f = bias1;
            if let Some(rp) = res_base {
                b0_f = _mm256_add_ps(b0_f, _mm256_loadu_ps(rp.add(row_off)));
                b1_f = _mm256_add_ps(b1_f, _mm256_loadu_ps(rp.add(row_off + 8)));
            }
            _mm256_storeu_ps(
                out_base.add(row_off),
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), _mm256_mul_ps(s_row, ws0), b0_f),
            );
            _mm256_storeu_ps(
                out_base.add(row_off + 8),
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), _mm256_mul_ps(s_row, ws1), b1_f),
            );
            r += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn gemm_x86_avx_vnni_fused(
    m: usize,
    a_q: &[i8],
    a_u8_scratch: &mut [i8],
    a_scale: &[f32],
    weights: &PackedLinearInt8,
    residual: Option<&[f32]>,
    out: &mut [f32],
) {
    let n_out = weights.n_out;
    let k_in = weights.k_in;
    let k_steps = k_in / 4;
    let n_panels = n_out / 16;

    // Shift A by +128 (`^ 0x80`) into unsigned u8
    let total_a = m * k_in;
    let bias_128 = _mm256_set1_epi8(-128i8);
    let mut idx = 0;
    while idx + 32 <= total_a {
        let v = _mm256_loadu_si256(a_q.as_ptr().add(idx) as *const __m256i);
        _mm256_storeu_si256(
            a_u8_scratch.as_mut_ptr().add(idx) as *mut __m256i,
            _mm256_xor_si256(v, bias_128),
        );
        idx += 32;
    }
    while idx < total_a {
        a_u8_scratch[idx] = (a_q[idx] as u8 ^ 0x80) as i8;
        idx += 1;
    }

    let a_base = a_u8_scratch.as_ptr();
    let b_base = weights.packed_w.as_ptr();
    let comp_base = weights.col_sum_comp.as_ptr();
    let ws_base = weights.w_scale.as_ptr();
    let bias_base = weights.bias.as_ptr();
    let out_base = out.as_mut_ptr();
    let res_base = residual.map(|r| r.as_ptr());

    for r in 0..m {
        let a_row = a_base.add(r * k_in) as *const i32;
        let s_row = _mm256_set1_ps(a_scale[r]);
        let row_off = r * n_out;

        for nb in 0..n_panels {
            let col = nb * 16;
            let mut pb = b_base.add(nb * k_steps * 64);
            let mut c0 = _mm256_loadu_si256(comp_base.add(col) as *const __m256i);
            let mut c1 = _mm256_loadu_si256(comp_base.add(col + 8) as *const __m256i);

            for ks in 0..k_steps {
                let va = _mm256_set1_epi32(*a_row.add(ks));
                let b0 = _mm256_loadu_si256(pb as *const __m256i);
                let b1 = _mm256_loadu_si256(pb.add(32) as *const __m256i);
                pb = pb.add(64);

                core::arch::asm!(
                    "{{vex}} vpdpbusd {c0:y}, {va:y}, {b0:y}",
                    "{{vex}} vpdpbusd {c1:y}, {va:y}, {b1:y}",
                    c0 = inout(ymm_reg) c0,
                    c1 = inout(ymm_reg) c1,
                    va = in(ymm_reg) va,
                    b0 = in(ymm_reg) b0,
                    b1 = in(ymm_reg) b1,
                    options(pure, nomem, nostack)
                );
            }

            let ws0 = _mm256_mul_ps(s_row, _mm256_loadu_ps(ws_base.add(col)));
            let ws1 = _mm256_mul_ps(s_row, _mm256_loadu_ps(ws_base.add(col + 8)));
            let mut b0_f = _mm256_loadu_ps(bias_base.add(col));
            let mut b1_f = _mm256_loadu_ps(bias_base.add(col + 8));
            if let Some(rp) = res_base {
                b0_f = _mm256_add_ps(b0_f, _mm256_loadu_ps(rp.add(row_off + col)));
                b1_f = _mm256_add_ps(b1_f, _mm256_loadu_ps(rp.add(row_off + col + 8)));
            }
            _mm256_storeu_ps(
                out_base.add(row_off + col),
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(c0), ws0, b0_f),
            );
            _mm256_storeu_ps(
                out_base.add(row_off + col + 8),
                _mm256_fmadd_ps(_mm256_cvtepi32_ps(c1), ws1, b1_f),
            );
        }
    }
}

pub fn gemm_scalar_fused(
    m: usize,
    a_q: &[i8],
    a_scale: &[f32],
    weights: &PackedLinearInt8,
    residual: Option<&[f32]>,
    out: &mut [f32],
) {
    let n_out = weights.n_out;
    let k_in = weights.k_in;
    let w = &weights.packed_w;

    for r in 0..m {
        let a_row = &a_q[r * k_in..(r + 1) * k_in];
        let s_r = a_scale[r];
        let row_off = r * n_out;
        for n in 0..n_out {
            let w_row = &w[n * k_in..(n + 1) * k_in];
            let mut acc: i32 = 0;
            for k in 0..k_in {
                acc += (a_row[k] as i32) * (w_row[k] as i32);
            }
            let mut b_val = weights.bias[n];
            if let Some(res) = residual {
                b_val += res[row_off + n];
            }
            out[row_off + n] = (acc as f32).mul_add(s_r * weights.w_scale[n], b_val);
        }
    }
}
