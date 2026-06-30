//! AABB batch intersection — architecture-specific hotpaths.
//!
//! Platform dispatch at compile time (zero runtime cost):
//!   x86_64 + avx512f  → 16 floats/cycle (zmm registers)
//!   x86_64 + avx2     → 8 floats/cycle  (ymm registers)
//!   x86_64 fallback   → 4 floats/cycle  (xmm SSE2, always available on x86_64)
//!   aarch64            → 4 floats/cycle  (NEON q-registers, mandatory on ARMv8)
//!   other              → scalar fallback

/// Intersection condition per axis: entityMin < queryMax && entityMax > queryMin.
/// All 3 axes must pass.  Output is 0 (no hit) or 1 (hit) per element.
pub fn intersect_batch(
    e_min_x: &[f32],
    e_min_y: &[f32],
    e_min_z: &[f32],
    e_max_x: &[f32],
    e_max_y: &[f32],
    e_max_z: &[f32],
    q_min_x: f32,
    q_min_y: f32,
    q_min_z: f32,
    q_max_x: f32,
    q_max_y: f32,
    q_max_z: f32,
    out: &mut [u8],
) {
    let n = out.len();
    debug_assert!(e_min_x.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: feature guard above
            return unsafe {
                intersect_avx512(
                    e_min_x, e_min_y, e_min_z, e_max_x, e_max_y, e_max_z,
                    q_min_x, q_min_y, q_min_z, q_max_x, q_max_y, q_max_z, out,
                )
            };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                intersect_avx2(
                    e_min_x, e_min_y, e_min_z, e_max_x, e_max_y, e_max_z,
                    q_min_x, q_min_y, q_min_z, q_max_x, q_max_y, q_max_z, out,
                )
            };
        }
        // SSE2 is guaranteed on all x86_64
        return unsafe {
            intersect_sse2(
                e_min_x, e_min_y, e_min_z, e_max_x, e_max_y, e_max_z,
                q_min_x, q_min_y, q_min_z, q_max_x, q_max_y, q_max_z, out,
            )
        };
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory for aarch64 Linux/Android/macOS — no runtime check needed.
        return unsafe {
            intersect_neon(
                e_min_x, e_min_y, e_min_z, e_max_x, e_max_y, e_max_z,
                q_min_x, q_min_y, q_min_z, q_max_x, q_max_y, q_max_z, out,
            )
        };
    }

    #[allow(unreachable_code)]
    intersect_scalar(
        e_min_x, e_min_y, e_min_z, e_max_x, e_max_y, e_max_z,
        q_min_x, q_min_y, q_min_z, q_max_x, q_max_y, q_max_z, out,
    );
}

// ── x86_64: AVX-512F — 16 entities per cycle ─────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn intersect_avx512(
    e_min_x: &[f32], e_min_y: &[f32], e_min_z: &[f32],
    e_max_x: &[f32], e_max_y: &[f32], e_max_z: &[f32],
    q_min_x: f32, q_min_y: f32, q_min_z: f32,
    q_max_x: f32, q_max_y: f32, q_max_z: f32,
    out: &mut [u8],
) {
    use std::arch::x86_64::*;
    let n = out.len();
    let lanes = 16usize;

    let vq_min_x = _mm512_set1_ps(q_min_x);
    let vq_min_y = _mm512_set1_ps(q_min_y);
    let vq_min_z = _mm512_set1_ps(q_min_z);
    let vq_max_x = _mm512_set1_ps(q_max_x);
    let vq_max_y = _mm512_set1_ps(q_max_y);
    let vq_max_z = _mm512_set1_ps(q_max_z);

    let mut i = 0usize;
    while i + lanes <= n {
        let ve_min_x = _mm512_loadu_ps(e_min_x.as_ptr().add(i));
        let ve_min_y = _mm512_loadu_ps(e_min_y.as_ptr().add(i));
        let ve_min_z = _mm512_loadu_ps(e_min_z.as_ptr().add(i));
        let ve_max_x = _mm512_loadu_ps(e_max_x.as_ptr().add(i));
        let ve_max_y = _mm512_loadu_ps(e_max_y.as_ptr().add(i));
        let ve_max_z = _mm512_loadu_ps(e_max_z.as_ptr().add(i));

        // AVX-512 comparison returns a bitmask (u16 for 16 floats)
        let mx = _mm512_cmp_ps_mask(ve_min_x, vq_max_x, _CMP_LT_OQ)
               & _mm512_cmp_ps_mask(ve_max_x, vq_min_x, _CMP_GT_OQ);
        let my = _mm512_cmp_ps_mask(ve_min_y, vq_max_y, _CMP_LT_OQ)
               & _mm512_cmp_ps_mask(ve_max_y, vq_min_y, _CMP_GT_OQ);
        let mz = _mm512_cmp_ps_mask(ve_min_z, vq_max_z, _CMP_LT_OQ)
               & _mm512_cmp_ps_mask(ve_max_z, vq_min_z, _CMP_GT_OQ);
        let bits = mx & my & mz;

        for j in 0..lanes {
            out[i + j] = ((bits >> j) & 1) as u8;
        }
        i += lanes;
    }
    // tail: fall through to AVX2 or scalar
    if i < n {
        if is_x86_feature_detected!("avx2") {
            intersect_avx2(
                &e_min_x[i..], &e_min_y[i..], &e_min_z[i..],
                &e_max_x[i..], &e_max_y[i..], &e_max_z[i..],
                q_min_x, q_min_y, q_min_z, q_max_x, q_max_y, q_max_z,
                &mut out[i..],
            );
        } else {
            intersect_scalar(
                &e_min_x[i..], &e_min_y[i..], &e_min_z[i..],
                &e_max_x[i..], &e_max_y[i..], &e_max_z[i..],
                q_min_x, q_min_y, q_min_z, q_max_x, q_max_y, q_max_z,
                &mut out[i..],
            );
        }
    }
}

// ── x86_64: AVX2 — 8 entities per cycle ──────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn intersect_avx2(
    e_min_x: &[f32], e_min_y: &[f32], e_min_z: &[f32],
    e_max_x: &[f32], e_max_y: &[f32], e_max_z: &[f32],
    q_min_x: f32, q_min_y: f32, q_min_z: f32,
    q_max_x: f32, q_max_y: f32, q_max_z: f32,
    out: &mut [u8],
) {
    use std::arch::x86_64::*;
    let n = out.len();
    let lanes = 8usize;

    let vq_min_x = _mm256_set1_ps(q_min_x);
    let vq_min_y = _mm256_set1_ps(q_min_y);
    let vq_min_z = _mm256_set1_ps(q_min_z);
    let vq_max_x = _mm256_set1_ps(q_max_x);
    let vq_max_y = _mm256_set1_ps(q_max_y);
    let vq_max_z = _mm256_set1_ps(q_max_z);

    let mut i = 0usize;
    while i + lanes <= n {
        let ve_min_x = _mm256_loadu_ps(e_min_x.as_ptr().add(i));
        let ve_min_y = _mm256_loadu_ps(e_min_y.as_ptr().add(i));
        let ve_min_z = _mm256_loadu_ps(e_min_z.as_ptr().add(i));
        let ve_max_x = _mm256_loadu_ps(e_max_x.as_ptr().add(i));
        let ve_max_y = _mm256_loadu_ps(e_max_y.as_ptr().add(i));
        let ve_max_z = _mm256_loadu_ps(e_max_z.as_ptr().add(i));

        // _CMP_LT_OQ = ordered, quiet less-than
        let mask_x = _mm256_and_ps(
            _mm256_cmp_ps(ve_min_x, vq_max_x, _CMP_LT_OQ),
            _mm256_cmp_ps(ve_max_x, vq_min_x, _CMP_GT_OQ),
        );
        let mask_y = _mm256_and_ps(
            _mm256_cmp_ps(ve_min_y, vq_max_y, _CMP_LT_OQ),
            _mm256_cmp_ps(ve_max_y, vq_min_y, _CMP_GT_OQ),
        );
        let mask_z = _mm256_and_ps(
            _mm256_cmp_ps(ve_min_z, vq_max_z, _CMP_LT_OQ),
            _mm256_cmp_ps(ve_max_z, vq_min_z, _CMP_GT_OQ),
        );
        let mask = _mm256_and_ps(mask_x, _mm256_and_ps(mask_y, mask_z));
        let bits = _mm256_movemask_ps(mask) as u8;

        for j in 0..lanes {
            out[i + j] = (bits >> j) & 1;
        }
        i += lanes;
    }
    // scalar tail
    intersect_scalar(
        &e_min_x[i..], &e_min_y[i..], &e_min_z[i..],
        &e_max_x[i..], &e_max_y[i..], &e_max_z[i..],
        q_min_x, q_min_y, q_min_z, q_max_x, q_max_y, q_max_z,
        &mut out[i..],
    );
}

// ── x86_64: SSE2 — 4 entities per cycle (guaranteed baseline) ────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn intersect_sse2(
    e_min_x: &[f32], e_min_y: &[f32], e_min_z: &[f32],
    e_max_x: &[f32], e_max_y: &[f32], e_max_z: &[f32],
    q_min_x: f32, q_min_y: f32, q_min_z: f32,
    q_max_x: f32, q_max_y: f32, q_max_z: f32,
    out: &mut [u8],
) {
    use std::arch::x86_64::*;
    let n = out.len();
    let lanes = 4usize;

    let vq_min_x = _mm_set1_ps(q_min_x);
    let vq_min_y = _mm_set1_ps(q_min_y);
    let vq_min_z = _mm_set1_ps(q_min_z);
    let vq_max_x = _mm_set1_ps(q_max_x);
    let vq_max_y = _mm_set1_ps(q_max_y);
    let vq_max_z = _mm_set1_ps(q_max_z);

    let mut i = 0usize;
    while i + lanes <= n {
        let ve_min_x = _mm_loadu_ps(e_min_x.as_ptr().add(i));
        let ve_min_y = _mm_loadu_ps(e_min_y.as_ptr().add(i));
        let ve_min_z = _mm_loadu_ps(e_min_z.as_ptr().add(i));
        let ve_max_x = _mm_loadu_ps(e_max_x.as_ptr().add(i));
        let ve_max_y = _mm_loadu_ps(e_max_y.as_ptr().add(i));
        let ve_max_z = _mm_loadu_ps(e_max_z.as_ptr().add(i));

        // SSE2 uses _mm_cmplt_ps / _mm_cmpgt_ps
        let mask_x = _mm_and_ps(_mm_cmplt_ps(ve_min_x, vq_max_x), _mm_cmpgt_ps(ve_max_x, vq_min_x));
        let mask_y = _mm_and_ps(_mm_cmplt_ps(ve_min_y, vq_max_y), _mm_cmpgt_ps(ve_max_y, vq_min_y));
        let mask_z = _mm_and_ps(_mm_cmplt_ps(ve_min_z, vq_max_z), _mm_cmpgt_ps(ve_max_z, vq_min_z));
        let mask = _mm_and_ps(mask_x, _mm_and_ps(mask_y, mask_z));
        let bits = _mm_movemask_ps(mask) as u8;

        for j in 0..lanes {
            out[i + j] = (bits >> j) & 1;
        }
        i += lanes;
    }
    intersect_scalar(
        &e_min_x[i..], &e_min_y[i..], &e_min_z[i..],
        &e_max_x[i..], &e_max_y[i..], &e_max_z[i..],
        q_min_x, q_min_y, q_min_z, q_max_x, q_max_y, q_max_z,
        &mut out[i..],
    );
}

// ── aarch64: NEON — 4 entities per cycle (Dimensity 7025, G99, M-series) ─────
//
// Cortex-A55 (efficiency core in Dimensity 7025, G99): 64-bit NEON, 4 f32/cycle
// Cortex-A76 (performance core in G99, Dimensity 7020): 128-bit micro-op NEON
// Cortex-A78 (performance core in Dimensity 7025): same as A76 but wider
// Apple M-series: ARMv8.5-a, 2x128-bit NEON, 8 f32/cycle via instruction fusion
//
// On A76/A78 with dotprod enabled, the compiler can use better scheduling.
// NEON is unconditionally available on all ARMv8+ — no runtime check needed.

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn intersect_neon(
    e_min_x: &[f32], e_min_y: &[f32], e_min_z: &[f32],
    e_max_x: &[f32], e_max_y: &[f32], e_max_z: &[f32],
    q_min_x: f32, q_min_y: f32, q_min_z: f32,
    q_max_x: f32, q_max_y: f32, q_max_z: f32,
    out: &mut [u8],
) {
    use std::arch::aarch64::*;
    let n = out.len();
    let lanes = 4usize;

    let vq_min_x = vdupq_n_f32(q_min_x);
    let vq_min_y = vdupq_n_f32(q_min_y);
    let vq_min_z = vdupq_n_f32(q_min_z);
    let vq_max_x = vdupq_n_f32(q_max_x);
    let vq_max_y = vdupq_n_f32(q_max_y);
    let vq_max_z = vdupq_n_f32(q_max_z);

    let mut i = 0usize;
    while i + lanes <= n {
        let ve_min_x = vld1q_f32(e_min_x.as_ptr().add(i));
        let ve_min_y = vld1q_f32(e_min_y.as_ptr().add(i));
        let ve_min_z = vld1q_f32(e_min_z.as_ptr().add(i));
        let ve_max_x = vld1q_f32(e_max_x.as_ptr().add(i));
        let ve_max_y = vld1q_f32(e_max_y.as_ptr().add(i));
        let ve_max_z = vld1q_f32(e_max_z.as_ptr().add(i));

        // vcltq_f32 / vcgtq_f32 already return uint32x4_t (0x00 or 0xFF per lane)
        let mask_x = vandq_u32(
            vcltq_f32(ve_min_x, vq_max_x),
            vcgtq_f32(ve_max_x, vq_min_x),
        );
        let mask_y = vandq_u32(
            vcltq_f32(ve_min_y, vq_max_y),
            vcgtq_f32(ve_max_y, vq_min_y),
        );
        let mask_z = vandq_u32(
            vcltq_f32(ve_min_z, vq_max_z),
            vcgtq_f32(ve_max_z, vq_min_z),
        );
        let mask = vandq_u32(mask_x, vandq_u32(mask_y, mask_z));

        // Extract top bit of each u32 lane → 0 or 1
        // vshrq_n_u32(mask, 31) gives 0 or 1 per lane as u32
        let bits = vshrq_n_u32(mask, 31);
        // Narrow u32x4 → u16x4 → u8x8 (lower 4 valid)
        let bits16 = vmovn_u32(bits);                             // u16x4
        let bits8  = vmovn_u16(vcombine_u16(bits16, vdup_n_u16(0))); // u8x8

        // Store 4 bytes
        let arr: [u8; 8] = std::mem::transmute(bits8);
        out[i]     = arr[0];
        out[i + 1] = arr[1];
        out[i + 2] = arr[2];
        out[i + 3] = arr[3];

        i += lanes;
    }
    intersect_scalar(
        &e_min_x[i..], &e_min_y[i..], &e_min_z[i..],
        &e_max_x[i..], &e_max_y[i..], &e_max_z[i..],
        q_min_x, q_min_y, q_min_z, q_max_x, q_max_y, q_max_z,
        &mut out[i..],
    );
}

// ── Scalar fallback (all architectures) ──────────────────────────────────────

#[inline]
fn intersect_scalar(
    e_min_x: &[f32], e_min_y: &[f32], e_min_z: &[f32],
    e_max_x: &[f32], e_max_y: &[f32], e_max_z: &[f32],
    q_min_x: f32, q_min_y: f32, q_min_z: f32,
    q_max_x: f32, q_max_y: f32, q_max_z: f32,
    out: &mut [u8],
) {
    for i in 0..out.len() {
        out[i] = (e_min_x[i] < q_max_x
            && e_max_x[i] > q_min_x
            && e_min_y[i] < q_max_y
            && e_max_y[i] > q_min_y
            && e_min_z[i] < q_max_z
            && e_max_z[i] > q_min_z) as u8;
    }
}
