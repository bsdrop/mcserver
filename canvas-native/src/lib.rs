//! Canvas native SIMD — JNI entry points.

#![allow(non_snake_case, clippy::too_many_arguments)]

mod aabb;

use jni::objects::JClass;
use jni::sys::{jbooleanArray, jfloat, jfloatArray, jint};
use jni::JNIEnv;

/// Java signature (io.canvasmc.canvas.simd.NativeBatchAABB):
///   private static native void nativeIntersectBatch(
///       float[] minX, float[] minY, float[] minZ,
///       float[] maxX, float[] maxY, float[] maxZ,
///       int count,
///       float qMinX, float qMinY, float qMinZ,
///       float qMaxX, float qMaxY, float qMaxZ,
///       boolean[] out);
#[no_mangle]
pub unsafe extern "system" fn Java_io_canvasmc_canvas_simd_NativeBatchAABB_nativeIntersectBatch(
    mut env: JNIEnv,
    _class: JClass,
    j_min_x: jfloatArray,
    j_min_y: jfloatArray,
    j_min_z: jfloatArray,
    j_max_x: jfloatArray,
    j_max_y: jfloatArray,
    j_max_z: jfloatArray,
    count: jint,
    q_min_x: jfloat,
    q_min_y: jfloat,
    q_min_z: jfloat,
    q_max_x: jfloat,
    q_max_y: jfloat,
    q_max_z: jfloat,
    j_out: jbooleanArray,
) {
    let n = count as usize;
    if n == 0 {
        return;
    }

    // Use GetPrimitiveArrayCritical for zero-copy pinned access.
    // This suspends GC for the critical section — keep it short.
    let raw = env.get_raw();
    let iface = &**raw;

    let pin_f = iface.GetPrimitiveArrayCritical.unwrap();
    let unpin = iface.ReleasePrimitiveArrayCritical.unwrap();
    let null = std::ptr::null_mut();

    let p_min_x = pin_f(raw, j_min_x as _, null) as *const f32;
    let p_min_y = pin_f(raw, j_min_y as _, null) as *const f32;
    let p_min_z = pin_f(raw, j_min_z as _, null) as *const f32;
    let p_max_x = pin_f(raw, j_max_x as _, null) as *const f32;
    let p_max_y = pin_f(raw, j_max_y as _, null) as *const f32;
    let p_max_z = pin_f(raw, j_max_z as _, null) as *const f32;
    let p_out   = pin_f(raw, j_out   as _, null) as *mut u8;

    {
        use std::slice;
        let min_x = slice::from_raw_parts(p_min_x, n);
        let min_y = slice::from_raw_parts(p_min_y, n);
        let min_z = slice::from_raw_parts(p_min_z, n);
        let max_x = slice::from_raw_parts(p_max_x, n);
        let max_y = slice::from_raw_parts(p_max_y, n);
        let max_z = slice::from_raw_parts(p_max_z, n);
        let out   = slice::from_raw_parts_mut(p_out, n);

        aabb::intersect_batch(
            min_x, min_y, min_z, max_x, max_y, max_z,
            q_min_x, q_min_y, q_min_z, q_max_x, q_max_y, q_max_z,
            out,
        );
    }

    // Release in reverse pin order.  mode=0 → copy back + free.
    const COMMIT: jni::sys::jint = 0;
    unpin(raw, j_out   as _, p_out   as *mut _, COMMIT);
    unpin(raw, j_max_z as _, p_max_z as *mut _, jni::sys::JNI_ABORT);
    unpin(raw, j_max_y as _, p_max_y as *mut _, jni::sys::JNI_ABORT);
    unpin(raw, j_max_x as _, p_max_x as *mut _, jni::sys::JNI_ABORT);
    unpin(raw, j_min_z as _, p_min_z as *mut _, jni::sys::JNI_ABORT);
    unpin(raw, j_min_y as _, p_min_y as *mut _, jni::sys::JNI_ABORT);
    unpin(raw, j_min_x as _, p_min_x as *mut _, jni::sys::JNI_ABORT);
}
