#!/usr/bin/env bash
# Canvas Native — multi-target build script.
# Produces one .so/.dylib per platform/arch variant inside out/.
#
# Prerequisites:
#   rustup + cargo-zigbuild (for cross-compilation via zig):
#     cargo install cargo-zigbuild
#   zig:
#     pacman -S zig  /  brew install zig  /  apt install zig
#
# Usage:
#   ./build-targets.sh            # build all targets
#   ./build-targets.sh x86_64     # only x86_64 targets
#   ./build-targets.sh aarch64    # only aarch64 targets

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$SCRIPT_DIR/out"
mkdir -p "$OUT"

# ── helpers ───────────────────────────────────────────────────────────────────

build() {
    local target="$1"
    local suffix="$2"
    local out_dir="$3"
    shift 3
    local extra_flags="$*"

    echo "── building $target ($suffix) ──"
    mkdir -p "$OUT/$out_dir"

    RUSTFLAGS="-C opt-level=3 -C lto=thin -C codegen-units=1 -C panic=abort $extra_flags" \
        cargo zigbuild --release --target "$target" \
        --manifest-path "$SCRIPT_DIR/Cargo.toml"

    local lib_name
    case "$target" in
        *apple*) lib_name="libcanvas_native.dylib" ;;
        *)       lib_name="libcanvas_native.so"    ;;
    esac

    local src="$SCRIPT_DIR/target/$target/release/$lib_name"
    local dst_name
    case "$suffix" in
        "") dst_name="canvas-native${lib_name##libcanvas_native}" ;;
        *)  dst_name="canvas-native-${suffix}${lib_name##libcanvas_native}" ;;
    esac
    cp "$src" "$OUT/$out_dir/$dst_name"
    echo "   → out/$out_dir/$dst_name"
}

FILTER="${1:-all}"

# ── Linux x86_64 ─────────────────────────────────────────────────────────────
# x86-64-v2: SSE4.2 (Nehalem 2008+)   — broadest compatibility
# x86-64-v3: AVX2 + FMA (Haswell 2013+) — sweet spot for modern servers
# x86-64-v4: AVX-512 (Skylake-X, Zen4+) — datacenter / AMD Ryzen 7000+
# Each variant: glibc (default ubuntu/arch) and musl (Alpine, scratch containers)

if [[ "$FILTER" == "all" || "$FILTER" == "x86_64" ]]; then
    # glibc, AVX2 (primary: Ubuntu/Arch/Debian servers, Ryzen/Intel Haswell+)
    build "x86_64-unknown-linux-gnu" "" "linux_x86_64" \
        "-C target-cpu=x86-64-v3"

    # glibc, AVX-512 (Zen4+ / Skylake-X / Intel Xeon datacenter)
    build "x86_64-unknown-linux-gnu" "avx512" "linux_x86_64" \
        "-C target-cpu=x86-64-v4"

    # glibc, SSE4.2 baseline (Nehalem 2008+, broadest compatibility)
    build "x86_64-unknown-linux-gnu" "compat" "linux_x86_64" \
        "-C target-cpu=x86-64-v2"

    # Note: x86_64-unknown-linux-musl cdylib is not supported by Rust's musl target
    # (musl requires fully static executables; cdylib needs PIC dynamic linking).
    # Alpine/musl users: use the glibc variant with LD_PRELOAD=./libcanvas_native.so
    # or build from source with: RUSTFLAGS="-C target-feature=-crt-static" zig cc
fi

# ── Linux aarch64 ─────────────────────────────────────────────────────────────
# cortex-a76: perf cores in MediaTek G99, Dimensity 7020 (ARMv8.2-a)
# cortex-a78: perf cores in Dimensity 7025 (ARMv8.2-a, wider issue)
# cortex-a55: efficiency cores in both (ARMv8.2-a, in-order)
# apple-m1:   Apple Silicon (ARMv8.5-a, 2-wide NEON) — also covers M2/M3
#
# NEON is mandatory for all aarch64 targets → always enabled.
# dotprod, fp16: ARMv8.2-a features available on A76/A78 → better scheduling.
# sve2: available on some A78 revisions and Neoverse N2 → not enabled here
#       (would need runtime check in the caller).

if [[ "$FILTER" == "all" || "$FILTER" == "aarch64" ]]; then
    # baseline: A55 in-order (efficiency cores in Dimensity 7025, G99)
    # NEON mandatory on all aarch64 — no runtime gate needed
    build "aarch64-unknown-linux-gnu" "" "linux_aarch64" \
        "-C target-cpu=cortex-a55 -C target-feature=+neon"

    # Cortex-A76 (perf cores in MediaTek G99, Dimensity 7020)
    # fp16, dotprod, rcpc: ARMv8.2-a features for better scheduler hints
    build "aarch64-unknown-linux-gnu" "a76" "linux_aarch64" \
        "-C target-cpu=cortex-a76 -C target-feature=+neon,+fp16,+dotprod,+rcpc"

    # Cortex-A78 (perf cores in Dimensity 7025)
    build "aarch64-unknown-linux-gnu" "a78" "linux_aarch64" \
        "-C target-cpu=cortex-a78 -C target-feature=+neon,+fp16,+dotprod,+rcpc"

    # Note: aarch64-unknown-linux-musl cdylib not supported (same as x86_64-musl).
fi

# ── macOS aarch64 (Apple M-series) ───────────────────────────────────────────
if [[ "$FILTER" == "all" || "$FILTER" == "macos" ]]; then
    build "aarch64-apple-darwin" "" "darwin_aarch64" \
        "-C target-cpu=apple-m1 -C target-feature=+neon,+fp16,+dotprod,+fullfp16,+sha3"

    build "x86_64-apple-darwin" "" "darwin_x86_64" \
        "-C target-cpu=x86-64-v3"
fi

echo ""
echo "── done ──"
find "$OUT" -type f | sort | while read -r f; do
    echo "  $(du -sh "$f" | cut -f1)  $f"
done
