#!/usr/bin/env bash
# Canvas MC optimized startup — Java 25+ GraalVM recommended
# Tuned for: ZGC, SIMD (Vector API), FMA, Panama FFM, low-latency GC

JAVA_BIN="${JAVA_HOME:-/usr/lib/jvm/java-25-openjdk}/bin/java"
JAR="canvas-server/build/libs/canvas-paperclip-*.jar"
MEM="${CANVAS_MEM:-4G}"

exec "$JAVA_BIN" \
    # ── GC: Generational ZGC for sub-1ms pauses ──────────────────────────
    -XX:+UseZGC \
    -XX:+ZGenerational \
    -XX:ZAllocationSpikeTolerance=5 \
    -XX:ZUncommitDelay=60 \
    -XX:+DisableExplicitGC \
    -Xms${MEM} -Xmx${MEM} \
    # ── GraalVM JIT extras ────────────────────────────────────────────────
    -XX:+UseStringDeduplication \
    -XX:+OptimizeStringConcat \
    -XX:+UseFMA \
    -XX:CompileThreshold=1500 \
    -XX:-RestrictContended \
    # ── SIMD / Vector API (incubator) ────────────────────────────────────
    --add-modules=jdk.incubator.vector \
    # ── Panama FFM ────────────────────────────────────────────────────────
    --enable-native-access=ALL-UNNAMED \
    # ── GC logging (optional, comment out for prod perf) ─────────────────
    # -Xlog:gc*:file=logs/gc.log:time,uptime:filecount=5,filesize=20m \
    -jar $JAR \
    --nogui \
    "$@"
