#!/bin/bash
# Canvas MC Server Start Script
# Java 26 + ZGC (Generational, default in Java 23+) + Canvas Vector API

JAVA=/usr/lib/jvm/java-26-temurin/bin/java

exec "$JAVA" \
  --add-modules jdk.incubator.vector \
  \
  -Xms8G -Xmx8G \
  \
  -XX:+UseZGC \
  -XX:+AlwaysPreTouch \
  -XX:+DisableExplicitGC \
  \
  -XX:SoftMaxHeapSize=6G \
  -XX:ZUncommitDelay=600 \
  \
  -XX:+UseStringDeduplication \
  \
  -XX:+PerfDisableSharedMem \
  -XX:+UnlockExperimentalVMOptions \
  \
  -Djava.net.preferIPv4Stack=true \
  -Dio.netty.allocator.type=pooled \
  -Dio.netty.allocator.maxOrder=9 \
  \
  -jar canvas.jar --nogui "$@"
