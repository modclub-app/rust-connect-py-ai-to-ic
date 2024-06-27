#!/bin/bash
set -ex

export RUSTFLAGS=$RUSTFLAGS' -C target-feature=+simd128'
cargo build --release --target=wasm32-wasi
wasi2ic ./target/wasm32-wasi/release/wasi_ic_ai_sequence_backend.wasm ./target/wasm32-wasi/release/wasi_ic_ai_sequence_backend-ic.wasm
wasm-opt -Os -o ./target/wasm32-wasi/release/wasi_ic_ai_sequence_backend-ic.wasm \
        ./target/wasm32-wasi/release/wasi_ic_ai_sequence_backend-ic.wasm