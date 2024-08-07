#!/bin/bash
set -ex

export RUSTFLAGS=$RUSTFLAGS' -C target-feature=+simd128'
cargo build --release --target=wasm32-wasi
wasi2ic ./target/wasm32-wasi/release/gpt2_backend.wasm ./target/wasm32-wasi/release/gpt2_backend-ic.wasm
wasm-opt -Os -o ./target/wasm32-wasi/release/gpt2_backend-ic.wasm \
        ./target/wasm32-wasi/release/gpt2_backend-ic.wasm