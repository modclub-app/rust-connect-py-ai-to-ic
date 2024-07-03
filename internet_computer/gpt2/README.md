# gpt2

## Usage

cargo run --manifest-path ../../rust/upload_byte_file/Cargo.toml gpt2_backend upload_model_bytes_chunks ../../python/onnx_model/ [gpt2_with_kv_in.onnx] 0

Once the setup is complete, you can proceed with the following steps to build, deploy, and run your project.

1. Start the Internet Computer network locally in the background:
   ```bash
   dfx start --background
   ```
2. Deploy your project using `dfx`:
   ```bash
   dfx deploy
   ```
3. Use the Cargo command to run specific tasks, such as uploading model chunks:
   ```bash
   cargo run --manifest-path ../../rust/upload_byte_file/Cargo.toml gpt2_backend upload_model_chunks ../../python/onnx_model/ [gpt2_with_kv_in.onnx] 0
   ```
4. **Model Preparation**: Follow the commands to prepare the model for use:

   ```plaintext
    dfx canister call gpt2_backend upload_wasm_to_stable
    dfx canister call gpt2_backend setup_model
   ```
5. **Model Demo**: Example command

   ```plaintext
    dfx canister call gpt2_backend model_inference '(vec {1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13})'
   ```
