# single_call

## Usage

Once the setup is complete, you can proceed with the following steps to build, deploy, and run your project.

1. **Setup Cargo**: Ensure you have Cargo installed and set up. If not, follow the instructions on the [Rust website](https://www.rust-lang.org/tools/install) to install Rust and Cargo.

2. Install the necessary Cargo package:
   ```bash
   cargo install ic-file-uploader
   ```

3. Start the Internet Computer network locally in the background:
   ```bash
   dfx start --background
   ```

4. Deploy your project using `dfx`:
   ```bash
   dfx deploy
   ```

5. Use the installed Cargo package to run specific tasks, such as uploading model chunks. Replace the demo model `gpt2_embedding.onnx` with your actual model file names:
   ```bash
   ic-file-uploader single_call_backend upload_model_bytes_chunks gpt2_embedding.onnx
   ```

6. Prepare the model for use with the following command:
   ```bash
   dfx canister call single_call_backend setup_model
   ```
   
## Demo Instructions

These instructions guide you through running a demonstration of our application, which illustrates using the Python Transformers library, executing a Python script for model partitioning, loading the model into a canister, and interacting with the backend Canister API.

### Prerequisites

- Python
- Cargo (for Rust projects)

### Step 1: Install Dependencies

- **Python Transformers Library**: The project uses the Transformers library for model management.
  ```bash
  pip install transformers
  ```

- **NodeJS Dependencies for the Frontend**:
  ```bash
  npm install --save-dev webpack webpack-cli
  sudo apt-get install wabt
  sudo apt-get install binaryen
  ```

### Step 2: Install WASI SDK 21

1. Download wasi-sdk-21.0 from [WASI SDK Releases](https://github.com/WebAssembly/wasi-sdk/releases/tag/wasi-sdk-21).
2. Export `CC_wasm32_wasi` in your shell such that it points to WASI clang and sysroot. Example:
   ```bash
   export CC_wasm32_wasi="/path/to/wasi-sdk-21.0/bin/clang --sysroot=/path/to/wasi-sdk-21.0/share/wasi-sysroot"
   ```

### Step 3: Install wasi2ic

1. Clone the repository:
   ```bash
   git clone https://github.com/wasm-forge/wasi2ic
   ```
2. Enter the `wasi2ic` folder.
3. Compile the project with:
   ```bash
   cargo build
   ```
   Alternatively, use:
   ```bash
   cargo install --path .
   ```
4. Ensure the `wasi2ic` binary is in your `$PATH`.

### Step 4: Partition the GPT-2 Model

- Run the script to partition the GPT-2 model, preparing it for backend use:
  ```bash
  python3 python/GPT2_max_partition_model_pool.py
  ```


### Step 5: Build and Deploy

1. Start the Internet Computer network locally in the background:
   ```bash
   dfx start --background
   ```
2. Deploy your project using `dfx`:
   ```bash
   dfx deploy
   ```

### Step 6: Load the Model into the Backend

1. **Model Upload**: Navigate to the canister scripts directory and perform the following:

   - For local deployment:
     ```bash
        ic-file-uploader single_call_backend upload_model_bytes_chunks gpt2_embedding.onnx
     ```

   - For Internet Computer mainnet deployment:
     ```bash
        ic-file-uploader single_call_backend upload_model_bytes_chunks gpt2_embedding.onnx --network ic
    ```

   - If an upload is interrupted, query the last successful upload with:
```plaintext
        dfx call canister single_call_backend upload_wasm_ref_cell_length
```
     And resume uploading using the result:
```bash
        ic-file-uploader single_call_backend upload_model_bytes_chunks gpt2_embedding.onnx --offset <result number>
```

2. **Model Storage**: This will store the model to stable memory so that it can be efficiently loaded after redeployment:
   ```plaintext
    dfx canister call single_call_backend upload_wasm_ref_cell_to_stable 
   ```

3. **Model Preparation**: Follow the commands to prepare the model for use:
   ```plaintext
    dfx canister call single_call_backend setup_model
   ```

### Step 7: Test the API

- Demonstrate the model's functionality with a call in command line such as `dfx canister call single_call_backend model_inference '(vec {1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13})'`

### Additional Setup for wasm-opt

- Install `wasm-opt`:
  ```bash
  cargo install wasm-opt
  ```

### Additional Notes on Using wasi2ic

- To convert a WASI-dependent Wasm module to run on the Internet Computer:
  ```bash
  wasi2ic <input-wasm-file> <output_wasm_file>
  ```
- Add the polyfill dependency to your project:
  ```bash
  cargo add --git https://github.com/wasm-forge/ic-wasi-polyfill
  ```
