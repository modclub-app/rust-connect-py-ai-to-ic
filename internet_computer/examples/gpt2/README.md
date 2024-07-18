# gpt2
 
## Performance Improvements with SIMD Integration and Key-Value Caching

### Overview
This project evaluates the performance improvements in token generation for a language model with the integration of SIMD (Single Instruction, Multiple Data) and Key-Value Caching.

### SIMD Integration
Experiments were conducted with single update calls. The table below shows that the SIMD integration, released in DFX version 0.20.2-beta.0, greatly increased the model's ability to process tokens.

#### Performance Before and After SIMD Integration
| Metric                               | Before SIMD Integration | After SIMD Integration |
|--------------------------------------|-------------------------|------------------------|
| Maximum input tokens for one output  | 4                       | 76                     |
| Maximum output tokens for four input | 1                       | 17                     |

The SIMD integration enabled the model to process 76 input tokens and produce a single output token, achieving a 19x increase in reading throughput. While there is a tradeoff between the number of tokens that can be read and the number of tokens that can be output, the baseline without SIMD can never produce more than a single token per update. 

Similarly, for a fixed point of 4 input tokens, the model could only output a single token before SIMD integration. With SIMD integration, it can now output 17 tokens, resulting in a 17x increase in output capability.

### Key-Value Caching
Key-Value (KV) Caching further enhances the model's performance by optimizing the input-output token ratio. The evaluation of KV caching shows significant improvements in efficiency and token generation speed.

#### Performance with and without KV Caching
| Input Tokens | Output Tokens without KV Caching | Output Tokens with KV Caching | Improvement Factor |
|--------------|----------------------------------|-------------------------------|--------------------|
| 48           | 1                                | 5                             | 5x                 |
| 24           | 2                                | 10                            | 5x                 |
| 12           | 5                                | 14                            | 2.8x                 |

The efficiency gains from KV caching are more pronounced with longer input sequences, showcasing a 5x improvement for 48 input tokens and a 2.8x improvement for 12 input tokens. This enhancement is also reflected in the speed of token generation.

### Conclusion
The integration of SIMD and Key-Value Caching significantly improves the model's performance, particularly in reading throughput and token generation efficiency. These optimizations are crucial for enhancing the overall capability of the language model.


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

5. Use the installed Cargo package to run specific tasks, such as uploading model chunks. Replace the demo model `gpt2_with_kv.onnx` with your actual model file names:
   ```bash
   ic-file-uploader gpt2_backend upload_model_bytes_chunks gpt2_with_kv.onnx
   ```
   
4. **Model Storage**: This will store the model to stable memory so that it can be efficiently loaded after redeployment:
   ```plaintext
    dfx canister call gpt2_backend upload_wasm_ref_cell_to_stable 
   ```

5. **Model Preparation**: Follow the commands to prepare the model for use:
   ```plaintext
    dfx canister call gpt2_backend setup_model
   ```

6. (Optional) Test the Model: 
   ```plaintext
   dfx canister call gpt2_backend model_inference '(14, vec {1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12})'
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
  python3 python/GPT2_kv_in.py
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
        ic-file-uploader gpt2_backend upload_model_bytes_chunks gpt2_with_kv.onnx
     ```

   - For Internet Computer mainnet deployment:
     ```bash
        ic-file-uploader gpt2_backend upload_model_bytes_chunks gpt2_with_kv.onnx --network ic
    ```

   - If an upload is interrupted, query the last successful upload with:
```plaintext
        dfx call canister gpt2_backend upload_wasm_ref_cell_length
```
     And resume uploading using the result:
```bash
        ic-file-uploader gpt2_backend upload_model_bytes_chunks gpt2_with_kv.onnx --offset <result number>
```


2. **Model Preparation**: Follow the commands to prepare the model for use:
   ```plaintext
    dfx canister call gpt2_backend setup_model
   ```
   
### Step 7: Interact with the Backend Canister API

- Access the backend Canister API at the given endpoint for testing:
   ```plaintext
   dfx canister call gpt2_backend model_inference '(14, vec {1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12})'
   ```

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
