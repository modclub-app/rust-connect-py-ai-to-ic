# multi_query_single_canister_demo



## Usage

Once the setup is complete, you can proceed with the following steps to build, deploy, and run your project.

1. Build the Rust project targeting WebAssembly:
   ```bash
   cargo build --target wasm32-unknown-unknown --release -p multi_query_single_canister_demo_backend
   ```
2. Start the Dfinity network locally in the background:
   ```bash
   dfx start --background
   ```
3. Deploy your project using `dfx`:
   ```bash
   dfx deploy
   ```
4. **Initialization**: Start by initializing the model pipeline.

   ```plaintext
   initialize_model_pipeline : () -> ();
   ```
5. Use the Cargo command to run specific tasks, such as uploading model chunks. Replace the demo models `[gpt2_embedding.onnx, gpt2_layer_0.onnx]` with your actual model file names:
   ```bash
   cargo run --manifest-path ../../rust/upload_byte_file/Cargo.toml demo_gpt2_model_backend upload_model_chunks ../../python/onnx_model/ [gpt2_embedding.onnx, gpt2_layer_0.onnx] 0
   ```
   
6. **Model Preparation**: Follow the commands to prepare the model for use:

   ```plaintext
   model_plan_to_plan: () → ();
   plan_to_running_model: () → ();
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

### Step 2: Partition the GPT-2 Model

- Run the script to partition the GPT-2 model, preparing it for backend use:

  ```bash
  python3 python/GPT2_max_partition_model_pool.py
  ```

### Step 3: Build and Deploy

1. cargo build --target wasm32-unknown-unknown --release -p multi_query_single_canister_demo_backend
2. dfx start --background
3. dfx deploy

### Step 4: Load the Model into the Backend

1. **Initialization**: Start by initializing the model pipeline.

   ```plaintext
   initialize_model_pipeline : () -> ();
   ```

2. **Model Upload**: Navigate to the canister scripts directory and perform the following:

   - For local deployment:

     ```bash
     cargo run --manifest-path ../../rust/upload_byte_file/Cargo.toml single_query_demo_backend upload_model_chunks ../../python/onnx_model/ [gpt2_embedding.onnx, gpt2_layer_0.onnx] 0
     ```

   - For Internet Computer mainnet deployment:

     ```bash
     cargo run --manifest-path ../../rust/upload_byte_file/Cargo.toml <canister id> upload_model_chunks ../../python/onnx_model/ [gpt2_embedding.onnx, gpt2_layer_0.onnx] 0 ic
     ```

   - If an upload is interrupted, query the last successful upload with:

     ```plaintext
     "wasm_ref_cell_length": () -> (nat64) query;
     ```

     And resume uploading using the result:

     ```bash
     cargo run --manifest-path ../../rust/upload_byte_file/Cargo.toml single_query_demo_backend upload_model_chunks ../../python/onnx_model/ [gpt2_embedding.onnx] <result number>
     ```

3. **Model Preparation**: Follow the commands to prepare the model for use:

   ```plaintext
   model_plan_to_plan: () → ();
   plan_to_running_model: () → ();
   ```

### Step 5: Interact with the Backend Canister API

- Access the backend Canister API at the given endpoint for testing:

  ```plaintext
  word_embeddings: (vec int64) → (vec float32) composite_query
  ```

### Step 6: Test the API

- Demonstrate the model's functionality by inputting tokens such as `[1, 4, 5]` to generate word embeddings.





