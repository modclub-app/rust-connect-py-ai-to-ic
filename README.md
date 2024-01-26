# rust-connect-py-ai-to-ic

Seamlessly Bridge Python AI Models with the Internet Computer.

## Overview

"Python (design and train), Rust (upload), IC (run)" - this succinct mantra captures the essence of `rust-connect-py-ai-to-ic`. Our toolkit is an open-source solution designed to enable AI researchers and developers to effortlessly upload and deploy Python-based machine learning models for inference on the Internet Computer (IC) platform.

Focused on ease of use and versatility, `rust-connect-py-ai-to-ic` aims to streamline the integration of advanced AI capabilities into the decentralized environment of the IC. This project represents a pivotal step towards harnessing the power of the IC for AI, with potential future expansions into model training and user-friendly drag-and-drop functionalities.

## Features

- **Effortless Upload**: Simplify the process of uploading Python-based AI models to the IC using Rust.
- **Inference on IC**: Run your machine learning models on a decentralized platform, leveraging the unique capabilities of the Internet Computer.
- **Future Expansion**: Potential for extending the toolkit to include model training and easy-to-use drag-and-drop features.


## Getting Started

This section guides you through the initial setup of the necessary tools and environments for this project. We are using Rust with WebAssembly, Python with PyTorch, and the Dfinity platform.

### Rust Setup

First, ensure you have Rust installed. We will then set the default toolchain to stable and add the WebAssembly target.

1. Install Rust and Cargo (if not already installed): Visit [Rust's installation page](https://www.rust-lang.org/tools/install).
2. Set the default toolchain to stable:
   ```bash
   rustup default stable
   ```
3. Add the WebAssembly target:
   ```bash
   rustup target add wasm32-unknown-unknown
   ```
4. Add Cargo's bin directory to your PATH:
   ```bash
   export PATH="$PATH:~/.cargo/bin"
   ```

### Python and PyTorch Setup

Ensure you have Python installed and then set up PyTorch.

1. Install Python (if not already installed): Visit [Python's download page](https://www.python.org/downloads/).
2. Install PyTorch using pip:
   ```bash
   pip install torch
   ```

### Dfinity's DFX Setup

We will be using Dfinity's `dfx` for our development environment.

1. Install Dfinity's `dfx`: Follow the instructions on [Dfinity's SDK documentation](https://sdk.dfinity.org/docs/quickstart/quickstart.html).

## Usage

Once the setup is complete, you can proceed with the following steps to build, deploy, and run your project.

1. Build the Rust project targeting WebAssembly:
   ```bash
   cargo build --target wasm32-unknown-unknown --release -p demo_gpt2_model_backend
   ```
2. Start the Dfinity network locally in the background:
   ```bash
   dfx start --background
   ```
3. Deploy your project using `dfx`:
   ```bash
   dfx deploy
   ```
4. Use the Cargo command to run specific tasks, such as uploading model chunks. Replace `<model_part_1.onnx,model_part_2.onnx>` with your actual model file names:
   ```bash
   cargo run --manifest-path ../../rust/upload_byte_file/Cargo.toml demo_gpt2_model_backend upload_model_chunks ../../python/onnx_model/ <model_part_1.onnx,model_part_2.onnx>
   ```


## Contributing

We welcome contributions! Please read our contributing guidelines to get started.

## License

(Include license information here)

## Acknowledgements

(Optionally include acknowledgements or credits here)
