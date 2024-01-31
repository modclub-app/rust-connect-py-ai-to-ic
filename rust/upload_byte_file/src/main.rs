

use std::env;
use std::fs;
use std::process::Command;
use std::io::Write;
use tempfile::NamedTempFile;
use std::path::Path;

pub const MAX_CANISTER_HTTP_PAYLOAD_SIZE: usize = 2 * 1000 * 1000; // 2 MiB
//cargo run --manifest-path ../../rust/upload_byte_file/Cargo.toml demo_gpt2_model_backend upload_model_chunks ../../python/onnx_model/ <gpt2_embedding.onnx,gpt2_layer_0.onnx>



//fn main(path: &str, canister_name: &str, canister_method_name: &str) -> Result<(), String> {
fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();

    // Ensure there are enough arguments
    if args.len() < 5 {
        return Err("Not enough arguments. Usage: <program> <canister_name> <canister_method_name> <model_folder_path> <models_list>".to_string());
    }

    //let program = &args[0];                                        //         ../../rust/upload_byte_file/Cargo.toml
    let canister_name = &args[1];                               //          demo_gpt2_model_backend
    let canister_method_name = &args[2];                        //          upload_model_chunks
    let model_directory = &args[3];                                  //          ../../python/onnx_model/
    //let model_files: Vec<&str> = args[4].split(',').collect();  //          <gpt2_embedding.onnx,gpt2_layer_0.onnx>
    let model_files_input = args[4].trim_matches(|c| c == '[' || c == ']');
    let model_files: Vec<&str> = model_files_input.split(',').collect();

    //println!("Hello, world!");
    simple_dfx_execute(canister_name, "initialize_model_pipeline");


    for model_file in model_files {
        let model_path = Path::new(model_directory).join(model_file);
        let model_path_str = model_path.to_str().ok_or("Failed to convert path to string")?;

        println!("Uploading {}", model_path_str);

        let model_data = fs::read(&model_path)
            .map_err(|e| e.to_string())?;

        let model_chunks = split_into_chunks(model_data, MAX_CANISTER_HTTP_PAYLOAD_SIZE);

        for (index, model_chunk) in model_chunks.iter().enumerate() {
            upload_chunk(
                &format!("{canister_name} model"),
                canister_name,
                model_chunk,
                canister_method_name,
                index,
                model_chunks.len(),
            )?;
        }

        simple_dfx_execute(canister_name, "model_bytes_to_plan");
        simple_dfx_execute(canister_name, "plan_to_running_model");
    }

    // loop through the models

    Ok(())
}

//pub fn simple_dfx_execute(canister_name: &str, canister_method_name: &str) -> Result<(), String> {
pub fn simple_dfx_execute(canister_name: &str, canister_method_name: &str){
    let output = dfx(
        "canister",
        "call",
        &vec![
            canister_name,
            canister_method_name,
        ],
    ).expect("Simple DFX Command Failed");
}


pub fn split_into_chunks(data: Vec<u8>, chunk_size: usize) -> Vec<Vec<u8>> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let data_len = data.len();

    while start < data_len {
        let end = usize::min(start + chunk_size, data_len);
        chunks.push(data[start..end].to_vec());
        start = end;
    }
    chunks
}


fn vec_u8_to_blob_string(data: &[u8]) -> String {
    let mut result = String::from("(blob \"");
    for &byte in data {
        result.push_str(&format!("\\{:02X}", byte));
    }
    result.push_str("\")");
    result
}



pub fn upload_chunk(name: &str,
    canister_name: &str,
    bytecode_chunk: &Vec<u8>,
    canister_method_name: &str,
    chunk_number: usize,
    chunk_total: usize) -> Result<(), String> {

    let blob_string = vec_u8_to_blob_string(bytecode_chunk);

    let mut temp_file =
        NamedTempFile::new()
        .map_err(|_| create_error_string("Failed to create temporary file"))?;

    temp_file
        .as_file_mut()
        .write_all(blob_string.as_bytes())
        .map_err(|_| create_error_string("Failed to write data to temporary file"))?;


    // Read the contents of the temporary file into a string
    //let temp_file_path = temp_file.path().to_str().ok_or(create_error_string(
    //    "temp_file path could not be converted to &str",
    //))?;
    //let file_contents = std::fs::read_to_string(temp_file_path)
    //    .map_err(|e| create_error_string(&format!("Failed to read temporary file: {}", e)))?;


    let output = dfx(
        "canister",
        "call",
        &vec![
            canister_name,
            canister_method_name,
            "--argument-file",
            temp_file.path().to_str().ok_or(create_error_string(
                "temp_file path could not be converted to &str",
            ))?,
            //&file_contents
        ],
    )?;

    let chunk_number = chunk_number + 1;

    if output.status.success() {
        println!(
            "{}",
            format!("Uploading {name} chunk {chunk_number}/{chunk_total}")
        );
    } else {
        return Err(create_error_string(&String::from_utf8_lossy(
            &output.stderr,
        )));
    }

    Ok(())
}


pub fn dfx(command: &str, subcommand: &str, args: &Vec<&str>) -> Result<std::process::Output, String> {

    //let dfx_network = std::env::var("DFX_NETWORK")
    //    .map_err(|_| create_error_string("DFX_NETWORK environment variable not present"))?;

    let mut dfx_command = Command::new("dfx");
    dfx_command.arg(command);
    dfx_command.arg(subcommand);
    //dfx_command.arg("--network");
    //dfx_command.arg(dfx_network);

    for arg in args {    dfx_command.arg(arg);    }

    dfx_command.output().map_err(|e| e.to_string())
}



pub fn error_to_string(e: &dyn std::error::Error) -> String {
    format!("Upload Error: {}", e.to_string())
}


pub fn create_error_string(message: &str) -> String {
    format!("Upload Error: {message}")
}


