
//use crate::{dfx::dfx, error::create_error_string};
use tempfile::NamedTempFile;
//use std::io::Write;
use std::process::Command;
use std::io::Write;


//pub const MAX_CANISTER_HTTP_PAYLOAD_SIZE: usize = 2 * 1024 * 1024; // 2 MiB
pub const MAX_CANISTER_HTTP_PAYLOAD_SIZE: usize = 2 * 1000 * 1000; // 2 MiB



// let path = "model/demo_regression.onnx"
// let canister_name = demo_gpt2_model_backend
// let canister_method_name = upload_model_chunks

use std::env;

//fn main(path: &str, canister_name: &str, canister_method_name: &str) -> Result<(), String> {
fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();

    // Ensure there are enough arguments
    if args.len() < 4 {
        return Err("Not enough arguments. Usage: <program> <path> <canister_name> <canister_method_name>".to_string());
    }

    let path = &args[1];
    let canister_name = &args[2];
    let canister_method_name = &args[3];

    println!("Hello, world!");

    let model_file = std::fs::read(
        path
    )
    .map_err(|e| error_to_string(&e))?;

    let model_chunks = split_into_chunks(model_file, MAX_CANISTER_HTTP_PAYLOAD_SIZE);

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

    Ok(())
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


