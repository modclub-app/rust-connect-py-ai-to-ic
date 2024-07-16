use std::cell::RefCell;
//use candid::{CandidType, Deserialize};
use ic_stable_structures::{memory_manager::{MemoryId, MemoryManager}, DefaultMemoryImpl};

mod onnx;
mod storage;

// WASI polyfill requires a virtual stable memory to store the file system.
// You can replace `0` with any index up to `254`.
const WASI_MEMORY_ID: MemoryId = MemoryId::new(0);

thread_local! {
    // The memory manager is used for simulating multiple memories.
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));

}



#[ic_cdk::query]
fn model_inference(numbers: Vec<i64>) -> Result<Vec<f32>, String> {
    match onnx::create_tensor_and_run_model(numbers) {
        Ok(result) => Ok(result),
        Err(err) => Err(err.to_string()),
    }
}


////////////////////////////////////////////////////////

#[ic_cdk::init]
fn init() {
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
    //onnx::setup().unwrap(); //do not load initially because need to pass the model into memory
}

#[ic_cdk::post_upgrade]
fn post_upgrade() {
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
    //onnx::setup().unwrap();
}

//////////////////////////////////////////////////////////////////////



const MODEL_FILE: &str = "onnx_model.onnx";

/// Clears the face detection model file.
/// This is used for incremental chunk uploading of large files.
#[ic_cdk::update]
//fn clear_model_bytes() {
fn upload_wasm_ref_cell_clear() {
    storage::clear_bytes(MODEL_FILE);
}

/// Appends the given chunk to the face detection model file.
/// This is used for incremental chunk uploading of large files.
#[ic_cdk::update]
//fn append_model_bytes(bytes: Vec<u8>) {
fn upload_model_bytes_chunks(bytes: Vec<u8>) {
    storage::append_bytes(MODEL_FILE, bytes);
}



/// Returns the length of the model bytes.
#[ic_cdk::query]
//fn get_model_bytes_length() -> usize {
fn upload_wasm_ref_cell_length() -> usize {
    storage::bytes_length(MODEL_FILE)
}
