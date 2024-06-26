use std::cell::RefCell;
//use candid::{CandidType, Deserialize};
use ic_stable_structures::{memory_manager::{MemoryId, MemoryManager}, DefaultMemoryImpl};

mod onnx;
mod upload_utils;
//mod stable_storage;


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
