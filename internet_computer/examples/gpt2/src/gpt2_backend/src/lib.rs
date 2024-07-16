//use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
//use ic_stable_structures::{DefaultMemoryImpl, StableBTreeMap}; //, Storable};
use ic_stable_structures::{memory_manager::{MemoryId, MemoryManager}, DefaultMemoryImpl};

use std::cell::RefCell;

//type Memory = VirtualMemory<DefaultMemoryImpl>;
mod onnx;
mod storage;

thread_local! {
    pub static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));

    //pub static MAP: RefCell<StableBTreeMap<u8, Vec<u8>, Memory>> = RefCell::new(
    //    StableBTreeMap::init(
    //        MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1))),
    //    )
    //);
}

#[ic_cdk::init]
fn init() {
    // Initialize the WASI memory
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(0)));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);

    // Initialize the application memory (StableBTreeMap)
    //let app_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1)));
    //MAP.with(|map| {
    //    *map.borrow_mut() = StableBTreeMap::init(app_memory);
    //});
}

#[ic_cdk::pre_upgrade]
fn pre_upgrade() {
    // Save any necessary state before upgrade if needed
}

#[ic_cdk::post_upgrade]
fn post_upgrade() {
    // Reinitialize the WASI memory after upgrade
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(0)));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);

    // Reinitialize the application memory (StableBTreeMap) after upgrade
    //let app_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1)));
    //MAP.with(|map| {
    //    *map.borrow_mut() = StableBTreeMap::init(app_memory);
    //});
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
