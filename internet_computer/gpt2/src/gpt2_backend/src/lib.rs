use ic_stable_structures::memory_manager::{MemoryId, MemoryManager, VirtualMemory};
use ic_stable_structures::{DefaultMemoryImpl, StableBTreeMap, Storable};
use std::cell::RefCell;

type Memory = VirtualMemory<DefaultMemoryImpl>;
mod onnx;
mod upload_utils;

thread_local! {
    pub static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));

    pub static MAP: RefCell<StableBTreeMap<u8, Vec<u8>, Memory>> = RefCell::new(
        StableBTreeMap::init(
            MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1))),
        )
    );
}

#[ic_cdk::init]
fn init() {
    // Initialize the WASI memory
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(0)));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);

    // Initialize the application memory (StableBTreeMap)
    let app_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1)));
    MAP.with(|map| {
        *map.borrow_mut() = StableBTreeMap::init(app_memory);
    });
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
    let app_memory = MEMORY_MANAGER.with(|m| m.borrow().get(MemoryId::new(1)));
    MAP.with(|map| {
        *map.borrow_mut() = StableBTreeMap::init(app_memory);
    });
}

