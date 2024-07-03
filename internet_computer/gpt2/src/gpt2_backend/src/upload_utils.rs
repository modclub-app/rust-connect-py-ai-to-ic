use std::cell::RefCell;
use bytes::Bytes;
use crate::MAP; // Import MAP

thread_local! {
    static WASM_REF_CELL: RefCell<Vec<u8>> = RefCell::new(vec![]);
}


// Inserts an entry into the map from a sample `Vec<u8>` using the fixed key (0_u8).
#[ic_cdk::update]
fn insert_sample_data(){
    let data: Vec<u8> = vec![1, 2, 3];
    MAP.with(|p| {
        let mut map = p.borrow_mut();
        map.insert(0_u8, data)
    });
}

// Inserts an entry into the map from WASM_REF_CELL using the fixed key (0_u8).
#[ic_cdk::update]
fn upload_wasm_ref_cell_to_stable() {
    WASM_REF_CELL.with(|cell| {
        //let data = std::mem::take(&mut *cell.borrow_mut());
        let data = cell.borrow().clone();
        MAP.with(|p| {
            let mut map = p.borrow_mut();
            map.insert(0_u8, data);
        });
    });
}


// Inserts an entry from the map into WASM_REF_CELL using the fixed key (0_u8).
#[ic_cdk::update]
fn upload_wasm_ref_cell_from_stable() {
    WASM_REF_CELL.with(|wasm_ref_cell| {
        let mut wasm_ref_mut = wasm_ref_cell.borrow_mut();
        MAP.with(|p| {
            if let Some(data) = p.borrow().get(&0_u8) {
                wasm_ref_mut.extend(data);
            }
        });
    });
}


// Retrieves the data from WASM_REF_CELL and converts it to Bytes.
pub fn call_model_bytes() -> Result<Bytes, String> {
    WASM_REF_CELL.with(|wasm_ref_cell| {
        let wasm_ref = std::mem::take(&mut *wasm_ref_cell.borrow_mut());
        let model_bytes = Bytes::from(wasm_ref);
        Ok(model_bytes)
    })
}

// Appends the given bytes to WASM_REF_CELL.
#[ic_cdk::update]
pub fn upload_model_bytes_chunks(bytes: Vec<u8>) {
    WASM_REF_CELL.with(|wasm_ref_cell| {
        let mut wasm_ref_mut = wasm_ref_cell.borrow_mut();
        wasm_ref_mut.extend(bytes);
    });
}

// Returns the length of WASM_REF_CELL.
#[ic_cdk::query]
pub fn upload_wasm_ref_cell_length() -> usize {
    WASM_REF_CELL.with(|wasm_ref_cell| wasm_ref_cell.borrow().len())
}

// Clears the contents of WASM_REF_CELL.
#[ic_cdk::update]
pub fn upload_wasm_ref_cell_clear() {
    WASM_REF_CELL.with(|wasm_ref_cell| wasm_ref_cell.borrow_mut().clear());
}
