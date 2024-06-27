use std::cell::RefCell;
use bytes::Bytes;

thread_local! {
    static WASM_REF_CELL: RefCell<Vec<u8>> = RefCell::new(vec![]);
}

pub fn call_model_bytes() -> Result<Bytes, String> {
    WASM_REF_CELL.with(|wasm_ref_cell| {
        // Use `std::mem::take` to replace the contents of the cell with an empty vector
        // and return the original contents.
        let wasm_ref = std::mem::take(&mut *wasm_ref_cell.borrow_mut());

        // Convert the vector into a `Bytes` instance. This avoids cloning as the vector's
        // ownership is transferred to the `Bytes` instance.
        let model_bytes = Bytes::from(wasm_ref);

        Ok(model_bytes)
    })
}


#[ic_cdk::update]
pub fn upload_model_bytes_chunks(bytes: Vec<u8>) { // -> Result<(), String>
    WASM_REF_CELL.with(|wasm_ref_cell| {
        let mut wasm_ref_mut = wasm_ref_cell.borrow_mut();
        wasm_ref_mut.extend(bytes);
    });
}

#[ic_cdk::query]
pub fn upload_wasm_ref_cell_length() -> usize {
    WASM_REF_CELL.with(|wasm_ref_cell| wasm_ref_cell.borrow().len())
}

#[ic_cdk::update]
pub fn upload_wasm_ref_cell_clear() {
    WASM_REF_CELL.with(|wasm_ref_cell| {
        wasm_ref_cell.borrow_mut().clear();
    });
}

