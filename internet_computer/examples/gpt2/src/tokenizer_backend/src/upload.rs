use std::cell::RefCell;


thread_local! {
    pub static WASM_REF_CELL: RefCell<Vec<u8>> = RefCell::new(vec![]);
}

#[ic_cdk::query]
fn wasm_ref_cell_length() -> usize {
    WASM_REF_CELL.with(|wasm_ref_cell| wasm_ref_cell.borrow().len())
}

#[ic_cdk::update]
fn clear_wasm_ref_cell() {
    WASM_REF_CELL.with(|wasm_ref_cell| {
        wasm_ref_cell.borrow_mut().clear();
    });
}

#[ic_cdk::update]
pub fn upload_data_chunks(bytes: Vec<u8>) { // -> Result<(), String>
    WASM_REF_CELL.with(|wasm_ref_cell| {
        let mut wasm_ref_mut = wasm_ref_cell.borrow_mut();
        wasm_ref_mut.extend(bytes);
    });
}
