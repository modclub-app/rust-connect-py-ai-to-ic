
// Uploading and Initializing Model
mod upload;





use std::cell::RefCell;


pub mod tokenizer;
pub mod vocab;
pub mod adapters;
pub mod error;
pub use tokenizer::base_tokenizer::{
    ConsolidatableTokens, ConsolidatedTokenIterator, Mask, Offset, OffsetSize, Token,
    TokenIdsWithOffsets, TokenIdsWithSpecialTokens, TokenRef, TokenTrait, TokenizedInput,
    TokensWithOffsets,
};
use tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use vocab::{BpePairVocab, Gpt2Vocab, Vocab};

/*
#############################

#############################
*/



thread_local! {
    static TOKENIZER: RefCell<Option<Gpt2Tokenizer>> = RefCell::new(None);
    static VOCAB_STR: RefCell<Option<String>> = RefCell::new(None);
    static MERGES_STR: RefCell<Option<String>> = RefCell::new(None);
}


#[ic_cdk::update]
fn bytes_to_vocab() -> String {
    // Attempt to retrieve and process vocab bytes
    let vocab_result = upload::WASM_REF_CELL.with(|cell| {
        // This empties the WASM_REF_CELL and retrieves its contents.
        let vocab_vec = std::mem::take(&mut *cell.borrow_mut());
        match std::str::from_utf8(&vocab_vec) {
            Ok(vocab_str) => {
                VOCAB_STR.with(|vocab_str_ref_cell| {
                    *vocab_str_ref_cell.borrow_mut() = Some(vocab_str.to_string());
                });
                Ok(())
            },
            Err(e) => Err(e.to_string()),
        }
    });

    // Decide what message to return based on the operation result
    match vocab_result {
        Ok(_) => "vocab.json loaded successfully.".to_string(),
        Err(e) => format!("Failed to load vocab.json: {}", e),
    }
}



#[ic_cdk::update]
fn bytes_to_merges() -> String {
    // Attempt to retrieve and process vocab bytes
    let merges_result = upload::WASM_REF_CELL.with(|cell| {
        // This empties the WASM_REF_CELL and retrieves its contents.
        let merges_vec = std::mem::take(&mut *cell.borrow_mut());
        match std::str::from_utf8(&merges_vec) {
            Ok(merges_str) => {
                MERGES_STR.with(|merges_str_ref_cell| {
                    *merges_str_ref_cell.borrow_mut() = Some(merges_str.to_string());
                });
                Ok(())
            },
            Err(e) => Err(e.to_string()),
        }
    });

    // Decide what message to return based on the operation result
    match merges_result {
        Ok(_) => "merges.txt loaded successfully.".to_string(),
        Err(e) => format!("Failed to load merges.txt: {}", e),
    }
}

#[ic_cdk::update]
fn initialize_tokenizer() -> String {
    // Attempt to retrieve and process the vocab and merges strings
    let vocab_str = match VOCAB_STR.with(|vocab_ref_cell| vocab_ref_cell.take()) {
        Some(str) => str,
        None => return "Vocab string not found.".to_string(),
    };

    let merges_str = match MERGES_STR.with(|merges_ref_cell| merges_ref_cell.take()) {
        Some(str) => str,
        None => return "Merges string not found.".to_string(),
    };

    // Attempt to initialize the tokenizer with the processed data
    match Gpt2Vocab::from_cache(&vocab_str) {
        Ok(vocab) => match BpePairVocab::from_cache(&merges_str) {
            Ok(merges) => {
                let tokenizer = Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, false);
                TOKENIZER.with(|tokenizer_ref_cell| {
                    *tokenizer_ref_cell.borrow_mut() = Some(tokenizer);
                });
                "Tokenizer initialized successfully.".to_string()
            },
            Err(_) => "Failed to load merges from cache.".to_string(),
        },
        Err(_) => "Failed to load vocab from cache.".to_string(),
    }
}

/*
pub fn tokenize_text(text: &str) -> Result<Vec<i64>> {


    // Retrieve a locked reference to the tokenizer
    let tokenizer_lock = get_tokenizer().ok_or("Failed to acquire tokenizer lock").expect("Failed to acquire tokenizer lock");
    let tokenizer = tokenizer_lock.as_ref().ok_or("Tokenizer is not initialized").expect("Tokenizer is not initialized");


    let tokenized_input = tokenizer.encode(text, None, 128, &TruncationStrategy::LongestFirst, 0);

    let token_ids = tokenized_input.token_ids.clone();
    Ok(token_ids)

}
*/
#[ic_cdk::query]
pub fn tokenize_text(text: String) -> (Vec<i32>, Vec<String>) {
    TOKENIZER.with(|tokenizer_ref_cell| {
        if let Some(tokenizer) = tokenizer_ref_cell.borrow().as_ref() {
            let tokenized_input = tokenizer.encode(&text, None, 128, &TruncationStrategy::LongestFirst, 0);
            let token_ids: Vec<i64> = tokenized_input.token_ids.clone();

            // Convert i64 to i32, clamping values that are out of i32 range
            let token_ids_i32: Vec<i32> = token_ids.iter()
                .map(|&id| id.clamp(i32::MIN as i64, i32::MAX as i64) as i32)
                .collect();

            let tokens_list = tokenizer.decode_to_vec(&token_ids, true);
            let tokens_cleaned = tokens_list.iter().map(|t| t.replace("Ġ", " ")).collect();

            (token_ids_i32, tokens_cleaned)
        } else {
            (vec![], vec![])
        }
    })
}

#[ic_cdk::query]
pub fn decode_token_ids(input_ids: Vec<i32>) -> Vec<String> {
    TOKENIZER.with(|tokenizer_ref_cell| {
        if let Some(tokenizer) = tokenizer_ref_cell.borrow().as_ref() {
            // Convert i32 to i64
            let input_ids_i64: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();

            let tokens_list = tokenizer.decode_to_vec(&input_ids_i64, true);
            let tokens_cleaned: Vec<String> = tokens_list.iter().map(|t| t.replace("Ġ", " ").replace("Ċ", "\n")).collect();
            //let tokens_cleaned: Vec<String> = tokens_list.iter().map(|t| t.replace("Ġ", " ").replace("Ċ", "{'\n'}")).collect();
            //let tokens_cleaned: Vec<String> = tokens_list.iter().map(|t| t.replace("Ġ", " ").replace("Ċ", "<br />")).collect();
            tokens_cleaned
        } else {
            vec![]
        }
    })
}
