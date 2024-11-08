
use crate::error::TokenizerError;
//use crate::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
//use protobuf::Message;
use std::collections::HashMap;
//use std::fs::File;
//use std::io::{BufRead, BufReader, Read};
//use std::io::BufRead;
use std::mem::ManuallyDrop;
//use std::path::Path;
use std::ptr;

/// # Byte pair query
/// Structure holding a pair of bytes for query in the BPE vocabulary
#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct BpePairRef<'a> {
    pub byte_1: &'a String,
    pub byte_2: &'a String,
}

/// # Byte pair Encoding Vocab
/// BPE vocab containing the merges (dictionary of pairs with their priority) used to merge
/// pairs together. This vocabulary element is used on BPE tokenizers such as GPT2 or RoBERTa.
/// This vocabulary is not meant to be used directly, but rather as part of a BPE Tokenizer.
#[derive(Debug, Clone)]
pub struct BpePairVocab {
    pub values: HashMap<(String, String), i64>,
}

impl BpePairVocab {

    pub fn from_cache(text: &str) -> Result<BpePairVocab, TokenizerError> {
        // Split the text into lines
        let lines = text.lines().skip(1);
        let mut data = HashMap::new();
        let mut index = 0;
        for line in lines {
            // Trim the line and split it into a tuple
            let tuple: Vec<String> = line.trim().split(' ').map(|v| v.to_owned()).collect();
            if tuple.len() > 1 {
                data.insert((tuple[0].clone(), tuple[1].clone()), index);
                index += 1;
            }
        }

        Ok(BpePairVocab { values: data })
    }

    /// Gets the id of a "byte pair" in the merges vocab. Returns an optional index for the pair if
    /// it is found in the vocabulary.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::vocab::{BpePairRef, BpePairVocab, Vocab};
    /// let path = "path/to/file";
    ///
    /// let bpe_vocab = BpePairVocab::from_file(path).unwrap();
    ///
    /// let query = BpePairRef {
    ///     byte_1: &"won".to_string(),
    ///     byte_2: &"derful".to_string(),
    /// };
    /// let id = bpe_vocab.byte_pair_to_id(&query);
    /// ```
    pub fn byte_pair_to_id(&self, byte_pair: &BpePairRef) -> Option<&i64> {
        unsafe {
            let byte_1 = byte_pair.byte_1;
            let byte_2 = byte_pair.byte_2;
            let k = (ptr::read(byte_1), ptr::read(byte_2));
            let k = ManuallyDrop::new(k);
            self.values.get(&k)
        }
    }
}
