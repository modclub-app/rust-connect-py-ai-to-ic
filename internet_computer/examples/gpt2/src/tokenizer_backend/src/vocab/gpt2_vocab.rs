
use crate::error::TokenizerError;
//use crate::vocab::base_vocab::{read_json_string, read_json_file, read_special_token_mapping_file, swap_key_values, SpecialTokenMap, Vocab, };
use crate::vocab::base_vocab::{ read_json_string, swap_key_values, SpecialTokenMap, Vocab };
use std::collections::HashMap;
//use std::path::Path;



/// # GPT2 Vocab
/// Vocabulary for GPT2 tokenizer. Contains the following special values:
/// - BOS token
/// - EOS token
///
/// Expects a JSON-format vocabulary when created from file.
#[derive(Debug, Clone)]
pub struct Gpt2Vocab {
    /// A mapping of tokens as string to indices (i.e. the encoder base)
    pub values: HashMap<String, i64>,

    /// A mapping of token ids to strings (i.e. the decoder base)
    pub indices: HashMap<i64, String>,

    /// Special tokens used by the vocabulary
    pub special_token_map: SpecialTokenMap,

    /// A mapping of special value tokens as strings to IDs (i.e. the encoder base for special
    /// values), special values typically include things like BOS/EOS markers, class markers, mask
    /// markers and padding markers
    pub special_values: HashMap<String, i64>,

    /// A mapping of special value tokens as IDs to strings (i.e. the decoder base for special values)
    pub special_indices: HashMap<i64, String>,
}

const DEFAULT_UNK_TOKEN: &str = "<|endoftext|>";
const DEFAULT_BOS_TOKEN: &str = DEFAULT_UNK_TOKEN;
const DEFAULT_EOS_TOKEN: &str = DEFAULT_UNK_TOKEN;

impl Gpt2Vocab {
    pub fn get_bos_value(&self) -> &str {
        self.special_token_map
            .bos_token
            .as_deref()
            .unwrap_or(DEFAULT_BOS_TOKEN)
    }

    pub fn get_eos_value(&self) -> &str {
        self.special_token_map
            .eos_token
            .as_deref()
            .unwrap_or(DEFAULT_EOS_TOKEN)
    }
}

impl Vocab for Gpt2Vocab {
    fn get_unknown_value(&self) -> &str {
        &self.special_token_map.unk_token
    }

    fn values(&self) -> &HashMap<String, i64> {
        &self.values
    }

    fn indices(&self) -> &HashMap<i64, String> {
        &self.indices
    }

    fn special_values(&self) -> &HashMap<String, i64> {
        &self.special_values
    }

    fn special_indices(&self) -> &HashMap<i64, String> {
        &self.special_indices
    }

    fn values_mut(&mut self) -> &mut HashMap<String, i64> {
        &mut self.values
    }

    fn indices_mut(&mut self) -> &mut HashMap<i64, String> {
        &mut self.indices
    }

    fn special_values_mut(&mut self) -> &mut HashMap<String, i64> {
        &mut self.special_values
    }

    fn special_indices_mut(&mut self) -> &mut HashMap<i64, String> {
        &mut self.special_indices
    }

    fn from_cache(text: &str) -> Result<Gpt2Vocab, TokenizerError> {
        let values = read_json_string(text)?;
        //let values: Result<HashMap<String, i64>, serde_json::Error> = serde_json::from_str(text).expect("Failed to read Json");
        let special_token_map = SpecialTokenMap {
            unk_token: DEFAULT_UNK_TOKEN.to_string(),
            pad_token: None,
            bos_token: Some(DEFAULT_BOS_TOKEN.to_string()),
            sep_token: None,
            cls_token: None,
            eos_token: Some(DEFAULT_EOS_TOKEN.to_string()),
            mask_token: None,
            additional_special_tokens: None,
        };
        Self::from_values_and_special_token_map(values, special_token_map)
    }

    fn from_values_and_special_token_map(
        values: HashMap<String, i64>,
        special_token_map: SpecialTokenMap,
    ) -> Result<Self, TokenizerError>
    where
        Self: Sized,
    {
        let mut special_values = HashMap::new();
        special_token_map.register_special_values(&values, &mut special_values)?;

        let indices = swap_key_values(&values);
        let special_indices = swap_key_values(&special_values);
        Ok(Self {
            values,
            indices,
            special_token_map,
            special_values,
            special_indices,
        })
    }
    fn token_to_id(&self, token: &str) -> i64 {
        self._token_to_id(
            token,
            &self.values,
            &self.special_values,
            self.get_unknown_value(),
        )
    }

    fn id_to_token(&self, id: &i64) -> String {
        self._id_to_token(
            id,
            &self.indices,
            &self.special_indices,
            self.get_unknown_value(),
        )
    }
}
