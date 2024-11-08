
use crate::error::TokenizerError;
//use crate::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
//use protobuf::Message;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
//use std::fs::File;
use std::hash::Hash;
//use std::io::{self, BufRead, BufReader, Read};
use std::io::{self, BufRead};
//use std::path::Path;
//use std::iter;

pub(crate) fn swap_key_values<T: Clone, U: Hash + Eq + Copy>(
    input_hashmap: &HashMap<T, U>,
) -> HashMap<U, T> {
    input_hashmap
        .iter()
        .map(|(key, &value)| (value, key.clone()))
        .collect()
}

pub(crate) fn read_flat_string(text: &str) -> Result<HashMap<String, i64>, TokenizerError> {
    // Convert the text to a buffered reader.
    let cursor = io::Cursor::new(text);
    let br = io::BufReader::new(cursor);

    let mut data = HashMap::new();

    for (index, line) in br.lines().enumerate() {
        let line = match line {
            Ok(value) => value,
            Err(e) => {
                return Err(TokenizerError::VocabularyParsingError(e.to_string()));
            }
        };
        data.insert(line.trim().to_owned(), index as i64);
    }
    Ok(data)
}



pub(crate) fn read_json_string(
    json_str: &str,
) -> Result<HashMap<String, i64>, TokenizerError> {
    let values: HashMap<String, i64> = match serde_json::from_str(json_str) {
        Ok(value) => value,
        Err(e) => {
            return Err(TokenizerError::VocabularyParsingError(e.to_string()));
        }
    };
    Ok(values)
}



/// Register a token as a special value
///
/// # Parameters
/// - token (`&str`): token to register as a special value
/// - values (`&HashMap<String, i64>`): mapping from tokens to ids. This should contain the token to add and will be used to read the id for registration in `special_values`
/// - special_values (`&HashMap<String, i64>`): mapping from special tokens to ids
pub(crate) fn register_as_special_value(
    token: &str,
    values: &HashMap<String, i64>,
    special_values: &mut HashMap<String, i64>,
) -> Result<(), TokenizerError> {
    let token_id = match values.get(token) {
        Some(index) => *index,
        None => {
            return Err(TokenizerError::TokenNotFound(format!(
                "The special value {token} could not be found in the vocabulary"
            )));
        }
    };
    special_values.insert(String::from(token), token_id);
    Ok(())
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct SpecialTokenMap {
    pub unk_token: String,
    pub pad_token: Option<String>,
    pub bos_token: Option<String>,
    pub sep_token: Option<String>,
    pub cls_token: Option<String>,
    pub eos_token: Option<String>,
    pub mask_token: Option<String>,
    pub additional_special_tokens: Option<HashSet<String>>,
}

impl SpecialTokenMap {
    /// Modifies special_values in-place, registering the existing special tokens registered in the
    /// special token map. Indices must be present in the provided `value` reference mapping.
    pub(crate) fn register_special_values(
        &self,
        values: &HashMap<String, i64>,
        special_values: &mut HashMap<String, i64>,
    ) -> Result<(), TokenizerError> {
        register_as_special_value(self.unk_token.as_str(), values, special_values)?;
        if let Some(pad_token) = &self.pad_token {
            register_as_special_value(pad_token, values, special_values)?;
        }
        if let Some(bos_token) = &self.bos_token {
            register_as_special_value(bos_token, values, special_values)?;
        }
        if let Some(sep_token) = &self.sep_token {
            register_as_special_value(sep_token, values, special_values)?;
        }
        if let Some(cls_token) = &self.cls_token {
            register_as_special_value(cls_token, values, special_values)?;
        }
        if let Some(eos_token) = &self.eos_token {
            register_as_special_value(eos_token, values, special_values)?;
        }
        if let Some(mask_token) = &self.mask_token {
            register_as_special_value(mask_token, values, special_values)?;
        }
        if let Some(additional_special_tokens) = &self.additional_special_tokens {
            for token in additional_special_tokens {
                register_as_special_value(token, values, special_values)?;
            }
        }
        Ok(())
    }
}

/// # Base Vocab trait
/// Defines a common interface to the vocabularies for use in the tokenizers.
pub trait Vocab {
    /// Returns the unknown value on an instance
    fn get_unknown_value(&self) -> &str;

    /// Return the map of token strings to IDs
    fn values(&self) -> &HashMap<String, i64>;

    /// Return the map of token IDs to strings
    fn indices(&self) -> &HashMap<i64, String>;

    /// Return the map of token strings to IDs
    fn special_values(&self) -> &HashMap<String, i64>;

    /// Return the map of token IDs to strings for special values
    fn special_indices(&self) -> &HashMap<i64, String>;

    /// Return a mutable reference to the map of token strings to IDs
    fn values_mut(&mut self) -> &mut HashMap<String, i64>;

    /// Return a mutable reference to the map of token IDs to strings
    fn indices_mut(&mut self) -> &mut HashMap<i64, String>;

    /// Return a mutable reference to the map of token strings to IDs
    fn special_values_mut(&mut self) -> &mut HashMap<String, i64>;

    /// Return a mutable reference to the map of token IDs to strings for special values
    fn special_indices_mut(&mut self) -> &mut HashMap<i64, String>;


    fn from_cache(text: &str) -> Result<Self, TokenizerError>
    where
        Self: Sized;


    fn from_values_and_special_token_map(
        values: HashMap<String, i64>,
        special_token_map: SpecialTokenMap,
    ) -> Result<Self, TokenizerError>
    where
        Self: Sized;

    /// Converts a token to an id, provided a `HashMap` of values, a `HashMap` of special values and
    /// the unknown value token string representation. This is not meant to be directly used, the method
    /// `token_to_id` offers a more convenient interface for most vocabularies, but needs to be implemented
    /// by the specific vocabulary.
    ///
    /// # Parameters
    /// - token (`&str`): token to convert
    /// - values (`&HashMap<String, i64>`): mapping from tokens to ids
    /// - special_values (`&HashMap<String, i64>`): mapping from special tokens to ids
    /// - unknown_value (`&str`): unknown token value
    ///
    /// # Returns
    /// - `i64`: index value for the provided token
    fn _token_to_id(
        &self,
        token: &str,
        values: &HashMap<String, i64>,
        special_values: &HashMap<String, i64>,
        unknown_value: &str,
    ) -> i64 {
        match special_values.get(token) {
            Some(index) => *index,
            None => match values.get(token) {
                Some(index) => *index,
                None => *values.get(unknown_value).unwrap(),
            },
        }
    }

    /// Converts an id to a token, provided a `HashMap` of values, a `HashMap` of special values and
    /// the unknown value token string representation. This is not meant to be directly used, the method
    /// `id_to_token` offers a more convenient interface for most vocabularies, but needs to be implemented
    /// by the specific vocabulary.
    ///
    /// # Parameters
    /// - id (`&i64`): token id to convert
    /// - indices (`&HashMap<i64, String>`): mapping from tokens to ids
    /// - special_indices (`&HashMap<i64, String>`): mapping from special tokens to ids
    /// - unknown_value (`&str`): unknown token value
    ///
    /// # Returns
    /// - `String`: token value for the index provided. If not found in the indices, returns the unknown token value
    fn _id_to_token(
        &self,
        id: &i64,
        indices: &HashMap<i64, String>,
        special_indices: &HashMap<i64, String>,
        unknown_value: &str,
    ) -> String {
        match special_indices.get(id) {
            Some(token) => token.clone(),
            None => match indices.get(id) {
                Some(token) => token.clone(),
                None => unknown_value.to_owned(),
            },
        }
    }

    /// Converts a token to an id.
    ///
    /// # Parameters
    /// - token (`&str`): token to convert
    ///
    /// # Returns
    /// - `i64`: token index for the value provided. If not found in the indices, returns the unknown token index
    fn token_to_id(&self, token: &str) -> i64;

    /// Converts an id to a token.
    ///
    /// # Parameters
    /// - id (`&i64`): token id to convert
    ///
    /// # Returns
    /// - `String`: token value for the index provided. If not found in the indices, returns the unknown token value
    fn id_to_token(&self, id: &i64) -> String;

    /// Converts a list of tokens to a list of indices.
    ///
    /// # Parameters
    /// - tokens (`&[&str]`): list of tokens to convert
    ///
    /// # Returns
    /// - `Vec<i64>`: Vector containing the indices for the tokens provided
    fn convert_tokens_to_ids(&self, tokens: &[&str]) -> Vec<i64> {
        tokens.iter().map(|v| self.token_to_id(v)).collect()
    }

    /// Add extra token ids to the vocab
    ///
    /// These tokens are generated automatically using the `<extra_id_{i}>` template and appended to
    /// the vocabulary. These are ignored from the tokenization algorithm chosen (pre-tokenized).
    /// This is used by some architectures to allow for further task-specific token identified
    /// following the pre-training phase (e.g. T5 adds 100 of these tokens at creation).
    ///
    /// # Parameters
    /// - num_extra_ids (`i64`): number of tokens to append
    fn add_extra_ids(&mut self, num_extra_ids: i64) {
        let mut additional_special_tokens: Vec<String> = Vec::with_capacity(num_extra_ids as usize);
        for extra_id in 0..num_extra_ids {
            additional_special_tokens.push(format!("<extra_id_{extra_id}>"));
        }
        self.add_tokens(
            additional_special_tokens
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<&str>>()
                .as_slice(),
        );
    }

    /// Add arbitrary tokens to the vocabulary.
    ///
    /// These tokens are added to the special token map and are ignored from the tokenization
    /// algorithm chosen (pre-tokenized).
    ///
    /// # Parameters
    /// - tokens (`&[&str]`): list of tokens to add to the vocabulary
    fn add_tokens(&mut self, tokens: &[&str]) {
        let mut tokens_to_add: Vec<&str> = Vec::with_capacity(tokens.len());
        for token in tokens {
            if !self.values().contains_key(*token) {
                tokens_to_add.push(token);
            }
        }
        let mut current_index = self.values().len() as i64;
        for token in tokens_to_add {
            self.values_mut().insert(token.to_string(), current_index);
            self.indices_mut().insert(current_index, token.to_string());
            self.special_values_mut()
                .insert(token.to_string(), current_index);
            self.special_indices_mut()
                .insert(current_index, token.to_string());
            current_index += 1;
        }
    }
}

/// # BaseVocab
/// Base vocabulary with [UNK] unknown token used as a pre-tokenization step for BERT-class tokenizers.
/// Expects a flat text vocabulary when created from file.
#[derive(Debug, Clone)]
pub struct BaseVocab {
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

const DEFAULT_UNK_TOKEN: &str = "[UNK]";

impl Vocab for BaseVocab {
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


    fn from_cache(text: &str) -> Result<BaseVocab, TokenizerError> {
        let values = read_flat_string(text)?;
        let special_token_map = SpecialTokenMap {
            unk_token: DEFAULT_UNK_TOKEN.to_string(),
            pad_token: None,
            bos_token: None,
            sep_token: None,
            cls_token: None,
            eos_token: None,
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
