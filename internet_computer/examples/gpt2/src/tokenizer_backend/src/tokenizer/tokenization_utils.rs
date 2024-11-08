// Copyright 2018 The Open AI Team Authors, The Google AI Language Team Authors
// Copyright 2018 The HuggingFace Inc. team.
// Copyright 2019-2020 Guillaume Becquin
// Copyright 2020 Maarten van Gompel
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::error::TokenizerError;
use crate::tokenizer::base_tokenizer::{TokenIdsWithOffsets, TruncationStrategy};
use crate::tokenizer::constants::BYTES_TO_UNICODE;
use crate::vocab::bpe_vocab::{BpePairRef, BpePairVocab};
use crate::vocab::Vocab;
use crate::{Mask, Offset, OffsetSize, Token, TokenRef};
use regex::Regex;
use std::borrow::BorrowMut;
//use std::char;
//use std::char::REPLACEMENT_CHARACTER;
//use std::cmp::{min, Ordering};
use std::cmp::{min};
use std::collections::{HashMap, HashSet};
use std::sync::RwLock;
//use unicode_normalization::char::decompose_canonical;
//use unicode_normalization_alignments::UnicodeNormalization;

pub type BpeCache = RwLock<HashMap<String, (Vec<String>, Vec<usize>)>>;

///Split a text on special tokens (like BOS/EOS/UNK markers), depending on the vocabulary
pub fn split_on_special_tokens<'a>(token: TokenRef<'a>, vocab: &impl Vocab) -> Vec<TokenRef<'a>> {
    let test_substr = |s: &str| {
        for special_value in vocab.special_values().keys() {
            if s.starts_with(special_value.as_str()) {
                return (
                    special_value.len(),
                    special_value.chars().count(),
                    if vocab.get_unknown_value() == special_value.as_str() {
                        Mask::Unknown
                    } else {
                        Mask::Special
                    },
                );
            }
        }
        (0, 0, Mask::None)
    };
    split_on_substr(token, test_substr, true)
}

///Lowercase
pub fn lowercase(token: &mut Token) {
    let capacity = token.text.capacity();
    let mut lower_cased_string: String = String::with_capacity(capacity);
    let mut character_mapping: Vec<OffsetSize> = Vec::with_capacity(capacity);
    for (character, position) in token.text.chars().zip(token.reference_offsets.iter()) {
        for c in character.to_lowercase() {
            lower_cased_string.push(c);
            character_mapping.push(*position);
        }
    }
    token.text = lower_cased_string;
    token.reference_offsets = character_mapping;
    token.offset.begin = *token.reference_offsets.first().unwrap_or(&(0));
    token.offset.end = *token.reference_offsets.last().unwrap_or(&(0)) + 1;
}

pub fn split_on_regex_with_lookahead<'a>(
    token: TokenRef<'a>,
    pattern_lookahead: &Regex,
    pattern_tokenization: &Regex,
) -> Vec<TokenRef<'a>> {
    if token.mask == Mask::None {
        let mut sub_words: Vec<&str> = vec![];
        let mut splits: Vec<&str> = vec![];

        let mut i: usize = 0;
        let mut end_byte: usize;
        for hit in pattern_lookahead.find_iter(token.text) {
            let mut hit_chars = hit.as_str().chars().rev();
            let start = hit_chars.next().unwrap();
            let sep = hit_chars.next().unwrap();
            end_byte = hit.end() - sep.len_utf8() - start.len_utf8();
            splits.push(&token.text[i..end_byte]);
            i = end_byte;
        }
        splits.push(&token.text[i..]);

        for sub_word in splits {
            for hit in pattern_tokenization.find_iter(sub_word) {
                sub_words.push(hit.as_str());
            }
        }

        let mut output_tokens: Vec<TokenRef> = Vec::with_capacity(sub_words.len());
        let mut begin_char: usize = 0;
        let mut end_char: usize;
        for sub_word in sub_words {
            end_char = begin_char + sub_word.chars().count();
            output_tokens.push(TokenRef {
                text: sub_word,
                offset: Offset::new(
                    token.offset.begin + begin_char as OffsetSize,
                    token.offset.begin + end_char as OffsetSize,
                ),
                reference_offsets: &token.reference_offsets[begin_char..end_char],
                mask: Default::default(),
            });
            begin_char = end_char;
        }

        output_tokens
    } else {
        vec![token]
    }
}

/// Split a token on one or more substrings (given a substring test function)
/// * token: The token to split
/// * test_str: A function that contains the string buffer from the current point forward and
/// returns a 3-tuple with the length of the match in bytes, chars and the mask to set (if the
/// length is zero then there is no match.
/// * add_separators: Add the separating characters to the tokens as well? (bool), separating tokens
/// will be indicated in the returned mask by the value set in `set_mask`, which is returned by the test_substr function
pub fn split_on_substr<'a, F>(
    token: TokenRef<'a>,
    test_substr: F,
    add_separators: bool,
) -> Vec<TokenRef<'a>>
where
    F: Fn(&'a str) -> (usize, usize, Mask),
{
    let mut tokens: Vec<TokenRef<'a>> = Vec::new();
    let mut char_begin: usize = 0;
    let mut bytes_begin: usize = 0;
    let mut char_count: usize = 0;

    if token.mask == Mask::None {
        //don't process a token that already got marked in the mask
        //iterate over all characters, returning the byte position with each
        for (char_idx, (bytes_idx, _)) in token.text.char_indices().enumerate() {
            char_count += 1;
            let (matched_bytes, matched_chars, set_mask): (usize, usize, Mask) =
                test_substr(&token.text[bytes_idx..]);
            if matched_chars > 0 {
                if char_begin < char_idx {
                    //add previous token
                    let trimmed_text =
                        token.text[bytes_begin..bytes_begin + (bytes_idx - bytes_begin)].trim_end();
                    let trimmed_text_len = trimmed_text.chars().count();
                    if trimmed_text_len > 0 {
                        tokens.push(TokenRef {
                            text: trimmed_text,
                            offset: Offset {
                                begin: token.offset.begin + char_begin as OffsetSize,
                                end: token.offset.begin
                                    + (char_begin + trimmed_text_len) as OffsetSize,
                            },
                            reference_offsets: &token.reference_offsets
                                [char_begin..(char_begin + trimmed_text_len)],
                            mask: Mask::None,
                        });
                    }
                }
                if add_separators {
                    //add separator as a singleton token
                    tokens.push(TokenRef {
                        text: &token.text[bytes_idx..bytes_idx + matched_bytes],
                        offset: Offset {
                            begin: token.offset.begin + char_idx as OffsetSize,
                            end: token.offset.begin + (char_idx + matched_chars) as OffsetSize,
                        },
                        reference_offsets: &token.reference_offsets
                            [char_idx..(char_idx + matched_chars)],
                        mask: set_mask,
                    });
                }
                //reset
                char_begin = char_idx + matched_chars;
                bytes_begin = bytes_idx + matched_bytes;
            }
        }
    }
    if bytes_begin < token.text.len() {
        //add last buffered token if there is anything left
        let bytes_idx = token.text.len();
        let text = &token.text[bytes_begin..bytes_begin + (bytes_idx - bytes_begin)];
        if char_count == 0 {
            char_count = text.chars().count();
        }
        tokens.push(TokenRef {
            text,
            offset: Offset {
                begin: token.offset.begin + char_begin as OffsetSize,
                end: token.offset.begin + char_count as OffsetSize,
            },
            reference_offsets: &token.reference_offsets[char_begin..char_count],
            mask: Mask::None,
        });
    }
    tokens
}


/// # Truncates a sequence pair in place to the maximum length.
///
///   * tokens_1: list of tokenized input ids. Can be obtained from a string by chaining the
///       `tokenize` and `convert_tokens_to_ids` methods.
///   * tokens_2: Optional second list of input ids. Can be obtained from a string by chaining the
///       `tokenize` and `convert_tokens_to_ids` methods.
///   * offsets: list of offsets for tokens_1 (must be same length or empty if not used at all)
///   * offsets_2: optional second list of offsets for tokens_2 (must be same length or empty if not used at all)
///   * tokens_2: Optional second list of input ids. Can be obtained from a string by chaining the
///       `tokenize` and `convert_tokens_to_ids` methods.
///   * num_tokens_to_remove
///       number of tokens to remove using the truncation strategy
///   * truncation_strategy: truncation strategy
///       - TruncationStrategy::LongestFirst (default) Iteratively reduce the inputs sequence until the input is under max_length
///           starting from the longest one at each token (when there is a pair of input sequences).
///           Overflowing tokens only contains overflow from the first sequence.
///       - TruncationStrategy::OnlyFirst: Only truncate the first sequence. raise an error if the first sequence is shorter or equal to than num_tokens_to_remove.
///       - TruncationStrategy::OnlySecond: Only truncate the second sequence
///       - TruncationStrategy::DoNotTruncate: Does not truncate (raise an error if the input sequence is longer than max_length)
///   * stride
///       If set to a number along with max_length, the overflowing tokens returned will contain some tokens
///       from the main sequence returned. The value of this argument defines the number of additional tokens.
pub fn truncate_sequences(
    mut token_ids_with_offsets_1: TokenIdsWithOffsets,
    mut token_ids_with_offsets_2: Option<TokenIdsWithOffsets>,
    num_tokens_to_remove: usize,
    truncation_strategy: &TruncationStrategy,
    stride: usize,
) -> Result<
    (
        TokenIdsWithOffsets,
        Option<TokenIdsWithOffsets>,
        Vec<i64>,
        Vec<Option<Offset>>,
    ),
    TokenizerError,
> {
    if num_tokens_to_remove == 0 {
        Ok((
            token_ids_with_offsets_1,
            token_ids_with_offsets_2,
            Vec::new(),
            Vec::new(),
        ))
    } else if let Some(token_ids_with_offsets_2_value) = token_ids_with_offsets_2.borrow_mut() {
        match truncation_strategy {
            TruncationStrategy::LongestFirst => {
                if (token_ids_with_offsets_1.ids.len() + token_ids_with_offsets_2_value.ids.len())
                    >= num_tokens_to_remove
                {
                    let mut overflow_tokens: Vec<i64> =
                        Vec::with_capacity(num_tokens_to_remove + stride);
                    let mut overflow_offsets: Vec<Option<Offset>> =
                        Vec::with_capacity(num_tokens_to_remove + stride);
                    for _ in 0..num_tokens_to_remove {
                        if token_ids_with_offsets_1.ids.len()
                            >= token_ids_with_offsets_2_value.ids.len()
                        {
                            overflow_tokens.insert(0, token_ids_with_offsets_1.ids.pop().unwrap());
                            if !token_ids_with_offsets_1.offsets.is_empty() {
                                overflow_offsets
                                    .insert(0, token_ids_with_offsets_1.offsets.pop().unwrap());
                            }
                            token_ids_with_offsets_1.reference_offsets.pop();
                            if !token_ids_with_offsets_1.masks.is_empty() {
                                token_ids_with_offsets_1.masks.pop();
                            }
                        } else {
                            overflow_tokens
                                .insert(0, token_ids_with_offsets_2_value.ids.pop().unwrap());
                            if !token_ids_with_offsets_2_value.offsets.is_empty() {
                                overflow_offsets.insert(
                                    0,
                                    token_ids_with_offsets_2_value.offsets.pop().unwrap(),
                                );
                            }
                            token_ids_with_offsets_2_value.reference_offsets.pop();
                            if !token_ids_with_offsets_2_value.masks.is_empty() {
                                token_ids_with_offsets_2_value.masks.pop();
                            }
                        }
                    }
                    let window_len = min(token_ids_with_offsets_1.ids.len(), stride);
                    if window_len > 0 {
                        let slice: &[i64] = &token_ids_with_offsets_1.ids
                            [token_ids_with_offsets_1.ids.len() - window_len..];
                        overflow_tokens.splice(0..0, slice.iter().cloned());
                        if !token_ids_with_offsets_1.offsets.is_empty() {
                            let offset_slice: &[Option<Offset>] = &token_ids_with_offsets_1.offsets
                                [token_ids_with_offsets_1.offsets.len() - window_len..];
                            overflow_offsets.splice(0..0, offset_slice.iter().cloned());
                        }
                    }
                    Ok((
                        token_ids_with_offsets_1,
                        token_ids_with_offsets_2,
                        overflow_tokens,
                        overflow_offsets,
                    ))
                } else {
                    Err(TokenizerError::ValueError(
                        "Combined sequence length too short for requested truncation amount".into(),
                    ))
                }
            }
            TruncationStrategy::OnlyFirst => {
                if token_ids_with_offsets_1.ids.len() >= num_tokens_to_remove {
                    let (overflow_tokens, overflow_offsets) = truncate_with_overflow(
                        &mut token_ids_with_offsets_1.ids,
                        token_ids_with_offsets_1.offsets.as_mut(),
                        token_ids_with_offsets_1.reference_offsets.as_mut(),
                        token_ids_with_offsets_1.masks.as_mut(),
                        num_tokens_to_remove,
                        stride,
                    );
                    Ok((
                        token_ids_with_offsets_1,
                        token_ids_with_offsets_2,
                        overflow_tokens,
                        overflow_offsets,
                    ))
                } else {
                    Err(TokenizerError::ValueError(
                        "First sequence too short for first only truncation".into(),
                    ))
                }
            }
            TruncationStrategy::OnlySecond => {
                if token_ids_with_offsets_2_value.ids.len() >= num_tokens_to_remove {
                    let (overflow_tokens, overflow_offsets) = truncate_with_overflow(
                        &mut token_ids_with_offsets_2_value.ids,
                        token_ids_with_offsets_2_value.offsets.as_mut(),
                        token_ids_with_offsets_2_value.reference_offsets.as_mut(),
                        token_ids_with_offsets_2_value.masks.as_mut(),
                        num_tokens_to_remove,
                        stride,
                    );
                    Ok((
                        token_ids_with_offsets_1,
                        token_ids_with_offsets_2,
                        overflow_tokens,
                        overflow_offsets,
                    ))
                } else {
                    Err(TokenizerError::ValueError(
                        "Second sequence too short for second only truncation".into(),
                    ))
                }
            }
            TruncationStrategy::DoNotTruncate => Err(TokenizerError::ValueError(
                "Truncation needed but no truncation requested".into(),
            )),
        }
    } else if token_ids_with_offsets_1.ids.len() >= num_tokens_to_remove {
        match truncation_strategy {
            TruncationStrategy::LongestFirst | TruncationStrategy::OnlyFirst => {
                let (overflow_tokens, overflow_offsets) = truncate_with_overflow(
                    &mut token_ids_with_offsets_1.ids,
                    &mut token_ids_with_offsets_1.offsets,
                    &mut token_ids_with_offsets_1.reference_offsets,
                    &mut token_ids_with_offsets_1.masks,
                    num_tokens_to_remove,
                    stride,
                );
                Ok((
                    token_ids_with_offsets_1,
                    token_ids_with_offsets_2,
                    overflow_tokens,
                    overflow_offsets,
                ))
            }
            TruncationStrategy::OnlySecond => Err(TokenizerError::ValueError(
                "Invalid truncation strategy for single sentence truncation".into(),
            )),
            TruncationStrategy::DoNotTruncate => Err(TokenizerError::ValueError(
                "Truncation needed but no truncation requested".into(),
            )),
        }
    } else {
        Err(TokenizerError::ValueError(
            "First sequence too short for first only truncation".into(),
        ))
    }
}

fn truncate_with_overflow(
    sequence: &mut Vec<i64>,
    offsets: &mut Vec<Option<Offset>>,
    original_positions: &mut Vec<Vec<OffsetSize>>,
    mask: &mut Vec<Mask>,
    num_tokens_to_remove: usize,
    stride: usize,
) -> (Vec<i64>, Vec<Option<Offset>>) {
    if !offsets.is_empty() {
        assert_eq!(sequence.len(), offsets.len());
    }
    if !mask.is_empty() {
        assert_eq!(sequence.len(), mask.len());
    }
    let cutoff = sequence.len() - num_tokens_to_remove;
    let mut overflow_tokens = sequence.split_off(cutoff);
    let mut overflow_offsets = if !offsets.is_empty() {
        offsets.split_off(cutoff)
    } else {
        Vec::new()
    };
    if !mask.is_empty() {
        mask.truncate(cutoff);
        original_positions.truncate(cutoff);
    }
    let window_len = min(sequence.len(), stride);
    if window_len > 0 {
        let slice: &[i64] = &sequence[sequence.len() - window_len..];
        overflow_tokens.splice(0..0, slice.iter().cloned());
        if !offsets.is_empty() {
            let offset_slice: &[Option<Offset>] = &offsets[offsets.len() - window_len..];
            overflow_offsets.splice(0..0, offset_slice.iter().cloned());
        }
    }
    (overflow_tokens, overflow_offsets)
}

pub fn get_pairs(token: &[String]) -> Option<HashSet<BpePairRef>> {
    match token.len() {
        0 | 1 => None,
        _ => {
            let mut output: HashSet<BpePairRef> = HashSet::with_capacity(token.len());
            for idx in 0..token.len() - 1 {
                if let [byte_1, byte_2] = &token[idx..idx + 2] {
                    output.insert(BpePairRef { byte_1, byte_2 });
                }
            }
            Some(output)
        }
    }
}

pub fn group_common_pairs(tokens: Vec<String>, bpe_ranks: &BpePairVocab) -> (Vec<String>, bool) {
    if let Some(pairs) = get_pairs(&tokens) {
        let bigram = pairs
            .iter()
            .min_by_key(|pair| match bpe_ranks.byte_pair_to_id(pair) {
                Some(&rank) => rank,
                None => i64::MAX,
            })
            .unwrap();
        if bpe_ranks.byte_pair_to_id(bigram).is_none() {
            return (tokens, true);
        }
        let mut temp_sub_tokens: Vec<String> = Vec::with_capacity(tokens.len());
        let mut i = 0;

        while i < tokens.len() {
            let j = if let Some(index) = &tokens[i..].iter().position(|r| r == bigram.byte_1) {
                index + i
            } else {
                temp_sub_tokens.extend_from_slice(&tokens[i..]);
                break;
            };
            temp_sub_tokens.extend_from_slice(&tokens[i..j]);
            i = j;
            if (&tokens[i] == bigram.byte_1) & (i < tokens.len() - 1) {
                if &tokens[i + 1] == bigram.byte_2 {
                    let mut combined_bytes =
                        String::with_capacity(bigram.byte_1.len() + bigram.byte_2.len());
                    combined_bytes.push_str(bigram.byte_1.as_str());
                    combined_bytes.push_str(bigram.byte_2.as_str());
                    temp_sub_tokens.push(combined_bytes);
                    i += 2;
                } else {
                    temp_sub_tokens.push(bigram.byte_1.clone());
                    i += 1;
                }
            } else {
                temp_sub_tokens.push(bigram.byte_1.clone());
                i += 1;
            }
        }
        if temp_sub_tokens.len() == 1 {
            return (temp_sub_tokens, true);
        }
        (temp_sub_tokens, false)
    } else {
        (tokens, true)
    }
}


///Default bpe function, as called by Roberta and GPT2
pub fn bpe(token: &str, bpe_ranks: &BpePairVocab) -> (Vec<String>, Vec<usize>) {
    let sub_tokens = token
        .chars()
        .map(|v| v.to_string())
        .collect::<Vec<String>>();

    let mut output = (sub_tokens, false);
    loop {
        output = group_common_pairs(output.0, bpe_ranks);
        if output.1 {
            break;
        }
    }
    let char_counts = output.0.iter().map(|v| v.chars().count()).collect();
    (output.0, char_counts)
}

fn bytes_offsets(text: &str) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(text.len());
    for (char_idx, character) in text.chars().enumerate() {
        for _ in 0..character.len_utf8() {
            offsets.push(char_idx)
        }
    }
    offsets
}

pub fn split_on_bpe_pairs<F>(
    token: TokenRef<'_>,
    bpe_function: F,
    bpe_ranks: &BpePairVocab,
    cache: &BpeCache,
    as_bytes: bool,
) -> Vec<Token>
where
    F: Fn(&str, &BpePairVocab) -> (Vec<String>, Vec<usize>),
{
    let mut tokens: Vec<Token> = Vec::new();
    let text: String;
    let reference_offsets_placeholder: Vec<OffsetSize>;
    let (text, reference_offsets) = if as_bytes {
        reference_offsets_placeholder = bytes_offsets(token.text)
            .iter()
            .map(|&pos| token.reference_offsets[pos])
            .collect();
        text = token
            .text
            .as_bytes()
            .iter()
            .map(|v| BYTES_TO_UNICODE.get(v).unwrap())
            .collect();
        (text.as_str(), reference_offsets_placeholder.as_slice())
    } else {
        (token.text, token.reference_offsets)
    };

    let cached: bool = if let Ok(ref mut cache) = cache.try_read() {
        match cache.get(text) {
            Some((cached_tokens, char_counts)) => {
                let mut start = 0;
                for (idx, (sub_token, &char_count)) in
                    cached_tokens.iter().zip(char_counts.iter()).enumerate()
                {
                    tokens.push(Token {
                        text: sub_token.clone(),
                        offset: Offset {
                            begin: reference_offsets[start],
                            end: reference_offsets[start + char_count - 1] + 1,
                        },
                        reference_offsets: reference_offsets[start..start + char_count].to_vec(),
                        mask: {
                            if cached_tokens.len() > 1 {
                                if idx == 0 {
                                    Mask::Begin
                                } else {
                                    Mask::Continuation
                                }
                            } else {
                                Mask::None
                            }
                        },
                    });
                    start += char_count;
                }
                true
            }
            None => false,
        }
    } else {
        false
    };

    if !cached {
        let (bpe_output, char_counts) = bpe_function(text, bpe_ranks);
        if let Ok(mut cache) = cache.try_write() {
            cache.insert(text.to_owned(), (bpe_output.clone(), char_counts.clone()));
        }
        let mut start = 0;
        for (idx, (sub_token, &char_count)) in bpe_output.iter().zip(char_counts.iter()).enumerate()
        {
            tokens.push(Token {
                text: sub_token.clone(),
                offset: Offset {
                    begin: reference_offsets[start],
                    end: reference_offsets[start + char_count - 1] + 1,
                },
                reference_offsets: reference_offsets[start..start + char_count].to_vec(),
                mask: {
                    if bpe_output.len() > 1 {
                        if idx == 0 {
                            Mask::Begin
                        } else {
                            Mask::Continuation
                        }
                    } else {
                        Mask::None
                    }
                },
            });
            start += char_count;
        }
    }
    tokens
}

pub fn fix_mask(tokens: &mut Vec<Token>) {
    for i in 1..tokens.len() {
        if tokens[i].mask == Mask::Continuation && tokens[i - 1].mask == Mask::None {
            if let Some(token) = tokens.get_mut(i - 1) {
                token.mask = Mask::Begin;
            }
        }
    }
}
