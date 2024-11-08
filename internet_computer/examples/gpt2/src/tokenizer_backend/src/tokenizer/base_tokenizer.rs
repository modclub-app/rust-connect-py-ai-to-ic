
//use std::path::Path;
//use crate::error::TokenizerError;

//use crate::tokenizer::tokenization_utils::{clean_text, lowercase};
//use crate::tokenizer::tokenization_utils::{
//    split_on_punct, split_on_special_tokens, strip_accents, tokenize_cjk_chars, truncate_sequences,
//    whitespace_tokenize,
//};
use crate::tokenizer::tokenization_utils::truncate_sequences;
use crate::vocab::Vocab;
use itertools::Itertools;
//use rayon::prelude::*;
use serde::{Deserialize, Serialize};


/// # Truncation strategy variants
/// Indicates if and how sequence pairs exceeding a given length should be truncated
pub enum TruncationStrategy {
    /// Truncate the longest sequence first
    LongestFirst,
    /// Truncate only the first sequence
    OnlyFirst,
    /// Truncate only the second sequence
    OnlySecond,
    /// Do not truncate the sequences
    DoNotTruncate,
}

/// Crate-wide primitive used to store offset positions
pub type OffsetSize = u32;

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy, Serialize, Deserialize, Eq)]
///Offset information (in unicode points) to relate a token back to its original input string
pub struct Offset {
    pub begin: OffsetSize,
    pub end: OffsetSize,
}

impl Offset {
    /// Create a new offset from a begin and end positions
    pub fn new(begin: OffsetSize, end: OffsetSize) -> Offset {
        Offset { begin, end }
    }

    /// Wrap the offset into an option
    pub fn into_option(self) -> Option<Offset> {
        if self.end > self.begin {
            Some(self)
        } else {
            None
        }
    }
}

/// # Type indication for tokens (e.g. special token, white space, unknown...)
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, PartialEq, PartialOrd, Clone, Copy, Serialize, Deserialize, Eq, Default)]
pub enum Mask {
    /// The token has no particular mask. This is the default situation. It may indicate that further processing can be done on a token.
    #[default]
    None,
    /// The token represents a whitespace (in any shape or form)
    Whitespace,
    /// The token represents punctuation (in any shape or form)
    Punctuation,
    /// The token represents a single Chinese/Japanese/Korean character (including kana and hangul)
    CJK,
    /// The token is a special marker (such as a separator marker, a class marker, etc)
    Special,
    /// The token is the begin in a series of subtokens, the offset refers specifically to the sub-token. Subsequent tokens in this sequence will carry the 'Continuation' mask
    Begin,
    /// The token is the continuation of the previous token, the offset refers specifically to the sub-token. All but the first sub-token in a sequence carry this mask (the first carries 'Begin'). (this is the reverse of Mask::Unfinished)
    Continuation,
    /// The token is the start of a token but not finished yet. All but the last sub-token in the a token sequence carry this mask. This is the reverse of Mask::Continuation.
    Unfinished,
    /// The token is out of vocabulary, it is unknown by the tokenizer and it will decode to unknown. Tokens that can be decoded properly (but may still be out of vocabulary) should not set this.
    Unknown,
}

/// Token abstraction trait to access token fields, irrespective of their form (reference of owned)
pub trait TokenTrait {
    /// Returns the offset of the token with respect to the original string
    fn offset(&self) -> Option<Offset>;
    /// Returns the token mask
    fn mask(&self) -> Mask;
    /// Returns a string representation for the token
    fn as_str(&self) -> &str;
}

#[derive(Debug, PartialEq, Clone, Copy, Eq)]
/// Reference token that references the original text, with a string slice representation
pub struct TokenRef<'a> {
    /// String representation
    pub text: &'a str,
    /// Start and end positions of the token with respect to the original text
    pub offset: Offset,
    /// Sequence of positions with respect to the original text contained in the token.
    /// For example, if the token offset is `start: 4, end: 10`, corresponding reference_offsets are `[4, 5, 6, 7, 8, 9]`
    pub reference_offsets: &'a [OffsetSize],
    /// Mask indicating the type of the token
    pub mask: Mask,
}

impl<'a> TokenRef<'a> {
    /// Creates a new token reference from a text and list of offsets.
    ///
    /// # Parameters
    /// - text (`&str`): text reference
    /// - offsets (`&[OffsetSize]`): reference positions with respect to the original text
    ///
    /// # Example
    /// ```
    /// use gpt2_tokenizer::TokenRef;
    /// let _original_text = "Hello, world";
    /// let text = "world";
    /// let offsets = &[7, 8, 9, 10, 11];
    ///
    /// let token_ref = TokenRef::new(text, offsets);
    /// ```
    pub fn new(text: &'a str, offsets: &'a [OffsetSize]) -> TokenRef<'a> {
        TokenRef {
            text,
            offset: Offset {
                begin: 0,
                end: offsets.len() as OffsetSize,
            },
            reference_offsets: offsets,
            mask: Mask::None,
        }
    }

    /// Converts a token reference to an owned form.
    /// # Example
    /// ```
    /// use gpt2_tokenizer::TokenRef;
    /// let _original_text = "Hello, world";
    /// let text = "world";
    /// let offsets = &[7, 8, 9, 10, 11];
    /// let token_ref = TokenRef::new(text, offsets);
    ///
    /// let owned_token = token_ref.to_owned();
    /// ```
    pub fn to_owned(self) -> Token {
        //not a real implementation of ToOwned because that can't work in the current setup
        Token::from(self)
    }
}

impl<'a> TokenTrait for TokenRef<'a> {
    fn offset(&self) -> Option<Offset> {
        self.offset.into_option()
    }

    fn mask(&self) -> Mask {
        self.mask
    }

    fn as_str(&self) -> &str {
        self.text
    }
}

impl TokenTrait for Token {
    fn offset(&self) -> Option<Offset> {
        self.offset.into_option()
    }

    fn mask(&self) -> Mask {
        self.mask
    }

    fn as_str(&self) -> &str {
        self.text.as_str()
    }
}

impl<'a> From<&'a Token> for TokenRef<'a> {
    fn from(other: &'a Token) -> Self {
        TokenRef {
            text: other.text.as_str(),
            offset: other.offset,
            reference_offsets: &other.reference_offsets,
            mask: other.mask,
        }
    }
}

impl From<&str> for Token {
    fn from(text: &str) -> Self {
        Token::new(text.to_owned())
    }
}

impl<'a> From<TokenRef<'a>> for Token {
    fn from(other: TokenRef<'a>) -> Self {
        Token {
            text: other.text.to_owned(),
            offset: other.offset,
            reference_offsets: other.reference_offsets.to_vec(),
            mask: other.mask,
        }
    }
}

/// # ConsolidatedTokenIterator
///
/// This iterator loops over collections of tokens (implementing `TokenTrait`)
/// and groups all subtokens that belong together (forming a word or something similar).
pub struct ConsolidatedTokenIterator<'a, T>
where
    T: TokenTrait,
{
    pub tokens: &'a [T],
    pub begin: usize,
    pub cursor: usize,
}

impl<'a, T> ConsolidatedTokenIterator<'a, T>
where
    T: TokenTrait,
{
    /// Creates a new `ConsolidatedTokenIterator` from a sequence of `Tokens` or `TokenRefs`
    pub fn new(tokens: &'a [T]) -> Self {
        ConsolidatedTokenIterator {
            tokens,
            begin: 0,
            cursor: 0,
        }
    }
}

impl<'a, T> Iterator for ConsolidatedTokenIterator<'a, T>
where
    T: TokenTrait,
{
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(sub_token) = self.tokens.get(self.cursor) {
                if sub_token.mask() != Mask::Continuation {
                    //return the previous buffer of subtokens (no copies!)
                    if self.cursor > self.begin {
                        let sub_tokens = &self.tokens[self.begin..self.cursor];
                        self.begin = self.cursor;
                        self.cursor += 1;
                        return Some(sub_tokens);
                    }
                }
                self.cursor += 1;
            } else {
                //we are at past the last item, return remaining buffer
                if self.begin < self.cursor {
                    let sub_tokens = &self.tokens[self.begin..self.cursor];
                    self.cursor += 1;
                    self.begin = self.cursor;
                    return Some(sub_tokens);
                } else {
                    //nothing in buffer, we're done
                    return None;
                }
            }
        }
    }
}

/// # ConsolidatableTokens
///
/// This trait can be implemented for collections of tokens (i.e. things that implement `TokenTrait`)
/// and instantiates an iterator to quickly iterate over the tokens in consolidated form, e.g.
/// grouping subtokens into words.
///
/// ```no_run
/// use gpt2_tokenizer::{ConsolidatableTokens, Token};
/// let tokens: Vec<Token> = vec![]; //add some tokens
/// for (wordcount, word_tokens) in tokens.iter_consolidate_tokens().enumerate() {
///     eprintln!("word #{} - {:?}", wordcount + 1, word_tokens);
/// }
/// ```
pub trait ConsolidatableTokens<T>
where
    T: TokenTrait,
{
    /// Creates an iterator from a sequence of `ConsolidatableTokens`.
    fn iter_consolidate_tokens(&self) -> ConsolidatedTokenIterator<T>;
}

impl ConsolidatableTokens<Token> for Vec<Token> {
    fn iter_consolidate_tokens(&self) -> ConsolidatedTokenIterator<Token> {
        ConsolidatedTokenIterator::new(self)
    }
}

impl<'a> ConsolidatableTokens<TokenRef<'a>> for Vec<TokenRef<'a>> {
    fn iter_consolidate_tokens(&self) -> ConsolidatedTokenIterator<TokenRef<'a>> {
        ConsolidatedTokenIterator::new(self)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
/// Owned token that references the original text but stores its own string representation.
pub struct Token {
    /// String representation
    pub text: String,
    /// Start and end positions of the token with respect to the original text
    pub offset: Offset,
    /// Sequence of positions with respect to the original text contained in the token.
    /// For example, if the token offset is `start: 4, end: 10`, corresponding reference_offsets are `[4, 5, 6, 7, 8, 9]`
    pub reference_offsets: Vec<OffsetSize>,
    /// Mask indicating the type of the token
    pub mask: Mask,
}

impl Token {
    /// Creates a new owned token from a `String`.
    ///
    /// # Parameters
    /// - text (`String`): text reference
    ///
    /// # Example
    /// ```
    /// use gpt2_tokenizer::Token;
    /// let text = "world".to_string();
    /// let token = Token::new(text);
    /// ```
    pub fn new(text: String) -> Token {
        let text_size: OffsetSize = text.chars().count() as OffsetSize;
        Token {
            text,
            offset: Offset {
                begin: 0,
                end: text_size,
            },
            reference_offsets: (0..text_size).collect(),
            mask: Mask::None,
        }
    }

    /// Converts an owned token to a reference form
    ///
    /// # Example
    /// ```
    /// use gpt2_tokenizer::Token;
    /// let text = "world".to_string();
    /// let token = Token::new(text);
    ///
    /// let token_ref = token.as_ref();
    /// ```
    pub fn as_ref(&self) -> TokenRef {
        //not a real implementation of AsRef because we do something slightly different
        TokenRef::from(self)
    }
}

/// # Tokenized Input, ready for processing in language models
/// This represents the final output of the encoding process (tokenized sentence with encoded values)
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone)]
pub struct TokenizedInput {
    /// Vector of token IDs
    pub token_ids: Vec<i64>,

    /// Vector segments ids (for example for BERT segments are separated with a [SEP] marker, each incrementing the segment ID).
    /// This vector has the same length as token_ids.
    pub segment_ids: Vec<i8>,

    /// Flags tokens as special tokens (1) or not (0). This vector has the same length as token_ids.
    pub special_tokens_mask: Vec<i8>,

    /// Vector containing overflowing tokens, populated following a truncation step
    pub overflowing_tokens: Vec<i64>,

    /// Number of overflowing tokens following a truncation step. this equals the length `overflowing_tokens`
    pub num_truncated_tokens: usize,

    /// Offset information (as start and end positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub token_offsets: Vec<Option<Offset>>,

    /// Offset information (as a sequence of positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub reference_offsets: Vec<Vec<OffsetSize>>,

    /// Masks tokens providing information on the type of tokens. This vector has the same length as token_ids.
    pub mask: Vec<Mask>,
}

/// # Encoded input with special tokens
/// Intermediate tokenization steps before truncation to a maximum length, after encoding and addition of special tokens
#[derive(Debug, Clone)]
pub struct TokenIdsWithSpecialTokens {
    /// Vector of token IDs
    pub token_ids: Vec<i64>,

    /// Vector segments ids (for example for BERT segments are separated with a [SEP] marker, each incrementing the segment ID).
    /// This vector has the same length as token_ids.
    pub segment_ids: Vec<i8>,

    /// Flags tokens as special tokens (1) or not (0). This vector has the same length as token_ids.
    pub special_tokens_mask: Vec<i8>,

    /// Offset information (as start and end positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub token_offsets: Vec<Option<Offset>>,

    /// Offset information (as a sequence of positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub reference_offsets: Vec<Vec<OffsetSize>>,

    /// Masks tokens providing information on the type of tokens. This vector has the same length as token_ids.
    pub mask: Vec<Mask>,
}

/// # Tokenized sequence
/// Intermediate tokenization steps before encoding, addition of special tokens and truncation
#[derive(Debug, Clone)]
pub struct TokensWithOffsets {
    /// Vector of token strings
    pub tokens: Vec<String>,

    /// Offset information (as start and end positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub offsets: Vec<Option<Offset>>,

    /// Offset information (as a sequence of positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub reference_offsets: Vec<Vec<OffsetSize>>,

    /// Masks tokens providing information on the type of tokens. This vector has the same length as token_ids.
    pub masks: Vec<Mask>,
}

/// # Encoded sequence
/// Intermediate tokenization steps before addition of special tokens, after encoding
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenIdsWithOffsets {
    /// Vector of token IDs
    pub ids: Vec<i64>,

    /// Offset information (as start and end positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub offsets: Vec<Option<Offset>>,

    /// Offset information (as a sequence of positions) in relation to the original text. Tokens that can not be related to the
    /// original source are registered as None.
    pub reference_offsets: Vec<Vec<OffsetSize>>,

    /// Masks tokens providing information on the type of tokens. This vector has the same length as token_ids.
    pub masks: Vec<Mask>,
}

/// # Base trait for tokenizers
pub trait Tokenizer<T: Vocab> {
    /// returns a reference to the tokenizer vocabulary
    fn vocab(&self) -> &T;

    /// returns a mutable reference to the tokenizer vocabulary
    fn vocab_mut(&mut self) -> &mut T;

    /// Tokenize a string, returns a vector of tokens as strings.
    /// Use `tokenize_with_offsets` or `tokenize_to_tokens` to return offset information.
    ///
    /// # Parameters
    /// - text : text (string-like) to tokenize
    ///
    /// # Returns
    /// `Vec<String>` containing the tokens string representation
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let text = "Hello, world!";
    /// let tokens = tokenizer.tokenize(text);
    /// ```
    fn tokenize(&self, text: &str) -> Vec<String> {
        self.tokenize_with_offsets(text).tokens
    }

    /// Tokenize a string, returning tokens with offset information
    ///
    /// # Parameters
    /// - text : text (string-like) to tokenize
    ///
    /// # Returns
    /// `TokensWithOffsets` with the tokens and their offset information
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let text = "Hello, world!";
    /// let tokens = tokenizer.tokenize_with_offsets(text);
    /// ```

    fn tokenize_with_offsets(&self, text: &str) -> TokensWithOffsets {
        if text.trim().is_empty() {
            return TokensWithOffsets {
                tokens: vec![],
                offsets: vec![],
                reference_offsets: vec![],
                masks: vec![],
            };
        }
        let initial_offsets = (0..text.chars().count() as OffsetSize).collect::<Vec<OffsetSize>>();
        let initial_token: TokenRef<'_> = TokenRef::new(text, &initial_offsets);
        let tokens = self.tokenize_to_tokens(initial_token);
        let length = tokens.len();
        let mut texts = Vec::with_capacity(length);
        let mut offsets = Vec::with_capacity(length);
        let mut original_positions = Vec::with_capacity(length);
        let mut masks = Vec::with_capacity(length);

        for token in tokens {
            texts.push(token.text);
            offsets.push(if !token.reference_offsets.is_empty() {
                Some(Offset {
                    begin: *token.reference_offsets.first().unwrap(),
                    end: *token.reference_offsets.last().unwrap() + 1,
                })
            } else {
                None
            });
            original_positions.push(token.reference_offsets);
            masks.push(token.mask);
        }
        TokensWithOffsets {
            tokens: texts,
            offsets,
            reference_offsets: original_positions,
            masks,
        }
    }

    /// Tokenize a TokenRef, returning a sequence of tokens
    ///
    /// # Parameters
    /// - text (`TokenRef`): TokenRef to tokenize (this is especially useful for nested tokenization,
    /// where a tokenizer is called on the ouput of a pre-tokenizer, such as BERT).
    ///
    /// # Returns
    /// `Vec<Token>` tokenization of the original `TokenRef`
    ///
    /// # Example
    ///
    /// ```no_run
    /// use itertools::Itertools;
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// use gpt2_tokenizer::{OffsetSize, TokenRef};
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let text = "Hello, world!";
    /// let offsets = (0..text.len() as OffsetSize).collect_vec();
    /// let text = TokenRef::new(text, &offsets);
    /// let tokens = tokenizer.tokenize_to_tokens(text);
    /// ```
    fn tokenize_to_tokens(&self, text: TokenRef) -> Vec<Token>;

    /// Tokenize a list of strings, returning tokens with offset information
    ///
    /// # Parameters
    /// - text_list: list of strings to tokenize
    ///
    /// # Returns
    /// `Vec<Vec<String>>` with the token strings representation
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let texts = ["Hello, world!", "Second sentence"];
    /// let tokens = tokenizer.tokenize_list(&texts);
    /// ```
    fn tokenize_list<S>(&self, text_list: &[S]) -> Vec<Vec<String>>
    where
        S: AsRef<str>,
    {
        text_list
            .as_ref()
            .iter()
            .map(|text| self.tokenize(text.as_ref()))
            .collect()
    }

    /// Tokenize a list of strings, where each corresponds to for example a sentence, returns a
    /// vector of TokensWithOffsets containing the tokens and their offset information. This calls
    /// `tokenize_with_offsets` on the list provided.
    ///
    /// # Parameters
    /// - text_list: list of strings to tokenize
    ///
    /// # Returns
    /// `Vec<TokensWithOffsets>` with the token strings representation and offsets
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let text = ["Hello, world!", "Second sentence"];
    /// let tokens = tokenizer.tokenize_list_with_offsets(&text);
    /// ```
    fn tokenize_list_with_offsets<S>(&self, text_list: &[S]) -> Vec<TokensWithOffsets>
    where
        S: AsRef<str>,
    {
        text_list
            .as_ref()
            .iter()
            .map(|text| self.tokenize_with_offsets(text.as_ref()))
            .collect()
    }

    /// Convert a slice of string-like to a vector ot token indices
    ///
    /// # Parameters
    /// - tokens: list of token string-like to convert to ids
    ///
    /// # Returns
    /// `Vec<i64>` with the token indices
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let tokens = ["Hello", ",", "world", "!"];
    /// let token_ids = tokenizer.convert_tokens_to_ids(&tokens);
    /// ```
    fn convert_tokens_to_ids<S>(&self, tokens: &[S]) -> Vec<i64>
    where
        S: AsRef<str>,
    {
        tokens
            .as_ref()
            .iter()
            .map(|v| self.vocab().token_to_id(v.as_ref()))
            .collect()
    }

    /// Encode a string-like (tokenization followed by encoding)
    ///
    /// # Parameters
    /// - text_1: input text (string-like) to encode
    /// - text_2: optional additional input text (string-like) to encode. When provided, both texts are
    /// combined into a single encoding by using the `build_input_with_special_tokens` method.
    /// - max_len (`usize`): maximum combined sequence length. If the combined encoding would exceed this
    /// max_len, the encoding is truncated following the `TruncationStrategy` provided.
    /// - truncation_strategy (`&TruncationStrategy`): strategy to follow for the truncation, if required
    /// - stride (`usize`): amount of tokens to shift the input by if truncation is required
    /// (allowing for the generation of overlapping sequences with overflowing tokens)
    ///
    /// # Returns
    /// `TokenizedInput` containing the encoding output (token indices, token types, segment ids,
    /// ovrflowing tokens and special token mask)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let text_1 = "Hello, world!";
    /// let text_2 = "How is it going?";
    /// let encoded_input = tokenizer.encode(
    ///     text_1,
    ///     Some(text_2),
    ///     5,
    ///     &TruncationStrategy::LongestFirst,
    ///     2,
    /// );
    /// ```
    fn encode(
        &self,
        text_1: &str,
        text_2: Option<&str>,
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> TokenizedInput {
        let tokens = self.tokenize_with_offsets(text_1);
        let token_ids_1 = self.convert_tokens_to_ids(&tokens.tokens);
        let len_1 = token_ids_1.len();
        let token_ids_with_offsets_1 = TokenIdsWithOffsets {
            ids: token_ids_1,
            offsets: tokens.offsets,
            reference_offsets: tokens.reference_offsets,
            masks: tokens.masks,
        };
        let (token_ids_with_offsets_2, len_2) = {
            if let Some(text) = text_2 {
                let tokens_2 = self.tokenize_with_offsets(text);
                let token_ids_2: Vec<i64> = self.convert_tokens_to_ids(&tokens_2.tokens);
                let len_2 = token_ids_2.len();
                (
                    Some(TokenIdsWithOffsets {
                        ids: token_ids_2,
                        offsets: tokens_2.offsets,
                        reference_offsets: tokens_2.reference_offsets,
                        masks: tokens_2.masks,
                    }),
                    len_2,
                )
            } else {
                (None, 0)
            }
        };
        let additional_tokens = self.build_input_with_special_tokens(
            TokenIdsWithOffsets {
                ids: vec![],
                offsets: vec![],
                reference_offsets: vec![],
                masks: vec![],
            },
            if token_ids_with_offsets_2.is_some() {
                Some(TokenIdsWithOffsets {
                    ids: vec![],
                    offsets: vec![],
                    reference_offsets: vec![],
                    masks: vec![],
                })
            } else {
                None
            },
        );
        let total_len = len_1 + len_2 + additional_tokens.token_ids.len();
        let num_truncated_tokens = if total_len > max_len {
            total_len - max_len
        } else {
            0
        };
        let (
            token_ids_with_offsets_1,
            token_ids_with_offsets_2,
            overflowing_tokens,
            _overflowing_offsets,
        ) = truncate_sequences(
            token_ids_with_offsets_1,
            token_ids_with_offsets_2,
            num_truncated_tokens,
            truncation_strategy,
            stride,
        )
        .unwrap();

        let merged_tokenized_input = self
            .build_input_with_special_tokens(token_ids_with_offsets_1, token_ids_with_offsets_2);

        TokenizedInput {
            token_ids: merged_tokenized_input.token_ids,
            segment_ids: merged_tokenized_input.segment_ids,
            special_tokens_mask: merged_tokenized_input.special_tokens_mask,
            overflowing_tokens,
            num_truncated_tokens,
            token_offsets: merged_tokenized_input.token_offsets,
            reference_offsets: merged_tokenized_input.reference_offsets,
            mask: merged_tokenized_input.mask,
        }
    }

    /// Encode a sequence of string-like texts (tokenization followed by encoding). Not that in contrast
    /// with `encode` optional second text, each text provided is encoded independently.
    ///
    /// # Parameters
    /// - text_list: sequence of input text (`&str`) to encode
    /// combined into a single encoding by using the `build_input_with_special_tokens` method.
    /// - max_len (`usize`): maximum combined sequence length. If the combined encoding would exceed this
    /// max_len, the encoding is truncated following the `TruncationStrategy` provided.
    /// - truncation_strategy (`&TruncationStrategy`): strategy to follow for the truncation, if required
    /// - stride (`usize`): amount of tokens to shift the input by if truncation is required
    /// (allowing for the generation of overlapping sequences with overflowing tokens)
    ///
    /// # Returns
    /// `Vec<TokenizedInput>` containing the encoding output (token indices, token types, segment ids,
    /// ovrflowing tokens and special token mask) for each provided text
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let text_1 = "Hello, world!";
    /// let text_2 = "How is it going?";
    /// let text_3 = "Very well thank you.";
    /// let encoded_input = tokenizer.encode_list(
    ///     &[text_1, text_2, text_3],
    ///     5,
    ///     &TruncationStrategy::LongestFirst,
    ///     2,
    /// );
    /// ```
    fn encode_list<S>(
        &self,
        text_list: &[S],
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput>
    where
        S: AsRef<str>,
    {
        text_list
            .as_ref()
            .iter()
            .map(|text| self.encode(text.as_ref(), None, max_len, truncation_strategy, stride))
            .collect()
    }

    /// Encode a sequence of string-like text pairs (tokenization followed by encoding). This combines
    /// with `encode` with the list processing of `encode_list`.
    ///
    /// # Parameters
    /// - text_list: sequence of input text (`&str`) to encode
    /// combined into a single encoding by using the `build_input_with_special_tokens` method.
    /// - max_len (`usize`): maximum combined sequence length. If the combined encoding would exceed this
    /// max_len, the encoding is truncated following the `TruncationStrategy` provided.
    /// - truncation_strategy (`&TruncationStrategy`): strategy to follow for the truncation, if required
    /// - stride (`usize`): amount of tokens to shift the input by if truncation is required
    /// (allowing for the generation of overlapping sequences with overflowing tokens)
    ///
    /// # Returns
    /// `Vec<TokenizedInput>` containing the encoding output (token indices, token types, segment ids,
    /// ovrflowing tokens and special token mask) for each provided text
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let text_1 = "Hello, world!";
    /// let text_2 = "This is a second sentence";
    /// let text_3 = "Very well thank you.";
    /// let text_4 = "This is another second sentence.";
    /// let encoded_input = tokenizer.encode_pair_list(
    ///     &[(text_1, text_2), (text_3, text_4)],
    ///     5,
    ///     &TruncationStrategy::LongestFirst,
    ///     2,
    /// );
    /// ```
    fn encode_pair_list<S>(
        &self,
        text_list: &[(S, S)],
        max_len: usize,
        truncation_strategy: &TruncationStrategy,
        stride: usize,
    ) -> Vec<TokenizedInput>
    where
        S: AsRef<str>,
    {
        text_list
            .as_ref()
            .iter()
            .map(|text| {
                self.encode(
                    text.0.as_ref(),
                    Some(text.1.as_ref()),
                    max_len,
                    truncation_strategy,
                    stride,
                )
            })
            .collect()
    }

    /// Decode a sequence of token indices to a sequence of Strings, optionally skipping special indices
    ///
    /// # Parameters
    /// - token_ids (`Vec<i64>`): tokens to decode
    /// - skip_special_tokens (`bool`): flag indicating if special tokens should be included in the output
    ///
    /// # Returns
    /// `Vec<String>` decoded token indices
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let tokens_ids = vec![0, 1, 2, 42];
    /// let tokens = tokenizer.decode_to_vec(&tokens_ids, false);
    /// ```
    fn decode_to_vec(&self, token_ids: &[i64], skip_special_tokens: bool) -> Vec<String> {
        let tokens: Vec<String> = if skip_special_tokens {
            token_ids
                .iter()
                .filter(|id| !self.vocab().special_indices().contains_key(id))
                .map(|id| self.vocab().id_to_token(id))
                .collect_vec()
        } else {
            token_ids
                .iter()
                .map(|id| self.vocab().id_to_token(id))
                .collect_vec()
        };
        tokens
    }

    /// Converts a sequence of ids (integer) into a string, using the tokenizer and vocabulary
    /// with options to remove special tokens and clean up tokenization spaces.
    ///
    /// # Arguments
    /// - token_ids: list of tokenized input ids. Can be obtained using the `encode` or `encode_plus` methods.
    /// - skip_special_tokens: if set to True, will replace special tokens.
    /// - clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
    ///
    /// # Returns
    /// - `String`: decoded sentence
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let skip_special_tokens = true;
    /// let clean_up_tokenization_spaces = true;
    /// let tokens = vec![0, 1, 2, 42];
    /// let decoded = tokenizer.decode(&tokens, skip_special_tokens, clean_up_tokenization_spaces);
    /// ```
    fn decode(
        &self,
        token_ids: &[i64],
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> String {
        let tokens = self.decode_to_vec(token_ids, skip_special_tokens);
        let decoded_string = self.convert_tokens_to_string(tokens);
        if clean_up_tokenization_spaces {
            self.clean_up_tokenization(decoded_string)
        } else {
            decoded_string
        }
    }

    /// Converts a sequence of strings into a single string. This will clean-up artifacts from tokenization
    /// (for example `sub ##word`) and generate a single output string
    ///
    /// # Arguments
    /// - tokens: list of tokens to concatenate.
    ///
    /// # Returns
    /// - `String`: concatenated sentence string
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let skip_special_tokens = true;
    /// let clean_up_tokenization_spaces = true;
    /// let tokens = vec![
    ///     "Hello".to_string(),
    ///     ",".to_string(),
    ///     "World".to_string(),
    ///     "!".to_string(),
    /// ];
    /// let decoded = tokenizer.convert_tokens_to_string(tokens);
    /// ```
    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        tokens.join(" ")
    }

    /// Cleans-up tokenization artifacts (for example whitespace before punctuation)
    ///
    /// # Arguments
    /// - input_string (`String`): input string to clean up
    ///
    /// # Returns
    /// - `String`: clean-up string
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let skip_special_tokens = true;
    /// let clean_up_tokenization_spaces = true;
    /// let input_string = "Hello . Do n't pay attention to the punctuation .".to_string();
    /// let cleaned_string = tokenizer.clean_up_tokenization(input_string);
    /// ```
    fn clean_up_tokenization(&self, input_string: String) -> String {
        input_string
            .replace(" .", ".")
            .replace(" !", "!")
            .replace(" ?", "?")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" do not", " don't")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
    }

    /// Converts a list of sequence of ids (integer) into a string, using the tokenizer and vocabulary
    /// with options to remove special tokens and clean up tokenization spaces. This calls `decode`
    /// for each provided sequence of ids
    ///
    /// # Arguments
    /// - token_ids: list of list of tokenized input ids. Can be obtained using the `encode` or `encode_plus` methods.
    /// - skip_special_tokens: if set to True, will replace special tokens.
    /// - clean_up_tokenization_spaces: if set to True, will clean up the tokenization spaces.
    ///
    /// # Returns
    /// - `String`: decoded sentence
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let skip_special_tokens = true;
    /// let clean_up_tokenization_spaces = true;
    /// let token_ids_list = vec![vec![0, 1, 2, 42], vec![99, 3]];
    /// let decoded_list = tokenizer.decode_list(
    ///     &token_ids_list,
    ///     skip_special_tokens,
    ///     clean_up_tokenization_spaces,
    /// );
    /// ```
    fn decode_list(
        &self,
        token_ids_list: &[Vec<i64>],
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> Vec<String> {
        token_ids_list
            .iter()
            .map(|token_ids| {
                self.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
            })
            .collect()
    }

    /// Build model inputs from a sequence or a pair of sequence for sequence classification tasks
    /// by concatenating and adding special tokens.
    ///
    /// For example, a RoBERTa sequence has the following format:
    /// - single sequence: <s> X </s>
    /// - pair of sequences: <s> A </s></s> B </s>
    ///
    /// # Parameters
    /// - tokens_ids_with_offsets_1 (`TokenIdsWithOffsets`): first sequence
    /// - tokens_ids_with_offsets_2 (`TokenIdsWithOffsets`): (optional) second sequence
    ///
    /// # Returns
    /// - `TokenIdsWithSpecialTokens` containing a concatenation of both sequences with added special tokens
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{BaseTokenizer, Tokenizer, TruncationStrategy};
    /// use gpt2_tokenizer::vocab::BaseVocab;
    /// use gpt2_tokenizer::TokenIdsWithOffsets;
    /// let strip_accents = false;
    /// let lower_case = false;
    /// let tokenizer: BaseTokenizer<BaseVocab> =
    ///     BaseTokenizer::from_file("path/to/vocab/file", lower_case, strip_accents).unwrap();
    ///
    /// let skip_special_tokens = true;
    /// let clean_up_tokenization_spaces = true;
    /// let first_sequence = "Hello, world";
    /// let second_sequence = "This is the second sentence";
    ///
    /// let first_tokens = tokenizer.tokenize_with_offsets(first_sequence);
    /// let first_ids = tokenizer.convert_tokens_to_ids(&first_tokens.tokens);
    /// let first_input = TokenIdsWithOffsets {
    ///     ids: first_ids,
    ///     offsets: first_tokens.offsets,
    ///     reference_offsets: first_tokens.reference_offsets,
    ///     masks: first_tokens.masks,
    /// };
    ///
    /// let second_tokens = tokenizer.tokenize_with_offsets(second_sequence);
    /// let second_ids = tokenizer.convert_tokens_to_ids(&second_tokens.tokens);
    /// let second_input = TokenIdsWithOffsets {
    ///     ids: second_ids,
    ///     offsets: second_tokens.offsets,
    ///     reference_offsets: second_tokens.reference_offsets,
    ///     masks: second_tokens.masks,
    /// };
    ///
    /// let combined_with_special_tokens =
    ///     tokenizer.build_input_with_special_tokens(first_input, Some(second_input));
    /// ```
    fn build_input_with_special_tokens(
        &self,
        mut tokens_ids_with_offsets_1: TokenIdsWithOffsets,
        tokens_ids_with_offsets_2: Option<TokenIdsWithOffsets>,
    ) -> TokenIdsWithSpecialTokens {
        let mut token_segment_ids: Vec<i8> = vec![0; tokens_ids_with_offsets_1.ids.len()];
        let mut special_tokens_mask: Vec<i8> = vec![0; tokens_ids_with_offsets_1.ids.len()];
        if let Some(tokens_ids_with_offsets_2_value) = tokens_ids_with_offsets_2 {
            let length = tokens_ids_with_offsets_2_value.ids.len();
            token_segment_ids.extend(vec![1; length]);
            special_tokens_mask.extend(vec![0; length]);
            tokens_ids_with_offsets_1
                .ids
                .extend(tokens_ids_with_offsets_2_value.ids);
            tokens_ids_with_offsets_1
                .offsets
                .extend(tokens_ids_with_offsets_2_value.offsets);
            tokens_ids_with_offsets_1
                .reference_offsets
                .extend(tokens_ids_with_offsets_2_value.reference_offsets);
            tokens_ids_with_offsets_1
                .masks
                .extend(tokens_ids_with_offsets_2_value.masks);
        };

        TokenIdsWithSpecialTokens {
            token_ids: tokens_ids_with_offsets_1.ids,
            segment_ids: token_segment_ids,
            special_tokens_mask,
            token_offsets: tokens_ids_with_offsets_1.offsets,
            reference_offsets: tokens_ids_with_offsets_1.reference_offsets,
            mask: tokens_ids_with_offsets_1.masks,
        }
    }

    /// Add arbitrary tokens to the vocabulary.
    ///
    /// These tokens are added to the special token map and are ignored from the tokenization
    /// algorithm chosen (e.g.
    ///
    /// # Parameters
    /// - tokens (`&[&str]`): list of tokens to add to the vocabulary
    fn add_tokens(&mut self, tokens: &[&str]) {
        self.vocab_mut().add_tokens(tokens);
    }

    /// Add arbitrary tokens to the vocabulary.
    ///
    /// These tokens are added to the special token map and are ignored from the tokenization
    /// algorithm chosen (e.g.
    ///
    /// # Parameters
    /// - num_extra_ids (`i64`): number of tokens to append
    fn add_extra_ids(&mut self, num_extra_ids: i64) {
        self.vocab_mut().add_extra_ids(num_extra_ids);
    }
}
