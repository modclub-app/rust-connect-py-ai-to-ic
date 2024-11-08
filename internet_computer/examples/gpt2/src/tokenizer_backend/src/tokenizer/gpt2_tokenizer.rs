
//use crate::error::TokenizerError;
use crate::tokenizer::constants::UNICODE_TO_BYTES;
use crate::tokenizer::tokenization_utils::{
    bpe, fix_mask, split_on_bpe_pairs, split_on_regex_with_lookahead, split_on_special_tokens,
};
use crate::tokenizer::tokenization_utils::{lowercase, BpeCache};
use crate::tokenizer::{Tokenizer};
use crate::vocab::bpe_vocab::BpePairVocab;
//use crate::vocab::{Gpt2Vocab, Vocab};
use crate::vocab::Gpt2Vocab;
use crate::{Mask, Token, TokenRef};
use itertools::Itertools;
use regex::Regex;
use std::collections::HashMap;
use std::iter::Iterator;
//use std::path::Path;
use std::sync::RwLock;

/// # GPT2 tokenizer
/// GPT2 tokenizer performing:
/// - splitting on special characters
/// - whitespace splitting
/// - (optional) lower casing
/// - BPE tokenization
pub struct Gpt2Tokenizer {
    vocab: Gpt2Vocab,
    bpe_ranks: BpePairVocab,
    cache: BpeCache,
    pattern_lookahead: Regex,
    pattern_tokenization: Regex,
    lower_case: bool,
}

impl Gpt2Tokenizer {

    /// Create a new instance of a `Gpt2Tokenizer` from an existing vocabulary and merges
    ///
    /// # Parameters
    /// - vocab (`Gpt2Vocab`): GPT-like vocabulary
    /// - merges (`BpePairVocab`): BPE pairs vocabulary
    /// - lower_case (`bool`): flag indicating if the text should be lower-cased as part of the tokenization
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gpt2_tokenizer::tokenizer::{Gpt2Tokenizer, Tokenizer};
    /// use gpt2_tokenizer::vocab::{BpePairVocab, Gpt2Vocab, Vocab};
    /// let lower_case = false;
    /// let vocab = Gpt2Vocab::from_file("path/to/vocab/file").unwrap();
    /// let merges = BpePairVocab::from_file("path/to/merges/file").unwrap();
    ///
    /// let tokenizer = Gpt2Tokenizer::from_existing_vocab_and_merges(vocab, merges, lower_case);
    /// ```
    pub fn from_existing_vocab_and_merges(
        vocab: Gpt2Vocab,
        merges: BpePairVocab,
        lower_case: bool,
    ) -> Gpt2Tokenizer {
        let cache = RwLock::new(HashMap::new());
        let pattern_lookahead = Regex::new(r"\s+\S").unwrap();
        let pattern_tokenization =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .unwrap();
        Gpt2Tokenizer {
            vocab,
            bpe_ranks: merges,
            cache,
            pattern_lookahead,
            pattern_tokenization,
            lower_case,
        }
    }
}

impl Tokenizer<Gpt2Vocab> for Gpt2Tokenizer {
    fn vocab(&self) -> &Gpt2Vocab {
        &self.vocab
    }
    fn vocab_mut(&mut self) -> &mut Gpt2Vocab {
        &mut self.vocab
    }

    fn tokenize_to_tokens(&self, initial_token: TokenRef) -> Vec<Token> {
        let mut tokens = split_on_special_tokens(initial_token, &self.vocab)
            .into_iter()
            .map(|token| token.to_owned())
            .collect::<Vec<Token>>();

        let mut sub_tokens = Vec::new();
        for token in tokens.iter_mut() {
            if token.mask != Mask::Special && token.mask != Mask::Unknown {
                if self.lower_case {
                    lowercase(token);
                }
                for token in split_on_regex_with_lookahead(
                    token.as_ref(),
                    &self.pattern_lookahead,
                    &self.pattern_tokenization,
                ) {
                    sub_tokens.extend(split_on_bpe_pairs(
                        token,
                        bpe,
                        &self.bpe_ranks,
                        &self.cache,
                        true,
                    ));
                }
            } else {
                sub_tokens.push(token.clone());
            }
        }

        fix_mask(&mut sub_tokens);
        sub_tokens
    }

    fn convert_tokens_to_string(&self, tokens: Vec<String>) -> String {
        let tokens = tokens
            .iter()
            .join("")
            .replace(" ##", "")
            .trim()
            .chars()
            .map(|character| *UNICODE_TO_BYTES.get(&character).unwrap())
            .collect::<Vec<u8>>();
        String::from_utf8_lossy(tokens.as_slice()).to_string()
    }
}
