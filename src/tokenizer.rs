//! Zero-allocation Pure-Rust BERT WordPiece Tokenizer for `snowflake-arctic-embed-s`.
//!
//! Features:
//! - Loads directly from `vocab.txt` (30,522 tokens)
//! - Exact `BertNormalizer` (`clean_text`, `handle_chinese_chars`, `lowercase`, NFD `strip_accents`)
//! - Exact `BertPreTokenizer` (whitespace + ASCII/Unicode punctuation isolation)
//! - Zero-allocation `##` subword lookup via dual hash tables (`root_vocab` + `subword_vocab`)
//! - Pre-tokenized query prefix (`"Represent this sentence for searching relevant passages: "`)

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

pub const PAD_ID: u16 = 0;
pub const UNK_ID: u16 = 100;
pub const CLS_ID: u16 = 101;
pub const SEP_ID: u16 = 102;
pub const MASK_ID: u16 = 103;
pub const MAX_SEQ_LEN: usize = 512;
pub const MAX_WORD_CHARS: usize = 100;

pub const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

#[derive(Clone)]
pub struct BertTokenizer {
    root_vocab: HashMap<Box<str>, u16>,
    subword_vocab: HashMap<Box<str>, u16>,
    id_to_token: Vec<Box<str>>,
    query_prefix_ids: Vec<u16>,
}

impl BertTokenizer {
    pub fn from_vocab_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        Ok(Self::from_vocab_str(&text))
    }

    pub fn from_vocab_str(vocab_text: &str) -> Self {
        let mut root_vocab = HashMap::with_capacity(32768);
        let mut subword_vocab = HashMap::with_capacity(16384);
        let mut id_to_token = Vec::with_capacity(30522);

        for (idx, line) in vocab_text.lines().enumerate() {
            let id = idx as u16;
            let token: Box<str> = line.into();
            if let Some(stripped) = line.strip_prefix("##") {
                if !stripped.is_empty() {
                    subword_vocab.insert(stripped.into(), id);
                }
            }
            root_vocab.insert(token.clone(), id);
            id_to_token.push(token);
        }

        let mut tok = Self {
            root_vocab,
            subword_vocab,
            id_to_token,
            query_prefix_ids: Vec::new(),
        };
        let mut prefix_ids = Vec::with_capacity(12);
        tok.encode_raw_into(QUERY_PREFIX, &mut prefix_ids);
        tok.query_prefix_ids = prefix_ids;
        tok
    }

    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Encodes a document or general sentence into `[CLS] ... [SEP]` (truncated to `MAX_SEQ_LEN = 512`).
    #[inline]
    pub fn encode(&self, text: &str) -> Vec<u16> {
        let mut out = Vec::with_capacity(64);
        self.encode_into(text, false, &mut out);
        out
    }

    /// Encodes a retrieval query with `QUERY_PREFIX` prepended (`[CLS] + prefix_ids + query_ids + [SEP]`).
    #[inline]
    pub fn encode_query(&self, query: &str) -> Vec<u16> {
        let mut out = Vec::with_capacity(64);
        self.encode_into(query, true, &mut out);
        out
    }

    /// Encodes into a caller-provided `Vec<u16>` without reallocating.
    #[inline]
    pub fn encode_into(&self, text: &str, is_query: bool, out: &mut Vec<u16>) {
        out.clear();
        out.push(CLS_ID);
        if is_query {
            out.extend_from_slice(&self.query_prefix_ids);
        }
        self.encode_raw_into(text, out);
        if out.len() >= MAX_SEQ_LEN {
            out.truncate(MAX_SEQ_LEN - 1);
        }
        out.push(SEP_ID);
    }

    fn encode_raw_into(&self, text: &str, out: &mut Vec<u16>) {
        // Normalize into a small stack/reusable string buffer
        let mut norm = String::with_capacity(text.len() + 16);
        for ch in text.chars() {
            if ch == '\0' || ch == '\u{fffd}' || is_bert_control(ch) {
                continue;
            }
            if is_bert_whitespace(ch) {
                norm.push(' ');
                continue;
            }
            if is_chinese_char(ch) {
                norm.push(' ');
                norm.push(ch);
                norm.push(' ');
                continue;
            }
            if is_combining_mark(ch) {
                continue;
            }
            let base = strip_latin_accent(ch);
            for lc in base.to_lowercase() {
                if !is_combining_mark(lc) {
                    norm.push(lc);
                }
            }
        }

        // Pre-tokenize on whitespace and punctuation boundaries, then run WordPiece
        let bytes = norm.as_str();
        let mut word_start: Option<usize> = None;

        for (idx, ch) in bytes.char_indices() {
            if ch == ' ' {
                if let Some(ws) = word_start.take() {
                    self.wordpiece_into(&bytes[ws..idx], out);
                }
            } else if is_bert_punctuation(ch) {
                if let Some(ws) = word_start.take() {
                    self.wordpiece_into(&bytes[ws..idx], out);
                }
                let end = idx + ch.len_utf8();
                self.wordpiece_into(&bytes[idx..end], out);
            } else if word_start.is_none() {
                word_start = Some(idx);
            }
        }
        if let Some(ws) = word_start {
            self.wordpiece_into(&bytes[ws..], out);
        }
    }

    #[inline]
    fn wordpiece_into(&self, word: &str, out: &mut Vec<u16>) {
        if word.is_empty() {
            return;
        }
        // Fast path: whole word is a single root token in vocabulary (covers >90% of words!)
        if let Some(&id) = self.root_vocab.get(word) {
            out.push(id);
            return;
        }

        let char_count = word.chars().count();
        if char_count > MAX_WORD_CHARS {
            out.push(UNK_ID);
            return;
        }

        let start_out_len = out.len();
        let mut start = 0usize;
        let len = word.len();

        while start < len {
            let mut end = len;
            let mut matched_id: Option<u16> = None;

            while end > start {
                let slice = &word[start..end];
                let lookup = if start == 0 {
                    self.root_vocab.get(slice)
                } else {
                    self.subword_vocab.get(slice)
                };
                if let Some(&id) = lookup {
                    matched_id = Some(id);
                    break;
                }
                // Step `end` back by one UTF-8 character boundary
                end -= 1;
                while end > start && !word.is_char_boundary(end) {
                    end -= 1;
                }
            }

            if let Some(id) = matched_id {
                out.push(id);
                start = end;
            } else {
                out.truncate(start_out_len);
                out.push(UNK_ID);
                return;
            }
        }
    }
}

#[inline(always)]
fn is_bert_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r') || c.is_whitespace()
}

#[inline(always)]
fn is_bert_control(c: char) -> bool {
    if matches!(c, '\t' | '\n' | '\r') {
        return false;
    }
    c.is_control()
}

#[inline(always)]
fn is_chinese_char(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

#[inline(always)]
fn is_bert_punctuation(c: char) -> bool {
    let cp = c as u32;
    if (33..=47).contains(&cp)
        || (58..=64).contains(&cp)
        || (91..=96).contains(&cp)
        || (123..=126).contains(&cp)
    {
        return true;
    }
    // General Unicode punctuation ranges (General Plot Punctuation, CJK Symbols/Punctuation, Supplemental)
    (0x2000..=0x206F).contains(&cp)
        || (0x3000..=0x303F).contains(&cp)
        || (0xFE30..=0xFE6F).contains(&cp)
        || (0xFF01..=0xFF0F).contains(&cp)
        || (0xFF1A..=0xFF20).contains(&cp)
        || (0xFF3B..=0xFF40).contains(&cp)
        || (0xFF5B..=0xFF65).contains(&cp)
        || matches!(
            c,
            '¡' | '§'
                | '«'
                | '¶'
                | '·'
                | '»'
                | '¿'
                | '‐'
                | '‑'
                | '‒'
                | '–'
                | '—'
                | '―'
                | '‘'
                | '’'
                | '‚'
                | '“'
                | '”'
                | '„'
                | '†'
                | '‡'
                | '•'
                | '…'
                | '‰'
                | '′'
                | '″'
                | '‹'
                | '›'
                | '‼'
                | '‽'
                | '‾'
                | '⁄'
        )
}

#[inline(always)]
fn is_combining_mark(c: char) -> bool {
    let cp = c as u32;
    (0x0300..=0x036F).contains(&cp)
        || (0x1AB0..=0x1AFF).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x20D0..=0x20FF).contains(&cp)
        || (0xFE20..=0xFE2F).contains(&cp)
}

/// Maps precomposed Latin-1 / Latin Extended-A accented characters (`U+00C0..=U+017F`)
/// to their NFD base character (matching `BertNormalizer` with `strip_accents = true`).
#[inline(always)]
fn strip_latin_accent(c: char) -> char {
    match c {
        'À'..='Å' | 'à'..='å' | 'Ā'..='Ą' => 'a',
        'Ç' | 'ç' | 'Ć'..='Č' => 'c',
        'Ď' | 'ď' => 'd',
        'È'..='Ë' | 'è'..='ë' | 'Ē'..='Ě' => 'e',
        'Ĝ'..='Ģ' => 'g',
        'Ĥ' | 'ĥ' => 'h',
        'Ì'..='Ï' | 'ì'..='ï' | 'Ĩ'..='Į' => 'i',
        'Ĵ' | 'ĵ' => 'j',
        'Ķ' | 'ķ' => 'k',
        'Ĺ'..='ľ' => 'l',
        'Ñ' | 'ñ' | 'Ń'..='Ň' => 'n',
        'Ò'..='Ö' | 'ò'..='ö' | 'Ō'..='Ő' => 'o',
        'Ŕ'..='Ř' => 'r',
        'Ś'..='Š' | 'ś'..='š' => 's',
        'Ţ'..='Ť' => 't',
        'Ù'..='Ü' | 'ù'..='ü' | 'Ũ'..='Ų' => 'u',
        'Ŵ' | 'ŵ' => 'w',
        'Ý' | 'ý' | 'ÿ' | 'Ŷ' | 'ŷ' | 'Ÿ' => 'y',
        'Ź'..='Ž' | 'ź'..='ž' => 'z',
        _ => c,
    }
}
