pub mod aligned;
pub mod gemm;
pub mod model;
pub mod parallel;
pub mod quantize;
pub mod simd_math;
pub mod tokenizer;

use crate::gemm::SimdBackend;
use crate::model::{PackedModel, Workspace};
use crate::quantize::{QuantizedModelRaw, HIDDEN_SIZE};
use crate::tokenizer::BertTokenizer;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Environment variable that points at the model directory (`vocab.txt` + `model.safetensors`).
pub const WEIGHTS_ENV_VAR: &str = "ARCTIC_EMBED_WEIGHTS";

/// Model directory used by the bundled binary, examples and tests: `$ARCTIC_EMBED_WEIGHTS` if set,
/// otherwise `./weights` (where `scripts/download_weights.sh` puts the files).
pub fn default_weights_dir() -> PathBuf {
    std::env::var_os(WEIGHTS_ENV_VAR)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("weights"))
}

/// High-performance Pure-Rust Int8 inference engine for `Snowflake/snowflake-arctic-embed-s`.
pub struct ArcticEmbedS {
    pub tokenizer: BertTokenizer,
    pub model: PackedModel,
    pub raw_size_bytes: usize,
}

impl ArcticEmbedS {
    /// Loads `vocab.txt` and either pre-quantized `model.ae8` or `model.safetensors` from `dir`.
    /// Automatically selects the fastest SIMD backend supported by the host CPU.
    pub fn from_dir<P: AsRef<Path>>(dir: P) -> io::Result<Self> {
        Self::from_dir_with_backend(dir, SimdBackend::detect())
    }

    /// Loads `vocab.txt` and either zero-copy memory-mapped `model.ae8p`, `model.ae8`, or `model.safetensors` from `dir`.
    pub fn from_dir_with_backend<P: AsRef<Path>>(dir: P, backend: SimdBackend) -> io::Result<Self> {
        let dir = dir.as_ref();
        let vocab_path = dir.join("vocab.txt");
        let ae8p_path = dir.join("model.ae8p");
        let ae8_path = dir.join("model.ae8");
        let st_path = dir.join("model.safetensors");

        let tokenizer = BertTokenizer::from_vocab_file(&vocab_path)?;
        if ae8p_path.exists() {
            if let Ok(model) = PackedModel::load_ae8p_mmap(&ae8p_path) {
                if model.backend == backend {
                    let raw_size_bytes = model.size_bytes();
                    return Ok(Self {
                        tokenizer,
                        model,
                        raw_size_bytes,
                    });
                }
            }
        }
        let raw = if ae8_path.exists() {
            QuantizedModelRaw::load_ae8(&ae8_path)?
        } else {
            QuantizedModelRaw::from_safetensors_file(&st_path)?
        };
        let model = PackedModel::from_raw(&raw, backend);
        let raw_size_bytes = model.size_bytes();
        Ok(Self {
            tokenizer,
            model,
            raw_size_bytes,
        })
    }

    /// Embeds a document/passage (`[CLS] + text + [SEP]`) using a reusable `Workspace` (zero heap allocations).
    #[inline]
    pub fn embed_document(&self, text: &str, ws: &mut Workspace) -> [f32; HIDDEN_SIZE] {
        let mut ids = std::mem::take(&mut ws.token_ids);
        self.tokenizer.encode_into(text, false, &mut ids);
        let mut out = [0.0f32; HIDDEN_SIZE];
        self.model.forward_cls(&ids, ws, &mut out);
        ws.token_ids = ids;
        out
    }

    /// Embeds a search query (`[CLS] + QUERY_PREFIX + query + [SEP]`) using a reusable `Workspace`.
    #[inline]
    pub fn embed_query(&self, query: &str, ws: &mut Workspace) -> [f32; HIDDEN_SIZE] {
        let mut ids = std::mem::take(&mut ws.token_ids);
        self.tokenizer.encode_into(query, true, &mut ids);
        let mut out = [0.0f32; HIDDEN_SIZE];
        self.model.forward_cls(&ids, ws, &mut out);
        ws.token_ids = ids;
        out
    }

    /// Embeds pre-tokenized `input_ids` using a reusable `Workspace`.
    #[inline]
    pub fn embed_tokens(&self, input_ids: &[u16], ws: &mut Workspace) -> [f32; HIDDEN_SIZE] {
        let mut out = [0.0f32; HIDDEN_SIZE];
        self.model.forward_cls(input_ids, ws, &mut out);
        out
    }

    /// Low-latency single-query embedding using several cores for one sequence (see [`parallel`]).
    pub fn embed_query_parallel(
        &self,
        query: &str,
        par: &mut crate::parallel::ParallelContext,
    ) -> [f32; HIDDEN_SIZE] {
        let ids = self.tokenizer.encode_query(query);
        self.embed_tokens_parallel(&ids, par)
    }

    /// Low-latency single-document embedding using several cores for one sequence (see [`parallel`]).
    pub fn embed_document_parallel(
        &self,
        text: &str,
        par: &mut crate::parallel::ParallelContext,
    ) -> [f32; HIDDEN_SIZE] {
        let ids = self.tokenizer.encode(text);
        self.embed_tokens_parallel(&ids, par)
    }

    /// Embeds pre-tokenized `input_ids` using several cores for one sequence (see [`parallel`]).
    pub fn embed_tokens_parallel(
        &self,
        input_ids: &[u16],
        par: &mut crate::parallel::ParallelContext,
    ) -> [f32; HIDDEN_SIZE] {
        let mut out = [0.0f32; HIDDEN_SIZE];
        par.forward_cls(&self.model, input_ids, &mut out);
        out
    }

    /// Multi-threaded lock-free work-stealing batch inference with unpadded ragged sequence packing ("Flash-VarLen").
    /// Packs multiple short sequences up to 64 tokens per micro-batch so 16-row SME / 8-row I8MM tiles run at full
    /// occupancy and Layer 11 executes all packed `[CLS]` tokens in a single batched tile.
    pub fn embed_batch<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
        is_query: bool,
        num_threads: usize,
    ) -> Vec<[f32; HIDDEN_SIZE]> {
        let n = texts.len();
        if n == 0 {
            return Vec::new();
        }
        let threads = num_threads.max(1).min(n);
        let mut results = vec![[0.0f32; HIDDEN_SIZE]; n];

        let run_worker = |next_ref: &AtomicUsize, results_ptr: usize| {
            let mut ws = Workspace::new();
            let mut tok_bufs: [Vec<u16>; 4] = [
                Vec::with_capacity(64),
                Vec::with_capacity(64),
                Vec::with_capacity(64),
                Vec::with_capacity(64),
            ];
            let mut batch_indices = [0usize; 4];
            let mut out_staging = [[0.0f32; HIDDEN_SIZE]; 4];
            const MAX_PACK_TOKENS: usize = 32;

            loop {
                let idx0 = next_ref.fetch_add(1, Ordering::Relaxed);
                if idx0 >= n {
                    break;
                }
                self.tokenizer
                    .encode_into(texts[idx0].as_ref(), is_query, &mut tok_bufs[0]);

                // If sequence already fills a 16-row SME tile, run directly with zero packing overhead
                if tok_bufs[0].len() >= 16 {
                    let mut emb = [0.0f32; HIDDEN_SIZE];
                    self.model.forward_cls(&tok_bufs[0], &mut ws, &mut emb);
                    unsafe {
                        let dst = (results_ptr as *mut [f32; HIDDEN_SIZE]).add(idx0);
                        std::ptr::write(dst, emb);
                    }
                    continue;
                }

                // Short sequence (< 16 tokens): pack up to 4 sequences into a single ragged tile
                batch_indices[0] = idx0;
                let mut pack_count = 1usize;
                let mut pack_tokens = tok_bufs[0].len();

                while pack_count < 4 && pack_tokens + 12 <= MAX_PACK_TOKENS {
                    let next_i = next_ref.fetch_add(1, Ordering::Relaxed);
                    if next_i >= n {
                        break;
                    }
                    self.tokenizer.encode_into(
                        texts[next_i].as_ref(),
                        is_query,
                        &mut tok_bufs[pack_count],
                    );
                    if pack_tokens + tok_bufs[pack_count].len() <= MAX_PACK_TOKENS {
                        batch_indices[pack_count] = next_i;
                        pack_tokens += tok_bufs[pack_count].len();
                        pack_count += 1;
                    } else {
                        // Execute the oversized sequence immediately, then break to flush the packed group
                        let mut emb = [0.0f32; HIDDEN_SIZE];
                        self.model
                            .forward_cls(&tok_bufs[pack_count], &mut ws, &mut emb);
                        unsafe {
                            let dst = (results_ptr as *mut [f32; HIDDEN_SIZE]).add(next_i);
                            std::ptr::write(dst, emb);
                        }
                        break;
                    }
                }

                if pack_count == 1 {
                    let mut emb = [0.0f32; HIDDEN_SIZE];
                    self.model.forward_cls(&tok_bufs[0], &mut ws, &mut emb);
                    unsafe {
                        let dst = (results_ptr as *mut [f32; HIDDEN_SIZE]).add(batch_indices[0]);
                        std::ptr::write(dst, emb);
                    }
                } else {
                    let mut seq_refs: [&[u16]; 4] = [&[]; 4];
                    for k in 0..pack_count {
                        seq_refs[k] = &tok_bufs[k];
                    }
                    self.model.forward_ragged_cls(
                        &seq_refs[..pack_count],
                        &mut ws,
                        &mut out_staging[..pack_count],
                    );
                    for k in 0..pack_count {
                        unsafe {
                            let dst =
                                (results_ptr as *mut [f32; HIDDEN_SIZE]).add(batch_indices[k]);
                            std::ptr::write(dst, out_staging[k]);
                        }
                    }
                }
            }
        };

        let next_idx = AtomicUsize::new(0);
        let results_ptr = results.as_mut_ptr() as usize;

        if threads == 1 {
            run_worker(&next_idx, results_ptr);
            return results;
        }

        std::thread::scope(|s| {
            for _ in 0..threads {
                let next_ref = &next_idx;
                s.spawn(move || {
                    run_worker(next_ref, results_ptr);
                });
            }
        });

        results
    }
}

/// Computes cosine similarity (dot product of L2-normalized 384-dim embeddings).
#[inline]
pub fn cosine_similarity(a: &[f32; HIDDEN_SIZE], b: &[f32; HIDDEN_SIZE]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..HIDDEN_SIZE {
        sum += a[i] * b[i];
    }
    sum
}
