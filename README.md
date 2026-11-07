# snowflake-arctic-embed-s-rust

Pure-Rust Int8 CPU inference for [`Snowflake/snowflake-arctic-embed-s`](https://huggingface.co/Snowflake/snowflake-arctic-embed-s)
(33M-parameter BERT, 384-dim embeddings). No ONNX Runtime, no BLAS, no dependencies: hand-written
SIMD kernels for Linux and macOS on ARM64 and x86_64.

- **W8A8 SmoothQuant** with per-row dynamic activation quantization; embeddings stay within
  cosine 0.9991 of the FP32 model (mean 0.9994 on short texts, 0.9993 on 124–512-token texts).
- **Runtime-selected kernels:** ARM64 SME2 (Apple M4 `smopa`), `FEAT_I8MM`, `DotProd`, NEON;
  x86_64 AVX-VNNI, AVX2; portable scalar fallback.
- **Latency mode:** one sequence split across both SME units of an M4 Pro/Max (`ParallelContext`).
- **Throughput mode:** multi-threaded ragged batching (`embed_batch`).
- **Fast startup:** pre-packed weights can be saved as `.ae8p` and memory-mapped with zero copies.

## Performance (Apple M4 Max, SME backend)

| Tokens | 1 thread | 2 threads (`ParallelContext`) |
|-------:|---------:|------------------------------:|
| 8      | 398 µs   | 298 µs |
| 32     | 1.14 ms  | 0.72 ms |
| 128    | 4.91 ms  | 2.89 ms |
| 512    | 31.3 ms  | 19.5 ms |

Reproduce with `cargo run --release --bin arctic-embed`.

## Quick start

```bash
scripts/download_weights.sh          # pinned Hub revision, SHA-256 verified, into ./weights
cargo run --release --bin arctic-embed
```

```rust
use arctic_embed_s_int8::{cosine_similarity, model::Workspace, ArcticEmbedS};

let engine = ArcticEmbedS::from_dir("weights")?;
let mut ws = Workspace::new();
let q = engine.embed_query("what is snowflake?", &mut ws); // adds the query prefix
let d = engine.embed_document("Snowflake is a cloud data platform.", &mut ws);
println!("similarity = {}", cosine_similarity(&q, &d));

// Lowest single-query latency on SME machines (2 threads; single-threaded elsewhere):
let mut par = arctic_embed_s_int8::parallel::ParallelContext::recommended(&engine);
let q = engine.embed_query_parallel("what is snowflake?", &mut par);

// Throughput: many texts across all cores.
let docs = engine.embed_batch(&["first doc", "second doc"], false, 8);
```

Embeddings are L2-normalized `[f32; 384]` taken from the `[CLS]` token, matching the reference
sentence-transformers pipeline. Inputs are truncated to 512 tokens.

## Weights

The crate reads `vocab.txt` and `model.safetensors` (or a pre-packed `model.ae8p`) from a directory.
The binary, examples and tests look in `$ARCTIC_EMBED_WEIGHTS`, falling back to `./weights`.
`scripts/download_weights.sh [dir]` fetches the files pinned to Hub revision
`e596f507467533e48a2e17c007f0e1dacc837b33` and checks their SHA-256.

## Testing

```bash
cargo test --release
```

| Test | Checks |
|------|--------|
| `accuracy_test` | Tokenizer ids and embeddings vs. FP32 reference vectors; every SIMD backend the host supports; `.ae8p` round-trip |
| `long_accuracy_test` | 124–512-token texts vs. a float64 reference (`scripts/fp32_reference.py`) |
| `parallel_test` | `ParallelContext` output is bit-identical to the single-threaded forward pass |

CI runs on x86_64 Linux, Neoverse N2 Linux (I8MM/SVE2) and Apple M1 macOS. No hosted runner has
SME, so the SME and intra-op parallel paths are only tested on local Apple M4 hardware.

Regenerating the long-sequence reference (needs Python with NumPy):

```bash
cargo run --release --example dump_ids -- weights tests/data/long_texts.txt /tmp/long_ids.json
python3 -I scripts/fp32_reference.py embed weights /tmp/long_ids.json tests/data/long_reference_vectors.json
```

## Layout

| Path | Contents |
|------|----------|
| `src/quantize.rs` | safetensors loading, SmoothQuant calibration, Int8 quantization |
| `src/gemm.rs` | Backend detection and Int8 GEMM kernels (SME, I8MM, DotProd, NEON, AVX-VNNI, AVX2) |
| `src/simd_math.rs` | Attention, softmax, GELU, LayerNorm and quantization kernels |
| `src/model.rs` | Packed model, forward pass, `.ae8p` serialization |
| `src/parallel.rs` | Intra-op multi-core forward pass for one sequence |
| `src/tokenizer.rs` | BERT WordPiece tokenizer |
| `examples/` | Profiling and micro-benchmarks |

## Model

`snowflake-arctic-embed-s` is © Snowflake Inc., released under Apache-2.0. Model weights are not
included in this repository.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project shall be licensed as above, without any additional terms or conditions.
