# snowflake-arctic-embed-s-rust

Pure-Rust Int8 CPU inference for [`Snowflake/snowflake-arctic-embed-s`](https://huggingface.co/Snowflake/snowflake-arctic-embed-s)
(33M-parameter BERT, 384-dim embeddings). No ONNX Runtime, no BLAS, no dependencies: hand-written
SIMD kernels for Linux and macOS on ARM64 and x86_64.

- **W8A8 SmoothQuant** with per-row dynamic activation quantization; embeddings stay within
  cosine 0.9990 of the FP32 model (mean 0.9994 on short texts, 0.9993 on 124–512-token texts).
- **Runtime-selected kernels:** ARM64 SME2 (Apple M4 `smopa`), `FEAT_I8MM`, `DotProd`, NEON;
  x86_64 AVX-VNNI, AVX2; portable scalar fallback.
- **Latency mode:** one sequence split across both SME units of an M4 Pro/Max (`ParallelContext`).
- **Throughput mode:** multi-threaded ragged batching (`embed_batch`).
- **Fast startup:** the pre-quantized `.ae8` model pack (33.8 MB) loads and is packed for the host CPU in ~10 ms.

## Performance (Apple M4 Max, SME backend)

| Tokens | 1 thread | 2 threads (`ParallelContext`) |
|-------:|---------:|------------------------------:|
| 8      | 398 µs   | 298 µs |
| 32     | 1.14 ms  | 0.72 ms |
| 128    | 4.91 ms  | 2.89 ms |
| 512    | 31.3 ms  | 19.5 ms |

Reproduce with `cargo run --release --bin arctic-embed` (add `-- --quick` for a fast smoke run with
~20x fewer iterations, as CI does).

## Quick start

```bash
scripts/fetch_model.sh                    # fetches model/model.ae8 (v1.0.0, SHA-256 verified)
cargo run --release --bin arctic-embed    # runs retrieval demo & benchmark suite
```

```rust
use arctic_embed_s_int8::{cosine_similarity, model::Workspace, ArcticEmbedS};

let engine = ArcticEmbedS::from_dir("model")?; // model.ae8 + vocab.txt
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

## Model files

The library reads a model directory containing `model.ae8` (Int8 SmoothQuant weights) and `vocab.txt`.
`model.ae8` is backend-neutral and is built by [`snowflake-arctic-embed-s-weights`](https://github.com/mdlockyer/snowflake-arctic-embed-s-weights)
(also mirrored in [`snowflake-arctic-embed-s-swift`](https://github.com/mdlockyer/snowflake-arctic-embed-s-swift)).

Fetch the pinned release pack into `./model` with:

```bash
scripts/fetch_model.sh
```

## Testing

```bash
scripts/fetch_model.sh
cargo test --release
```

| Test | Checks |
|------|--------|
| `accuracy_test` | Tokenizer ids and embeddings vs. FP32 reference vectors; every SIMD backend the host supports |
| `long_accuracy_test` | 124–512-token texts vs. a float64 reference |
| `parallel_test` | `ParallelContext` output is bit-identical to the single-threaded forward pass (SME only) |

CI runs on x86_64 Linux, Neoverse N2 Linux (I8MM/SVE2) and Apple M1 macOS. No hosted runner has
SME, so the SME and intra-op parallel paths are only tested on local Apple M4 hardware.

## Layout

| Path | Contents |
|------|----------|
| `src/quantize.rs` | `.ae8` binary model loader (`QuantizedModelRaw`) |
| `src/gemm.rs` | Backend detection and Int8 GEMM kernels (SME, I8MM, DotProd, NEON, AVX-VNNI, AVX2) |
| `src/simd_math.rs` | Attention, softmax, GELU, LayerNorm and quantization kernels |
| `src/model.rs` | Per-backend packed model and forward pass |
| `src/parallel.rs` | Intra-op multi-core forward pass for one sequence |
| `src/tokenizer.rs` | BERT WordPiece tokenizer |
| `scripts/fetch_model.sh` | Fetches `model.ae8` + `vocab.txt` from `snowflake-arctic-embed-s-weights` |
| `examples/` | Profiling and micro-benchmarks |

## Model

`snowflake-arctic-embed-s` is © Snowflake Inc., released under Apache-2.0. `model/model.ae8` is a
modified (Int8-quantized) derivative of its weights; see [`model/README.md`](model/README.md) and
[`NOTICE`](NOTICE).

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project shall be licensed as above, without any additional terms or conditions.
