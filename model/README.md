# Model pack

| File | Contents |
|------|----------|
| `model.ae8` | Int8 quantized weights of `Snowflake/snowflake-arctic-embed-s` (33.8 MB) |
| `vocab.txt` | WordPiece vocabulary, unmodified from the original model |

This is the only model format the library reads. `ArcticEmbedS::from_dir("model")` loads it and
packs the weights for the host CPU's SIMD backend at startup (a few milliseconds).

## Provenance

- Source: [`Snowflake/snowflake-arctic-embed-s`](https://huggingface.co/Snowflake/snowflake-arctic-embed-s),
  revision `e596f507467533e48a2e17c007f0e1dacc837b33`
  (`model.safetensors` SHA-256 `0350e62666ee2403db0223fc7cef6293c951cc63d1d2e71468da9a5aea3b544a`).
- `model.ae8` SHA-256: `89b7d536f4c37d18a042d958635277acbdb7f0ac741a85bde9027c2c6f7d61dc`.

**Modifications:** `model.ae8` is a derived work of the original FP32 weights. It was produced by
SmoothQuant calibration (alpha = 0.40) followed by symmetric per-row Int8 quantization of all
linear layers and embedding tables, and serialized in this project's `.ae8` format.

## Regenerating

The conversion is deterministic, and CI checks that the committed pack matches it:

```bash
scripts/download_weights.sh                     # pinned checkpoint into ./weights
cargo run --release --features convert --bin arctic-embed-convert -- weights/model.safetensors model/model.ae8
```

## License

The original model is © Snowflake Inc. and licensed under the Apache License, Version 2.0; these
derived files are distributed under the same license (see [`../LICENSE`](../LICENSE) and
[`../NOTICE`](../NOTICE)).
