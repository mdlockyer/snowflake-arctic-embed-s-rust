# Model pack

| File | Contents |
|------|----------|
| `model.ae8` | Int8 quantized weights of `Snowflake/snowflake-arctic-embed-s` (33.8 MB, fetched by `scripts/fetch_model.sh`) |
| `vocab.txt` | WordPiece vocabulary, unmodified from the original model |

This is the only model format the library reads. `ArcticEmbedS::from_dir("model")` loads it and
packs the weights for the host CPU's SIMD backend at startup (~2.8 ms).

## Fetching

`model/model.ae8` is built and published by [`mdlockyer/snowflake-arctic-embed-s-weights`](https://github.com/mdlockyer/snowflake-arctic-embed-s-weights)
and fetched via:

```bash
scripts/fetch_model.sh   # downloads v1.0.0 release assets into ./model (SHA-256 verified)
```

## Provenance

- Source: [`Snowflake/snowflake-arctic-embed-s`](https://huggingface.co/Snowflake/snowflake-arctic-embed-s),
  revision `e596f507467533e48a2e17c007f0e1dacc837b33`
  (`model.safetensors` SHA-256 `0350e62666ee2403db0223fc7cef6293c951cc63d1d2e71468da9a5aea3b544a`).
- `model.ae8` SHA-256: `d68b00aca78b143fb2fe3f8116eb2330d175f2a8318774725cf40bc906d564f1`.

## License

The original model is © Snowflake Inc. and licensed under the Apache License, Version 2.0; these
derived files are distributed under the same license (see [`../LICENSE`](../LICENSE) and
[`../NOTICE`](../NOTICE)).
