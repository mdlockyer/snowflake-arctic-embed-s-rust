//! The committed `model/model.ae8` must be exactly what the converter produces from the pinned
//! FP32 checkpoint: catches a quantizer change that was not followed by regenerating the pack.
//!
//! Needs `--features convert` and the checkpoint from `scripts/download_weights.sh` (in
//! `<crate>/weights`, or the directory named by `$ARCTIC_EMBED_SOURCE_WEIGHTS`).
#![cfg(feature = "convert")]

use arctic_embed_s_int8::quantize::QuantizedModelRaw;
use std::path::PathBuf;

#[test]
fn test_committed_pack_matches_conversion_of_pinned_checkpoint() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = std::env::var_os("ARCTIC_EMBED_SOURCE_WEIGHTS")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("weights"))
        .join("model.safetensors");
    assert!(
        source.is_file(),
        "{} not found — run `scripts/download_weights.sh` or set ARCTIC_EMBED_SOURCE_WEIGHTS",
        source.display()
    );

    let raw = QuantizedModelRaw::from_safetensors_file(&source).expect("convert checkpoint");
    let out = std::env::temp_dir().join(format!("convert_test_{}.ae8", std::process::id()));
    raw.save_ae8(&out).expect("write pack");
    let converted = std::fs::read(&out).expect("read converted pack");
    let _ = std::fs::remove_file(&out);

    let committed = std::fs::read(root.join("model/model.ae8")).expect("read committed pack");
    assert_eq!(
        converted.len(),
        committed.len(),
        "pack size differs from model/model.ae8"
    );
    if let Some(i) = converted.iter().zip(&committed).position(|(a, b)| a != b) {
        panic!(
            "model/model.ae8 is stale: first difference at byte {i}. Regenerate it with \
             `cargo run --release --features convert --bin arctic-embed-convert -- \
             weights/model.safetensors model/model.ae8`"
        );
    }
}
