//! Shared helpers for the integration tests.
use std::path::PathBuf;

/// Model directory for tests: `$ARCTIC_EMBED_MODEL_DIR`, else the pack committed in `<crate>/model`.
pub fn model_dir() -> PathBuf {
    let dir = std::env::var_os(arctic_embed_s_int8::MODEL_DIR_ENV_VAR)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model"));
    assert!(
        dir.join(arctic_embed_s_int8::MODEL_FILE).is_file()
            && dir.join(arctic_embed_s_int8::VOCAB_FILE).is_file(),
        "model pack not found in {} (expected model.ae8 + vocab.txt)",
        dir.display()
    );
    dir
}
