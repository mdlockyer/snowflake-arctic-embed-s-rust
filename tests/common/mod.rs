//! Shared helpers for the integration tests.
use std::path::PathBuf;

/// Model directory for tests: `$ARCTIC_EMBED_WEIGHTS`, else `<crate>/weights`.
/// Panics with setup instructions when the weights are missing.
pub fn weights_dir() -> PathBuf {
    let dir = std::env::var_os(arctic_embed_s_int8::WEIGHTS_ENV_VAR)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("weights"));
    assert!(
        dir.join("model.safetensors").is_file() && dir.join("vocab.txt").is_file(),
        "model weights not found in {} — run `scripts/download_weights.sh` or set {}",
        dir.display(),
        arctic_embed_s_int8::WEIGHTS_ENV_VAR
    );
    dir
}
