//! Intra-op parallel forward pass must match the single-threaded forward pass.
mod common;
use arctic_embed_s_int8::gemm::SimdBackend;
use arctic_embed_s_int8::model::Workspace;
use arctic_embed_s_int8::parallel::ParallelContext;
use arctic_embed_s_int8::{cosine_similarity, ArcticEmbedS};

#[test]
fn test_parallel_forward_matches_single_thread() {
    let engine = ArcticEmbedS::from_dir(common::model_dir()).expect("load");
    if engine.model.backend != SimdBackend::ArmSme {
        // Without SME the context always takes the single-threaded path, so this would only
        // compare the single-threaded forward pass with itself (and is slow on small CI runners).
        println!(
            "skipped: intra-op parallel path needs the SME backend (have {})",
            engine.model.backend.name()
        );
        return;
    }
    let mut ws = Workspace::new();
    let lengths = [1usize, 2, 5, 8, 15, 16, 17, 31, 33, 64, 100, 128, 257, 512];
    for threads in [2usize, 3, 4] {
        let mut par = ParallelContext::new(threads);
        for &len in &lengths {
            let mut ids = vec![101u16];
            for i in 0..len.saturating_sub(2) {
                ids.push(1996 + ((i * 37) % 5000) as u16);
            }
            if len >= 2 {
                ids.push(102);
            }
            let single = engine.embed_tokens(&ids, &mut ws);
            // Run twice to also exercise reuse of the warm helper threads.
            let _ = engine.embed_tokens_parallel(&ids, &mut par);
            let multi = engine.embed_tokens_parallel(&ids, &mut par);
            let cos = cosine_similarity(&single, &multi);
            let max_diff = single
                .iter()
                .zip(&multi)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("threads={threads} len={len:3}: cos={cos:.9} max_diff={max_diff:.2e}");
            assert!(
                cos > 0.99999,
                "parallel diverged: threads={threads} len={len} cos={cos}"
            );
        }
    }
}

/// Regression: a helper thread that the OS starts only after the first job was published must
/// still pick that job up (it used to read the already-advanced epoch and park forever).
#[test]
fn test_fresh_contexts_first_call_completes() {
    let engine = ArcticEmbedS::from_dir(common::model_dir()).expect("load");
    let mut ws = Workspace::new();
    let ids = [101u16, 2023, 2003, 1037, 3231, 102];
    let single = engine.embed_tokens(&ids, &mut ws);
    for i in 0..200 {
        let mut par = ParallelContext::new(2 + i % 3);
        let multi = engine.embed_tokens_parallel(&ids, &mut par);
        assert_eq!(single, multi, "fresh context {i} diverged");
    }
}
