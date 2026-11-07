//! Long-sequence (124..512 tokens) accuracy against a float64 numpy reference forward pass
//! (`scripts/fp32_reference.py`, validated to cos = 1.00000000 against the original reference vectors).
//! Exercises the SME attention path at lengths the short reference set never reaches.
mod common;
use arctic_embed_s_int8::model::Workspace;
use arctic_embed_s_int8::quantize::HIDDEN_SIZE;
use arctic_embed_s_int8::{cosine_similarity, ArcticEmbedS};
use std::path::PathBuf;

struct LongRecord {
    name: String,
    ids: Vec<u16>,
    embedding: [f32; HIDDEN_SIZE],
}

fn parse(json: &str) -> Vec<LongRecord> {
    let mut out = Vec::new();
    for block in json.split("{\"name\": \"").skip(1) {
        let name = block[..block.find('"').unwrap()].to_string();
        let list = |key: &str| -> &str {
            let pat = format!("\"{key}\": [");
            let s = block.find(&pat).expect(key) + pat.len();
            &block[s..s + block[s..].find(']').unwrap()]
        };
        let ids = list("ids")
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let emb: Vec<f32> = list("embedding_fp32")
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let mut embedding = [0.0f32; HIDDEN_SIZE];
        embedding.copy_from_slice(&emb);
        out.push(LongRecord {
            name,
            ids,
            embedding,
        });
    }
    out
}

#[test]
fn test_long_sequence_accuracy_against_fp64_reference() {
    let weights_dir = common::weights_dir();
    let data =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/long_reference_vectors.json");
    let records = parse(&std::fs::read_to_string(data).expect("long_reference_vectors.json"));
    assert_eq!(records.len(), 7);

    let engine = ArcticEmbedS::from_dir(&weights_dir).expect("load ArcticEmbedS");
    let mut ws = Workspace::new();
    let (mut sum, mut min) = (0.0f32, 1.0f32);
    for rec in &records {
        let emb = engine.embed_tokens(&rec.ids, &mut ws);
        let cos = cosine_similarity(&emb, &rec.embedding);
        println!(
            "{} ({:3} tokens): cosine_sim vs FP64 = {:.6}",
            rec.name,
            rec.ids.len(),
            cos
        );
        sum += cos;
        min = min.min(cos);
    }
    let mean = sum / records.len() as f32;
    println!("Long-sequence summary: mean_cos = {mean:.6}, min_cos = {min:.6}");
    assert!(min > 0.998, "long-sequence min cosine too low: {min}");
    assert!(mean > 0.999, "long-sequence mean cosine too low: {mean}");
}
