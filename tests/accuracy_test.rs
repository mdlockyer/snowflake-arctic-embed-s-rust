mod common;
use arctic_embed_s_int8::gemm::SimdBackend;
use arctic_embed_s_int8::model::{PackedModel, Workspace};
use arctic_embed_s_int8::quantize::{QuantizedModelRaw, HIDDEN_SIZE};
use arctic_embed_s_int8::tokenizer::BertTokenizer;
use arctic_embed_s_int8::{cosine_similarity, ArcticEmbedS};
use std::fs;
use std::path::PathBuf;

struct RefRecord {
    kind: String,
    raw_text: String,
    ids: Vec<u16>,
    embedding_fp32: [f32; HIDDEN_SIZE],
}

fn parse_ref_records(json_str: &str) -> Vec<RefRecord> {
    let mut out = Vec::new();
    for block in json_str.split('{').skip(1) {
        let extract_str = |key: &str| -> String {
            let pat = format!("\"{}\": \"", key);
            let start = block.find(&pat).expect(key) + pat.len();
            let mut end = start;
            let bytes = block.as_bytes();
            while end < bytes.len() {
                if bytes[end] == b'"' && bytes[end - 1] != b'\\' {
                    break;
                }
                end += 1;
            }
            block[start..end]
                .replace("\\u00e9", "é")
                .replace("\\u00ef", "ï")
                .replace("\\u2014", "—")
        };

        let kind = extract_str("kind");
        let raw_text = extract_str("raw_text");

        let ids_start = block.find("\"ids\": [").unwrap() + 8;
        let ids_end = ids_start + block[ids_start..].find(']').unwrap();
        let ids: Vec<u16> = block[ids_start..ids_end]
            .split(',')
            .map(|s| s.trim().parse::<u16>().unwrap())
            .collect();

        let emb_start = block.find("\"embedding_fp32\": [").unwrap() + 19;
        let emb_end = emb_start + block[emb_start..].find(']').unwrap();
        let emb_vec: Vec<f32> = block[emb_start..emb_end]
            .split(',')
            .map(|s| s.trim().parse::<f32>().unwrap())
            .collect();
        assert_eq!(emb_vec.len(), HIDDEN_SIZE);
        let mut embedding_fp32 = [0.0f32; HIDDEN_SIZE];
        embedding_fp32.copy_from_slice(&emb_vec);

        out.push(RefRecord {
            kind,
            raw_text,
            ids,
            embedding_fp32,
        });
    }
    out
}

#[test]
fn test_tokenizer_and_int8_accuracy_against_fp32_reference() {
    let model_dir = common::model_dir();
    let ref_json = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/reference_test_vectors.json"),
    )
    .expect("reference_test_vectors.json must exist");
    let records = parse_ref_records(&ref_json);
    assert_eq!(records.len(), 12);

    let engine = ArcticEmbedS::from_dir(&model_dir).expect("load ArcticEmbedS");
    let mut ws = Workspace::new();

    let mut cos_sum = 0.0f32;
    let mut min_cos = 1.0f32;

    for (idx, rec) in records.iter().enumerate() {
        let is_query = rec.kind == "query";
        let got_ids = if is_query {
            engine.tokenizer.encode_query(&rec.raw_text)
        } else {
            engine.tokenizer.encode(&rec.raw_text)
        };
        assert_eq!(
            got_ids, rec.ids,
            "Tokenizer mismatch on record {} ({:?})",
            idx, rec.raw_text
        );

        let emb = if is_query {
            engine.embed_query(&rec.raw_text, &mut ws)
        } else {
            engine.embed_document(&rec.raw_text, &mut ws)
        };
        let cos = cosine_similarity(&emb, &rec.embedding_fp32);
        println!(
            "Record {:2} ({:<8} | {:<45}): cosine_sim vs FP32 = {:.6}",
            idx,
            rec.kind,
            &rec.raw_text[..rec.raw_text.len().min(45)],
            cos
        );
        assert!(
            cos > 0.9985,
            "Cosine similarity too low on record {}: {}",
            idx,
            cos
        );
        cos_sum += cos;
        if cos < min_cos {
            min_cos = cos;
        }
    }

    let mean_cos = cos_sum / (records.len() as f32);
    println!(
        "Summary across {} reference vectors: mean_cos = {:.6}, min_cos = {:.6}",
        records.len(),
        mean_cos,
        min_cos
    );
    assert!(
        mean_cos > 0.9992,
        "Mean cosine similarity below 0.9992: {}",
        mean_cos
    );

    // Verify that unpadded ragged sequence packing (`embed_batch` / `forward_ragged_cls`) matches single-sequence inference
    let doc_texts: Vec<&str> = records
        .iter()
        .filter(|r| r.kind == "document")
        .map(|r| r.raw_text.as_str())
        .collect();
    let batch_embs = engine.embed_batch(&doc_texts, false, 1);
    for (i, text) in doc_texts.iter().enumerate() {
        let single_emb = engine.embed_document(text, &mut ws);
        let cos_ragged = cosine_similarity(&single_emb, &batch_embs[i]);
        assert!(
            cos_ragged > 0.99999,
            "Ragged batch mismatch at doc {}: cos={}",
            i,
            cos_ragged
        );
    }
}

#[test]
fn test_all_simd_backends_and_ae8_roundtrip_bit_identical() {
    let model_dir = common::model_dir();
    let raw = QuantizedModelRaw::load_ae8(model_dir.join("model.ae8")).expect("load model.ae8");
    let tok = BertTokenizer::from_vocab_file(model_dir.join("vocab.txt")).expect("load vocab");

    let ids = tok.encode_query("what is snowflake?");
    let mut ws = Workspace::new();

    let mut scalar_out = [0.0f32; HIDDEN_SIZE];
    PackedModel::from_raw(&raw, SimdBackend::Scalar).forward_cls(&ids, &mut ws, &mut scalar_out);

    for &b in &[
        SimdBackend::ArmSme,
        SimdBackend::ArmI8mm,
        SimdBackend::ArmDotProd,
        SimdBackend::ArmNeon,
        SimdBackend::X86AvxVnni,
        SimdBackend::X86Avx2,
    ] {
        if !b.is_supported() {
            println!(
                "Backend {:<32} not supported on this CPU, skipped",
                b.name()
            );
            continue;
        }
        let mut out = [0.0f32; HIDDEN_SIZE];
        PackedModel::from_raw(&raw, b).forward_cls(&ids, &mut ws, &mut out);
        let cos = cosine_similarity(&scalar_out, &out);
        let max_diff = scalar_out
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "Backend {:<32} vs Scalar: cos={:.8}, max_diff={:.8}",
            b.name(),
            cos,
            max_diff
        );
        assert!(
            cos > 0.999,
            "Backend {:?} diverged from Scalar: cos={}",
            b,
            cos
        );
    }

    // Test .ae8 serialization round-trip
    let tmp_ae8 = std::env::temp_dir().join("test_arctic_embed_s.ae8");
    raw.save_ae8(&tmp_ae8).expect("save .ae8");
    let loaded_raw = QuantizedModelRaw::load_ae8(&tmp_ae8).expect("load .ae8");
    let _ = fs::remove_file(&tmp_ae8);

    let mut detected_out = [0.0f32; HIDDEN_SIZE];
    PackedModel::from_raw(&raw, SimdBackend::detect()).forward_cls(
        &ids,
        &mut ws,
        &mut detected_out,
    );

    let mut loaded_out = [0.0f32; HIDDEN_SIZE];
    let packed_detected = PackedModel::from_raw(&loaded_raw, SimdBackend::detect());
    packed_detected.forward_cls(&ids, &mut ws, &mut loaded_out);
    assert_eq!(
        detected_out, loaded_out,
        ".ae8 roundtrip must be bit-identical"
    );

    assert_eq!(
        packed_detected.layers[11].qkv.packed_w.len(),
        0,
        "Layer 11 dead qkv must be pruned"
    );
}
