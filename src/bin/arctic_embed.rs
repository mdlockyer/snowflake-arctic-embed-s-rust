use arctic_embed_s_int8::gemm::SimdBackend;
use arctic_embed_s_int8::model::{PackedModel, Workspace};
use arctic_embed_s_int8::quantize::{QuantizedModelRaw, HIDDEN_SIZE};
use arctic_embed_s_int8::{cosine_similarity, ArcticEmbedS};
use std::env;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let weights_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(arctic_embed_s_int8::default_weights_dir);

    println!("=========================================================================");
    println!("  Snowflake Arctic-Embed-S — Pure-Rust Int8 CPU Inference Engine");
    println!("=========================================================================");
    println!(
        "Host OS / Arch : {} / {}",
        env::consts::OS,
        env::consts::ARCH
    );
    let detected = SimdBackend::detect();
    println!("Active SIMD ISA: {}", detected.name());

    let t0 = Instant::now();
    let raw = QuantizedModelRaw::from_safetensors_file(weights_dir.join("model.safetensors"))
        .expect("Failed to load and quantize model.safetensors");
    let quant_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t_pack = Instant::now();
    let packed_model = PackedModel::from_raw(&raw, detected);
    let pack_ms = t_pack.elapsed().as_secs_f64() * 1000.0;

    // Benchmark zero-copy 64-byte-aligned .ae8p mmap load latency
    let tmp_ae8p = std::env::temp_dir().join("bench_arctic_embed_s.ae8p");
    packed_model
        .save_ae8p(&tmp_ae8p)
        .expect("Failed to save .ae8p");
    let t_mmap = Instant::now();
    let mmap_model = PackedModel::load_ae8p_mmap(&tmp_ae8p).expect("Failed to mmap .ae8p");
    let mmap_us = t_mmap.elapsed().as_secs_f64() * 1e6;
    std::hint::black_box(mmap_model.word_emb_q[0]);
    drop(mmap_model);
    let _ = std::fs::remove_file(&tmp_ae8p);

    let tokenizer = arctic_embed_s_int8::tokenizer::BertTokenizer::from_vocab_file(
        weights_dir.join("vocab.txt"),
    )
    .expect("Failed to load vocab.txt");

    let engine = ArcticEmbedS {
        tokenizer,
        raw_size_bytes: packed_model.size_bytes(),
        model: packed_model,
    };

    println!(
        "Quantized Size : {:.2} MB (3.97x compression vs 133.09 MB FP32)",
        engine.raw_size_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "Init Latency   : {:.2} ms (SmoothQuant + quantize), {:.2} ms (SIMD B-pack), {:.1} µs (Zero-Copy .ae8p mmap)",
        quant_ms, pack_ms, mmap_us
    );
    println!("-------------------------------------------------------------------------");

    let mut ws = Workspace::new();

    // 1. Retrieval Demo (Snowflake official queries & documents)
    let queries = [
        "what is snowflake?",
        "Where can I get the best tacos?",
        "How does SIMD Int8 matrix multiplication with ARM64 I8MM SMMLA and AVX2 work?",
    ];
    let documents = [
        "The Data Cloud!",
        "Mexico City of course!",
        "ARMv8.2-A introduces the I8MM extension (SMMLA instruction) which multiplies a 2x8 matrix of 8-bit integers by an 8x2 matrix of 8-bit integers into a 2x2 matrix of 32-bit integers.",
    ];

    let q_embs: Vec<[f32; HIDDEN_SIZE]> = queries
        .iter()
        .map(|q| engine.embed_query(q, &mut ws))
        .collect();
    let d_embs: Vec<[f32; HIDDEN_SIZE]> = documents
        .iter()
        .map(|d| engine.embed_document(d, &mut ws))
        .collect();

    println!("Retrieval Similarity Matrix (Query x Document):");
    for (qi, q) in queries.iter().enumerate() {
        print!("  Q{qi} ({:<42}): ", truncate_str(q, 42));
        for (di, _) in documents.iter().enumerate() {
            let score = cosine_similarity(&q_embs[qi], &d_embs[di]);
            print!("D{di}={:+.4}  ", score);
        }
        println!();
    }
    println!("-------------------------------------------------------------------------");

    // 2. Single-Thread Latency Benchmarks across sequence lengths
    println!(
        "Single-Thread Inference Latency (Pure Rust Int8, {}):",
        detected.name()
    );
    for &(label, seq_len, iters) in &[
        ("Short   ( 8 tokens)", 8usize, 500usize),
        ("Query   (16 tokens)", 16usize, 400usize),
        ("Medium  (32 tokens)", 32usize, 300usize),
        ("Passage (64 tokens)", 64usize, 200usize),
        ("Long   (128 tokens)", 128usize, 100usize),
        ("Max    (512 tokens)", 512usize, 30usize),
    ] {
        let mut fake_ids = vec![101u16];
        for i in 0..(seq_len - 2) {
            fake_ids.push(2000 + (i as u16 % 1000));
        }
        fake_ids.push(102u16);

        // Warmup
        for _ in 0..10 {
            let _ = engine.embed_tokens(&fake_ids, &mut ws);
        }

        let start = Instant::now();
        for _ in 0..iters {
            let emb = engine.embed_tokens(&fake_ids, &mut ws);
            std::hint::black_box(emb[0]);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let us_per_call = (elapsed * 1e6) / (iters as f64);
        let seqs_per_sec = (iters as f64) / elapsed;
        let toks_per_sec = seqs_per_sec * (seq_len as f64);
        println!(
            "  {label}: {:7.1} µs/embed | {:7.1} embeds/s | {:9.0} tokens/s",
            us_per_call, seqs_per_sec, toks_per_sec
        );
    }
    println!("-------------------------------------------------------------------------");

    // 2.1 Intra-op parallel latency (one sequence on several cores; SME backend only)
    let mut par = arctic_embed_s_int8::parallel::ParallelContext::recommended(&engine);
    if par.num_threads() > 1 {
        println!(
            "Intra-Op Parallel Latency ({} threads, one sequence):",
            par.num_threads()
        );
        for &(label, seq_len, iters) in &[
            ("Short   ( 8 tokens)", 8usize, 500usize),
            ("Query   (16 tokens)", 16usize, 400usize),
            ("Medium  (32 tokens)", 32usize, 300usize),
            ("Passage (64 tokens)", 64usize, 200usize),
            ("Long   (128 tokens)", 128usize, 100usize),
            ("Max    (512 tokens)", 512usize, 30usize),
        ] {
            let mut ids = vec![101u16];
            for i in 0..(seq_len - 2) {
                ids.push(2000 + (i as u16 % 1000));
            }
            ids.push(102u16);
            for _ in 0..10 {
                let _ = engine.embed_tokens_parallel(&ids, &mut par);
            }
            let start = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(engine.embed_tokens_parallel(&ids, &mut par)[0]);
            }
            let us = start.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("  {label}: {us:7.1} µs/embed");
        }
        println!("-------------------------------------------------------------------------");
    }
    drop(par);

    // 2.5 Per-Operator Breakdown (M = 32, Layer 0)
    {
        use arctic_embed_s_int8::gemm::gemm_int8_fused;
        use arctic_embed_s_int8::simd_math::{
            gelu_smooth_quantize_batch, layernorm_smooth_quantize_row,
        };
        let l0 = &engine.model.layers[0];
        let m = 32usize;
        let iters = 500usize;

        let t0 = Instant::now();
        for _ in 0..iters {
            gemm_int8_fused(
                m,
                &ws.act_q[..m * 384],
                &mut ws.act_packed[..(m + 8) * 384],
                &ws.act_scale[..m],
                &l0.qkv,
                None,
                &mut ws.qkv_buf[..m * 1152],
            );
        }
        let qkv_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let t1 = Instant::now();
        for _ in 0..iters {
            gemm_int8_fused(
                m,
                &ws.act_q[..m * 384],
                &mut ws.act_packed[..(m + 8) * 384],
                &ws.act_scale[..m],
                &l0.attn_out,
                Some(&ws.hidden[..m * 384]),
                &mut ws.tmp_hidden[..m * 384],
            );
        }
        let ao_us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let t2 = Instant::now();
        for _ in 0..iters {
            gemm_int8_fused(
                m,
                &ws.act_q[..m * 384],
                &mut ws.act_packed[..(m + 8) * 384],
                &ws.act_scale[..m],
                &l0.ffn_inter,
                None,
                &mut ws.inter_buf[..m * 1536],
            );
        }
        let fi_us = t2.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let t3 = Instant::now();
        for _ in 0..iters {
            gemm_int8_fused(
                m,
                &ws.inter_q[..m * 1536],
                &mut ws.inter_packed[..(m + 8) * 1536],
                &ws.act_scale[..m],
                &l0.ffn_out,
                Some(&ws.hidden[..m * 384]),
                &mut ws.tmp_hidden[..m * 384],
            );
        }
        let fo_us = t3.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let t4 = Instant::now();
        for _ in 0..iters {
            for r in 0..m {
                layernorm_smooth_quantize_row(
                    &mut ws.hidden[r * 384..(r + 1) * 384],
                    &l0.ln1_weight,
                    &l0.ln1_bias,
                    &l0.ffn_inter.inv_smooth,
                    &mut ws.act_q[r * 384..(r + 1) * 384],
                    1e-12,
                );
            }
        }
        let ln_us = t4.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let t5 = Instant::now();
        for _ in 0..iters {
            gelu_smooth_quantize_batch(
                m,
                &mut ws.inter_buf[..m * 1536],
                &l0.ffn_out.inv_smooth,
                &mut ws.inter_q[..m * 1536],
                &mut ws.act_scale[..m],
                detected == SimdBackend::ArmSme,
            );
        }
        let gelu_us = t5.elapsed().as_secs_f64() * 1e6 / iters as f64;

        println!(
            "Layer Breakdown (M=32): QKV={:.1}µs, AttnOut={:.1}µs, FFN_Up={:.1}µs, FFN_Down={:.1}µs, LN+Q={:.1}µs, GELU+Q={:.1}µs",
            qkv_us, ao_us, fi_us, fo_us, ln_us, gelu_us
        );
        println!("-------------------------------------------------------------------------");
    }

    // 3. Compare SIMD Backends on the same hardware (16-token query & 64-token passage)
    println!("Backend Comparison on Host CPU (32-token sequence, single thread):");
    let mut bench_ids = vec![101u16];
    for i in 0..30 {
        bench_ids.push(2000 + i);
    }
    bench_ids.push(102u16);

    // Every backend this CPU can run (forcing an unsupported one is an illegal instruction).
    let backends: Vec<SimdBackend> = [
        SimdBackend::ArmSme,
        SimdBackend::ArmI8mm,
        SimdBackend::ArmDotProd,
        SimdBackend::ArmNeon,
        SimdBackend::X86AvxVnni,
        SimdBackend::X86Avx2,
        SimdBackend::Scalar,
    ]
    .into_iter()
    .filter(|b| b.is_supported())
    .collect();

    let ref_emb = engine.embed_tokens(&bench_ids, &mut ws);
    for &b in &backends {
        let pm = PackedModel::from_raw(&raw, b);
        let mut out = [0.0f32; HIDDEN_SIZE];
        for _ in 0..5 {
            pm.forward_cls(&bench_ids, &mut ws, &mut out);
        }
        let iters = if b == SimdBackend::Scalar { 20 } else { 200 };
        let start = Instant::now();
        for _ in 0..iters {
            pm.forward_cls(&bench_ids, &mut ws, &mut out);
            std::hint::black_box(out[0]);
        }
        let us = (start.elapsed().as_secs_f64() * 1e6) / (iters as f64);
        let cos = cosine_similarity(&ref_emb, &out);
        println!(
            "  {:<32}: {:8.1} µs/embed ({:6.1} embeds/s) | cos vs top tier = {:.7}",
            b.name(),
            us,
            1e6 / us,
            cos
        );
    }
    println!("-------------------------------------------------------------------------");

    // 4. Multi-Threaded Batch Throughput
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let batch_size = 256;
    let sample_docs: Vec<&str> = (0..batch_size)
        .map(|i| match i % 4 {
            0 => "Snowflake Arctic Embed is a suite of text embedding models optimized for retrieval performance.",
            1 => "That is a happy person enjoying a sunny day in Mexico City with great tacos.",
            2 => "Pure Rust SIMD Int8 inference using ARM64 I8MM SMMLA and x86_64 AVX2 instructions.",
            _ => "The Data Cloud enables organizations to mobilize their data with near-unlimited scale.",
        })
        .collect();

    for &threads in &[1, 2, 4, 8, num_cpus] {
        let _ = engine.embed_batch(&sample_docs[..32], false, threads);
        let start = Instant::now();
        let out = engine.embed_batch(&sample_docs, false, threads);
        std::hint::black_box(out[0][0]);
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "Passage Batch ({:2} threads, {} docs): {:7.1} docs/sec ({:.2} ms total)",
            threads,
            batch_size,
            (batch_size as f64) / elapsed,
            elapsed * 1000.0
        );
    }
    println!("-------------------------------------------------------------------------");
    let short_docs: Vec<&str> = (0..batch_size)
        .map(|i| match i % 4 {
            0 => "The Data Cloud!",
            1 => "Mexico City of course!",
            2 => "That is a happy person",
            _ => "Today is a sunny day",
        })
        .collect();
    for &threads in &[1, 2, 4, 8, num_cpus] {
        let _ = engine.embed_batch(&short_docs[..32], false, threads);
        let start = Instant::now();
        let out = engine.embed_batch(&short_docs, false, threads);
        std::hint::black_box(out[0][0]);
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "Short Ragged  ({:2} threads, {} docs): {:7.1} docs/sec ({:.2} ms total)",
            threads,
            batch_size,
            (batch_size as f64) / elapsed,
            elapsed * 1000.0
        );
    }
    println!("=========================================================================");
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
