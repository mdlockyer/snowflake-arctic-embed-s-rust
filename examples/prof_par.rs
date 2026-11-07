use arctic_embed_s_int8::model::Workspace;
use arctic_embed_s_int8::parallel::ParallelContext;
use arctic_embed_s_int8::ArcticEmbedS;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let seq: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(32);
    let secs: f64 = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(2.0);
    let threads: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(2);
    let e = ArcticEmbedS::from_dir(arctic_embed_s_int8::default_weights_dir()).unwrap();
    let mut ids = vec![101u16];
    for i in 0..seq - 2 {
        ids.push(2000 + (i as u16 % 1000));
    }
    ids.push(102);
    let mut par = ParallelContext::new(threads);
    let mut ws = Workspace::new();
    for _ in 0..20 {
        std::hint::black_box(e.embed_tokens_parallel(&ids, &mut par));
    }
    let t = Instant::now();
    let mut n = 0u64;
    while t.elapsed().as_secs_f64() < secs {
        if threads == 1 {
            std::hint::black_box(e.embed_tokens(&ids, &mut ws));
        } else {
            std::hint::black_box(e.embed_tokens_parallel(&ids, &mut par));
        }
        n += 1;
    }
    println!(
        "seq={seq} {:.1} us/embed",
        t.elapsed().as_secs_f64() * 1e6 / n as f64
    );
}
