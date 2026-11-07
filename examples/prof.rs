use arctic_embed_s_int8::model::Workspace;
use arctic_embed_s_int8::ArcticEmbedS;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let seq: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(32);
    let secs: f64 = a.get(2).map(|s| s.parse().unwrap()).unwrap_or(5.0);
    let e = ArcticEmbedS::from_dir(arctic_embed_s_int8::default_weights_dir()).unwrap();
    let mut ws = Workspace::new();
    let mut ids = vec![101u16];
    for i in 0..seq - 2 {
        ids.push(2000 + (i as u16 % 1000));
    }
    ids.push(102);
    let t = Instant::now();
    let mut n = 0u64;
    while t.elapsed().as_secs_f64() < secs {
        std::hint::black_box(e.embed_tokens(&ids, &mut ws));
        n += 1;
    }
    println!(
        "seq={seq} {:.1} us/embed",
        t.elapsed().as_secs_f64() * 1e6 / n as f64
    );
}
