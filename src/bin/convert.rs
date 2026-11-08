//! Builds the quantized model pack (`model.ae8`) from the original FP32 checkpoint.
//!
//! ```text
//! cargo run --release --features convert --bin arctic-embed-convert -- <model.safetensors> <out.ae8>
//! ```
//!
//! Runs SmoothQuant calibration and symmetric Int8 quantization (deterministic: the same input
//! always gives a byte-identical pack). Fetch the pinned checkpoint with
//! `scripts/download_weights.sh`.
use arctic_embed_s_int8::quantize::QuantizedModelRaw;
use std::process::ExitCode;
use std::time::Instant;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [input, output] = args.as_slice() else {
        eprintln!("usage: arctic-embed-convert <model.safetensors> <out.ae8>");
        return ExitCode::from(2);
    };
    let t = Instant::now();
    let raw = match QuantizedModelRaw::from_safetensors_file(input) {
        Ok(raw) => raw,
        Err(e) => {
            eprintln!("error: reading {input}: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = raw.save_ae8(output) {
        eprintln!("error: writing {output}: {e}");
        return ExitCode::FAILURE;
    }
    let bytes = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
    println!(
        "wrote {output}: {:.1} MB in {:.0} ms",
        bytes as f64 / 1e6,
        t.elapsed().as_secs_f64() * 1e3
    );
    ExitCode::SUCCESS
}
