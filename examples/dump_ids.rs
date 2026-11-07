//! Tokenizes each line of a text file as a document and writes `[{"name", "ids"}]` JSON for the FP32 reference.
use arctic_embed_s_int8::tokenizer::BertTokenizer;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let tok = BertTokenizer::from_vocab_file(format!("{}/vocab.txt", a[1])).unwrap();
    let text = std::fs::read_to_string(&a[2]).unwrap();
    let mut out = String::from("[");
    for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let ids = tok.encode(line);
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("{{\"name\": \"long{i}\", \"ids\": {:?}}}", ids));
        eprintln!("long{i}: {} tokens", ids.len());
    }
    out.push(']');
    std::fs::write(&a[3], out).unwrap();
}
