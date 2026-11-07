"""Pure-numpy FP32 reference forward pass for Snowflake/snowflake-arctic-embed-s (BERT, CLS pooling, L2 norm).

Usage (run with `python3 -I`):
    python3 -I scripts/fp32_reference.py validate <weights_dir>
        Recomputes tests/data/reference_test_vectors.json embeddings from their token ids and reports cosine similarity.
    python3 -I scripts/fp32_reference.py embed <weights_dir> <ids_json> <out_json>
        <ids_json> is a JSON list of {"name": str, "ids": [int, ...]}; writes the same records plus
        "embedding_fp32" (384 floats) to <out_json>.

Computation is done in float64 for a tight reference (the model weights are float32).
"""

import json
import math
import os
import struct
import sys

import numpy as np

HIDDEN = 384
HEADS = 12
HEAD_DIM = 32
LAYERS = 12
EPS = 1e-12


def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        assert meta["dtype"] == "F32", (name, meta["dtype"])
        start, end = meta["data_offsets"]
        out[name] = np.frombuffer(data[start:end], dtype="<f4").reshape(meta["shape"]).astype(np.float64)
    return out


def layer_norm(x, w, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + EPS) * w + b


_erf = np.vectorize(math.erf)


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def embed(W, ids):
    ids = np.asarray(ids[:512])
    t = len(ids)
    h = W["embeddings.word_embeddings.weight"][ids] + W["embeddings.position_embeddings.weight"][:t] + W[
        "embeddings.token_type_embeddings.weight"
    ][0]
    h = layer_norm(h, W["embeddings.LayerNorm.weight"], W["embeddings.LayerNorm.bias"])
    for l in range(LAYERS):
        p = f"encoder.layer.{l}."

        def lin(name, x):
            return x @ W[p + name + ".weight"].T + W[p + name + ".bias"]

        q = lin("attention.self.query", h).reshape(t, HEADS, HEAD_DIM).transpose(1, 0, 2)
        k = lin("attention.self.key", h).reshape(t, HEADS, HEAD_DIM).transpose(1, 0, 2)
        v = lin("attention.self.value", h).reshape(t, HEADS, HEAD_DIM).transpose(1, 0, 2)
        s = q @ k.transpose(0, 2, 1) / math.sqrt(HEAD_DIM)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        ctx = (s @ v).transpose(1, 0, 2).reshape(t, HIDDEN)
        h = layer_norm(
            lin("attention.output.dense", ctx) + h,
            W[p + "attention.output.LayerNorm.weight"],
            W[p + "attention.output.LayerNorm.bias"],
        )
        inter = gelu(lin("intermediate.dense", h))
        h = layer_norm(lin("output.dense", inter) + h, W[p + "output.LayerNorm.weight"], W[p + "output.LayerNorm.bias"])
    cls = h[0]
    return cls / np.linalg.norm(cls)


def main():
    mode, wdir = sys.argv[1], sys.argv[2]
    W = load_safetensors(wdir + "/model.safetensors")
    if mode == "validate":
        refs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data", "reference_test_vectors.json")
        refs = json.load(open(refs_path))
        worst = 1.0
        for r in refs:
            e = embed(W, r["ids"])
            c = float(e @ np.asarray(r["embedding_fp32"]) / np.linalg.norm(r["embedding_fp32"]))
            worst = min(worst, c)
            print(f"{len(r['ids']):4d} tokens  cos={c:.8f}  {r['raw_text'][:50]!r}")
        print(f"min cos vs stored reference = {worst:.8f}")
    elif mode == "embed":
        recs = json.load(open(sys.argv[3]))
        for r in recs:
            r["embedding_fp32"] = [float(x) for x in embed(W, r["ids"])]
            print(f"{r['name']}: {len(r['ids'])} tokens", file=sys.stderr)
        json.dump(recs, open(sys.argv[4], "w"))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
