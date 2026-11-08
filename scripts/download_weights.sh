#!/usr/bin/env bash
# Downloads the original FP32 Snowflake/snowflake-arctic-embed-s checkpoint, pinned to a fixed Hub
# revision and verified by SHA-256. Only needed to rebuild model/model.ae8 (feature `convert`) or
# run scripts/fp32_reference.py; inference uses the pack committed in model/.
#   usage: scripts/download_weights.sh [dest_dir]   (default: ./weights, or $ARCTIC_EMBED_SOURCE_WEIGHTS)
set -euo pipefail

REPO="Snowflake/snowflake-arctic-embed-s"
REVISION="e596f507467533e48a2e17c007f0e1dacc837b33"
DEST="${1:-${ARCTIC_EMBED_SOURCE_WEIGHTS:-weights}}"

# file  sha256
FILES=(
  "model.safetensors 0350e62666ee2403db0223fc7cef6293c951cc63d1d2e71468da9a5aea3b544a"
  "vocab.txt 07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
  "config.json 4e519aa92ec40943356032afe458c8829d70c5766b109e4a57490b82f72dcfb7"
)

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

mkdir -p "$DEST"
for entry in "${FILES[@]}"; do
  read -r name want <<<"$entry"
  path="$DEST/$name"
  if [[ -f "$path" && "$(sha256 "$path")" == "$want" ]]; then
    echo "ok        $path"
    continue
  fi
  echo "download  $name"
  curl -fL --retry 3 --progress-bar -o "$path.part" "https://huggingface.co/$REPO/resolve/$REVISION/$name"
  got="$(sha256 "$path.part")"
  if [[ "$got" != "$want" ]]; then
    rm -f "$path.part"
    echo "error: checksum mismatch for $name (got $got, want $want)" >&2
    exit 1
  fi
  mv "$path.part" "$path"
done
echo "weights ready in $DEST"
