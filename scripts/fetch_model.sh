#!/usr/bin/env bash
# Fetches the pre-built Int8 SmoothQuant model pack (`model.ae8` + `vocab.txt`) from
# https://github.com/mdlockyer/snowflake-arctic-embed-s-weights, pinned by version and SHA-256.
#   usage: scripts/fetch_model.sh [dest_dir]   (default: ./model, or $ARCTIC_EMBED_MODEL_DIR)
set -euo pipefail

WEIGHTS_REPO="mdlockyer/snowflake-arctic-embed-s-weights"
WEIGHTS_TAG="v1.0.0"
DEST="${1:-${ARCTIC_EMBED_MODEL_DIR:-model}}"

FILES=(
  "model.ae8 d68b00aca78b143fb2fe3f8116eb2330d175f2a8318774725cf40bc906d564f1"
  "vocab.txt 07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
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
  echo "download  $name ($WEIGHTS_REPO@$WEIGHTS_TAG)"
  curl -fL --retry 3 --progress-bar -o "$path.part" \
    "https://github.com/$WEIGHTS_REPO/releases/download/$WEIGHTS_TAG/$name"
  got="$(sha256 "$path.part")"
  if [[ "$got" != "$want" ]]; then
    rm -f "$path.part"
    echo "error: checksum mismatch for $name (got $got, want $want)" >&2
    exit 1
  fi
  mv "$path.part" "$path"
done
echo "model ready in $DEST"
