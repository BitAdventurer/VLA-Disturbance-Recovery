#!/usr/bin/env bash
# Download gs://openpi-assets/checkpoints/<name> over public HTTPS into openpi's cache
# layout ($OPENPI_DATA_HOME/openpi-assets/checkpoints/<name>) and verify every MD5.
set -euo pipefail
source ${VLA_ENV:-$HOME/disk2/vla-env}/env.sh
NAME=${1:-pi05_libero}
DEST=$OPENPI_DATA_HOME/openpi-assets/checkpoints/$NAME
LIST=$VLA_ENV/tmp/$NAME.list.json
TSV=$VLA_ENV/tmp/$NAME.tsv
mkdir -p "$DEST"

curl -fsS "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/$NAME/&fields=items(name,size,md5Hash)&maxResults=1000" -o "$LIST"
python3 - "$LIST" "$DEST" "$NAME" > "$TSV" <<'PY'
import json, sys
items = json.load(open(sys.argv[1]))["items"]
prefix = "checkpoints/" + sys.argv[3] + "/"
for i in items:
    if i["name"].endswith("/"):
        continue
    rel = i["name"][len(prefix):]
    print("\t".join([i["name"], sys.argv[2] + "/" + rel, i["size"], i.get("md5Hash", "-")]))
PY
echo "files: $(wc -l < "$TSV")  total: $(awk -F'\t' '{s+=$3} END {printf "%.2f GB", s/1e9}' "$TSV")"

fetch() {
  local name=$1 path=$2 size=$3
  mkdir -p "$(dirname "$path")"
  if [ -f "$path" ] && [ "$(stat -c %s "$path")" = "$size" ]; then return 0; fi
  curl -fsS --retry 5 --retry-delay 3 -C - "https://storage.googleapis.com/openpi-assets/$name" -o "$path"
}
export -f fetch
cut -f1-3 "$TSV" | xargs -P 8 -L 1 bash -c 'fetch "$0" "$1" "$2"'

bad=0
while IFS=$'\t' read -r name path size md5; do
  # Composite GCS objects carry no MD5 (crc32c only); fall back to the exact size for those.
  if [ "$md5" = "-" ]; then
    if [ "$(stat -c %s "$path")" != "$size" ]; then echo "SIZE MISMATCH: $path"; bad=1; fi
  else
    got=$(openssl dgst -md5 -binary "$path" | base64)
    if [ "$got" != "$md5" ]; then echo "MD5 MISMATCH: $path"; bad=1; fi
  fi
done < "$TSV"
if [ $bad = 0 ]; then echo "ALL CHECKS OK"; du -sh "$DEST"; else exit 1; fi
