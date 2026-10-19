#!/usr/bin/env bash
# Fetch AMI meeting recordings for local benchmarking into test-audio/.
#
# The AMI meetings are recordings of real meetings distributed under the
# AMI corpus license terms; they are NOT committed to this repo (see
# .gitignore). This script re-fetches the standard benchmark meetings
# used in the diarization benchmarks (ES2004a: the 17.5-min AMI meeting
# used in coro's /v1/diarize benchmarks; IS1000a/b, ES2002a/b, ...: the
# multi-meeting set).
#
# Usage: bash scripts/fetch_ami_test_audio.sh [meeting-id ...]
#        (no args = the default set below)
set -euo pipefail

# Default benchmark set (AMI meeting ids, Mix-Headset channel).
DEFAULTS=(
    ES2004a ES2002a ES2002b ES2003a ES2003b ES2004b
    IS1001b IS1000a IS1000b
)

BASE_URL="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus"
OUT_DIR="$(dirname "$0")/../test-audio/ami"
mkdir -p "$OUT_DIR"

MEETINGS=("$@")
[ "${#MEETINGS[@]}" -eq 0 ] && MEETINGS=("${DEFAULTS[@]}")

for m in "${MEETINGS[@]}"; do
    url="$BASE_URL/$m/audio/$m.Mix-Headset.wav"
    out="$OUT_DIR/$m.Mix-Headset.wav"
    if [ -f "$out" ]; then
        echo "have   $m"
        continue
    fi
    echo "fetch  $m"
    curl -fL --retry 3 -o "$out" "$url"
done

echo
du -sh "$OUT_DIR"
echo "Done. Benchmark with e.g.:"
echo "  cargo run --release -- diarize $OUT_DIR/ES2004a.Mix-Headset.wav --mode cuda-fast"
