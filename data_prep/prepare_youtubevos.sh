#!/usr/bin/env bash
# YouTube-VOS 2019 ALL-FRAMES (30 fps) valid + test -> dataset/YouTubeVOS/JPEGImages/480p/<split>_<video>/*.jpg
#
# Why all-frames valid/test and not train: train.zip keeps only every 5th frame (6 fps), i.e. 4-5x the
# inter-frame motion of DAVIS (24 fps); a T=32 snapshot would span >5 s. The all-frames splits keep
# full temporal density. We never evaluate on YouTube-VOS, so using its valid/test frames for training is fine.
#
# Source: Hugging Face mirror yinloonga/YouTube_VOS_2019 (no login). Frames are resized to height 480
# (DAVIS 480p scale) so random 256x256 crops cover the same field of view as DAVIS crops.
# Layout matches DAVIS, so video_dataset.py loads it with --davis_root dataset/YouTubeVOS.
#
# Usage (repo root):  bash data_prep/prepare_youtubevos.sh            # both splits
#                     SPLITS="valid" bash data_prep/prepare_youtubevos.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=dataset/YouTubeVOS
RAW=$ROOT/raw
HF=https://huggingface.co/datasets/yinloonga/YouTube_VOS_2019/resolve/main
SPLITS=${SPLITS:-valid test}
mkdir -p "$RAW" "$ROOT/JPEGImages/480p"
command -v 7z >/dev/null && command -v aria2c >/dev/null || { apt-get update -qq && apt-get install -y -qq p7zip-full aria2 >/dev/null; }

# sha256 of each volume, from the Hugging Face API (LFS oid)
declare -A SHA=(
  [valid.001]=77c515a350d1847f72eedf2f5cb2ff4bed63a01b2a4b14f74f59e3487ece2811
  [valid.002]=8fe271a7f7f0354fd805c03eda5d30e8bd13f45278f5be715d04065a3a3edf1e
  [test.001]=807f3c9d5d4a2e9b1a060cce75f7313a05ddcadada04697b212941ed8a565076
  [test.002]=775274f9530a327c0a25a2eb197b8eaedf816506b107f4abecc9c45a0abcab52
)

for s in $SPLITS; do
  if [ -f "$ROOT/$s.done" ]; then echo "[skip] $s"; continue; fi
  # $RAW/$s exists once the verified volumes were extracted (the volumes are then deleted)
  [ -d "$RAW/$s" ] || for part in 001 002; do
    f=$RAW/${s}_all_frames.7z.$part
    for attempt in 1 2 3; do
      [ -f "$f" ] && [ ! -f "$f.aria2" ] && [ "$(sha256sum "$f" | cut -d' ' -f1)" = "${SHA[$s.$part]}" ] && break
      echo "[download] $f (attempt $attempt)"
      rm -f "$f" "$f.aria2"
      aria2c -q -x16 -s16 -k4M -d "$RAW" -o "$(basename "$f")" "$HF/${s}_all_frames_zip/${s}_all_frames.7z.$part" || true
    done
    [ "$(sha256sum "$f" | cut -d' ' -f1)" = "${SHA[$s.$part]}" ] || { echo "[FAIL] $f checksum"; exit 1; }
    echo "[ok] $f sha256 verified"
  done
  if [ ! -d "$RAW/$s" ]; then
    echo "[extract] $s"
    7z x -y -bd -o"$RAW/$s.part" "$RAW/${s}_all_frames.7z.001" >/dev/null
    mv "$RAW/$s.part" "$RAW/$s" && rm -f "$RAW/${s}_all_frames.7z."*
  fi
  # the 7z volumes wrap a single .zip of the frame folders
  for z in $(find "$RAW/$s" -name '*.zip'); do
    echo "[unzip] $z"; unzip -q -o "$z" -d "$(dirname "$z")" && rm -f "$z"
  done
  echo "[layout] $s: $(find "$RAW/$s" -type f | wc -l) files; extensions:"
  # (|| true: head closing the pipe early must not trip set -o pipefail)
  find "$RAW/$s" -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn | head -5 || true
  find "$RAW/$s" -maxdepth 3 | head -8 || true
  echo "[resize] $s -> 480p"
  python data_prep/resize_to_480p.py "$RAW/$s" "$ROOT/JPEGImages/480p" "$s"
  n=$(ls "$ROOT/JPEGImages/480p" | grep -c "^${s}_" || true)
  # keep the archives and extracted files when nothing came out, so the cause can be inspected
  [ "$n" -gt 0 ] || { echo "[FAIL] $s produced 0 videos; raw data kept in $RAW"; exit 1; }
  rm -rf "$RAW/$s"
  touch "$ROOT/$s.done"
  echo "[done] $s: $(ls "$ROOT/JPEGImages/480p" | grep -c "^${s}_") videos"
done
