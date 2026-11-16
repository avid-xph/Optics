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

for s in $SPLITS; do
  if [ -f "$ROOT/$s.done" ]; then echo "[skip] $s"; continue; fi
  for part in 001 002; do
    f=$RAW/${s}_all_frames.7z.$part
    echo "[download] $f"
    aria2c -q -c -x16 -s16 -k4M -d "$RAW" -o "$(basename "$f")" "$HF/${s}_all_frames_zip/${s}_all_frames.7z.$part"
  done
  echo "[extract] $s"
  7z x -y -bd -o"$RAW/$s" "$RAW/${s}_all_frames.7z.001" >/dev/null
  rm -f "$RAW/${s}_all_frames.7z."*
  echo "[resize] $s -> 480p"
  python data_prep/resize_to_480p.py "$RAW/$s" "$ROOT/JPEGImages/480p" "$s"
  rm -rf "$RAW/$s"
  touch "$ROOT/$s.done"
  echo "[done] $s: $(ls "$ROOT/JPEGImages/480p" | grep -c "^${s}_") videos"
done
