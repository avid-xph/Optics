#!/usr/bin/env bash
# Merge every prepared video source into one DAVIS-layout root via symlinks (no copies):
#   dataset/merged/JPEGImages/480p/<source>_<sequence> -> real sequence folder
# Then train with --davis_root dataset/merged.
#
# Sources (each is used only if present):
#   DAVIS 2017 trainval / test-dev / test-challenge (24 fps)
#   YouTube-VOS 2019 all-frames valid + test (30 fps), from data_prep/prepare_youtubevos.sh
set -euo pipefail
cd "$(dirname "$0")/../dataset"
OUT=merged/JPEGImages/480p
rm -rf merged && mkdir -p "$OUT"

link_all() {  # prefix src_dir
  [ -d "$2" ] || { echo "[absent] $2"; return; }
  local n=0
  for s in "$2"/*/; do
    ln -s "$(realpath "$s")" "$OUT/$1_$(basename "$s")"; n=$((n + 1))
  done
  echo "[linked] $1: $n sequences from $2"
}

link_all davis      DAVIS/JPEGImages/480p
link_all davisdev   DAVIS_test-dev/DAVIS/JPEGImages/480p
link_all davischal  DAVIS_test-challenge/DAVIS/JPEGImages/480p
link_all yt         YouTubeVOS/JPEGImages/480p
echo "[total] $(ls "$OUT" | wc -l) sequences in dataset/merged"
