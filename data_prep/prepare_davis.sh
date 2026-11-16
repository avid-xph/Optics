#!/usr/bin/env bash
# DAVIS 2017 480p: trainval (90 seq) + test-dev (30) + test-challenge (30), 24 fps.
#   dataset/DAVIS/JPEGImages/480p/<seq>                         (trainval; used by v2)
#   dataset/DAVIS_test-<split>/DAVIS/JPEGImages/480p/<seq>      (extra sequences for the scale-up)
# We never evaluate on DAVIS, so its test splits are plain training video here.
# Usage (repo root): bash data_prep/prepare_davis.sh
set -euo pipefail
cd "$(dirname "$0")/.." && mkdir -p dataset && cd dataset
URL=https://data.vision.ee.ethz.ch/csergi/share/davis
command -v aria2c >/dev/null || { apt-get update -qq && apt-get install -y -qq aria2 >/dev/null; }

get() {  # zip_name dest_dir
  [ -d "$2/JPEGImages/480p" ] && { echo "[skip] $2"; return; }
  aria2c -q -x16 -s16 -k4M -o "$1" "$URL/$1"
  mkdir -p "$3" && unzip -q -o "$1" -d "$3" && rm -f "$1"
  echo "[done] $2: $(ls "$2/JPEGImages/480p" | wc -l) sequences"
}

get DAVIS-2017-trainval-480p.zip      DAVIS                      .
get DAVIS-2017-test-dev-480p.zip      DAVIS_test-dev/DAVIS       DAVIS_test-dev
get DAVIS-2017-test-challenge-480p.zip DAVIS_test-challenge/DAVIS DAVIS_test-challenge
