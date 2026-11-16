#!/usr/bin/env bash
# Scale-up sweep (mentor request): does T-free close the gap to per-T specialists with more data + compute?
#
# Every model continues from its OWN previous-stage weights, with identical extra steps and identical data:
#   v2      : 6k steps,  200 clips/T, DAVIS trainval (90 seq)                      -> perT_sweep_v2/
#   stage A : +6k  (12k cumulative), 2000 clips/T, dataset/merged (DAVIS all + YouTube-VOS all-frames)
#   stage B : +12k (24k cumulative), same data as A
# Each stage is a full warmup + cosine schedule, so every stage ends at a converged point.
# Clip counts per T match between T-free and specialists (n_per_T = n_train / len(T_list)); clips
# come from the same sequence draw (seed + T), with crop/start positions from the global RNG.
#
# Usage (repo root, Linux, after data_prep/build_merged.sh):
#   STAGE=A MODELS="tfree T8 T16 T24" GPUS="0 1 2 3" bash perT_sweep/run_scale.sh all
#   STAGE=A MODELS="T32" GPUS="0" bash perT_sweep/run_scale.sh all          # other machine
#   STAGE=B ... (needs stage-A weights of the same models on this machine)
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=${STAGE:?set STAGE=A or STAGE=B}
case $STAGE in
  A) INIT=perT_sweep_v2;     EPOCHS=30 ;;
  B) INIT=perT_sweep_scaleA; EPOCHS=60 ;;
  *) echo "STAGE must be A or B"; exit 1 ;;
esac
OUT=perT_sweep_scale$STAGE
PY=${PY:-python}
GPUS=(${GPUS:-0})
DATA=${DATA:-dataset/merged}
N_PER_T=${N_PER_T:-2000}
EVAL_TS="8 12 16 20 24 28 32"
MODELS=(${MODELS:-tfree T8 T16 T24 T32})
declare -A TLIST=([tfree]="8 16 24 32" [T8]=8 [T16]=16 [T24]=24 [T32]=32)

COMMON=(--epochs "$EPOCHS" --steps_per_epoch 200 --warmup_epochs 3
        --legacy_T 8 --davis_root "$DATA" --real_ratio 0.7 --seed 42
        --amp --ckpt --ckpt_blocks --vgg_weight 0.02 --log_every 50)

done_model() { [ -f "$OUT/$1/history.json" ] && grep -q '"per_T"' "$OUT/$1/history.json"; }

train_one() {
  local m=$1 gpu=$2 d=$OUT/$1
  mkdir -p "$d"
  if done_model "$m"; then echo "[skip] $m complete"; return; fi
  local nt=$(( N_PER_T * $(echo ${TLIST[$m]} | wc -w) ))
  local args=(--T_list ${TLIST[$m]} --n_train $nt --warm_start "$INIT/$m/checkpoint.pth"
              "${COMMON[@]}" --out_dir "$d")
  [ -f "$d/checkpoint.pth" ] && args+=(--resume "$d/checkpoint.pth")
  echo "$(date '+%F %T') gpu=$gpu sha256(init)=$(sha256sum "$INIT/$m/checkpoint.pth" | cut -c1-16)" >> "$d/command.txt"
  echo "$PY -u dd_madun_v3_train_tfree.py ${args[*]}" >> "$d/command.txt"
  echo "[train] $STAGE/$m on GPU $gpu -> $d/train.log"
  if CUDA_VISIBLE_DEVICES=$gpu $PY -u dd_madun_v3_train_tfree.py "${args[@]}" >> "$d/train.log" 2>&1; then
    echo "[done] $STAGE/$m"
  else
    echo "[FAIL] $STAGE/$m -- see $d/train.log; rerun to resume"
  fi
}

train() {
  local n=${#GPUS[@]}
  for ((g = 0; g < n; g++)); do
    ( for ((i = g; i < ${#MODELS[@]}; i += n)); do train_one "${MODELS[$i]}" "${GPUS[$g]}"; done ) &
  done
  wait
}

eval_one() {  # label weight gpu
  local csv=$OUT/eval/$1.csv
  [ -s "$csv" ] && { echo "[skip] eval $1"; return; }
  [ -f "$2" ] || { echo "[skip] eval $1: $2 missing"; return; }
  echo "[eval] $1 <- $2"
  CUDA_VISIBLE_DEVICES=$3 $PY -m sci_ablation.run_ablation --exp T --Ts $EVAL_TS \
    --methods tfree --ckpt "tfree=$2" --device cuda --out "$csv" > "$OUT/eval/$1.log" 2>&1
}

evaluate() {
  mkdir -p "$OUT/eval"
  local jobs=()
  for m in "${MODELS[@]}"; do
    jobs+=("${m}_nonEMA $OUT/$m/checkpoint.pth" "${m}_EMA $OUT/$m/best_model.pth")
  done
  local n=${#GPUS[@]}
  for ((g = 0; g < n; g++)); do
    ( for ((i = g; i < ${#jobs[@]}; i += n)); do eval_one ${jobs[$i]} "${GPUS[$g]}"; done ) &
  done
  wait
}

case ${1:-all} in
  train) train ;;
  eval)  evaluate ;;
  all)   train; evaluate ;;
  *) echo "usage: STAGE=A|B $0 {train|eval|all}"; exit 1 ;;
esac
