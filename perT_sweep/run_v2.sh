#!/usr/bin/env bash
# Per-T sweep v2: T-free vs 4 per-T specialists, converged AND data-matched.
#
# Fixes the two confounds of v1 (results/ablation/RESULTS_perT_sweep_20261107.md):
#   1. v1 trained 1500 steps -> EMA unconverged. v2: 6000 steps for every model
#      (= the published T-free run, output_tfree / checkpoints/tfree_trained).
#   2. v1 gave the T=8 specialist 4x the clips of T-free's T=8 share. v2: every
#      model sees exactly 200 clips per T (create_mixed_dataset seed = seed+T,
#      so specialist T and T-free's T split are the SAME clips).
# Steps are fixed by --steps_per_epoch, not n_train, so compute is identical.
#
# Usage (repo root, Linux):
#   GPUS="0 1 2 3 4" bash perT_sweep/run_v2.sh train   # models round-robin over GPUs
#   GPUS="0" bash perT_sweep/run_v2.sh eval
#   bash perT_sweep/run_v2.sh plot
#   GPUS="0 1 2 3 4" bash perT_sweep/run_v2.sh all
# Re-running `train` resumes every unfinished model from its checkpoint.pth.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=perT_sweep_v2
PY=${PY:-python}
GPUS=(${GPUS:-0})
DAVIS=${DAVIS:-DAVIS}
EVAL_TS="8 12 16 20 24 28 32"     # 12/20/28: no specialist exists -> T-free only
MODELS=(tfree T8 T16 T24 T32)     # tfree first: it is the reference row
declare -A TLIST=([tfree]="8 16 24 32" [T8]=8 [T16]=16 [T24]=24 [T32]=32)
declare -A NTRAIN=([tfree]=800 [T8]=200 [T16]=200 [T24]=200 [T32]=200)

COMMON=(--epochs 30 --steps_per_epoch 200 --warmup_epochs 3
        --warm_start checkpoints/best_model.pth --legacy_T 8
        --davis_root "$DAVIS" --real_ratio 0.7 --seed 42
        --amp --ckpt --ckpt_blocks --vgg_weight 0.02 --log_every 20)

done_model() {  # history.json is written once, after the final per-T eval
  [ -f "$OUT/$1/history.json" ] && grep -q '"per_T"' "$OUT/$1/history.json"
}

train_one() {
  local m=$1 gpu=$2 d=$OUT/$1
  mkdir -p "$d"
  if done_model "$m"; then echo "[skip] $m complete"; return; fi
  local args=(--T_list ${TLIST[$m]} --n_train ${NTRAIN[$m]} "${COMMON[@]}" --out_dir "$d")
  [ -f "$d/checkpoint.pth" ] && args+=(--resume "$d/checkpoint.pth")
  echo "$(date '+%F %T') gpu=$gpu $PY -u dd_madun_v3_train_tfree.py ${args[*]}" >> "$d/command.txt"
  echo "[train] $m on GPU $gpu -> $d/train.log"
  if CUDA_VISIBLE_DEVICES=$gpu $PY -u dd_madun_v3_train_tfree.py "${args[@]}" >> "$d/train.log" 2>&1; then
    echo "[done] $m"
  else
    echo "[FAIL] $m -- see $d/train.log; rerun 'train' to resume"
  fi
}

train() {
  local n=${#GPUS[@]}
  for ((g = 0; g < n; g++)); do
    (
      for ((i = g; i < ${#MODELS[@]}; i += n)); do
        train_one "${MODELS[$i]}" "${GPUS[$g]}"
      done
    ) &
  done
  wait
}

eval_one() {  # label weight_path gpu
  local csv=$OUT/eval/$1.csv
  [ -s "$csv" ] && { echo "[skip] eval $1"; return; }
  echo "[eval] $1 <- $2"
  CUDA_VISIBLE_DEVICES=$3 $PY -m sci_ablation.run_ablation --exp T --Ts $EVAL_TS \
    --methods tfree --ckpt "tfree=$2" --device cuda --out "$csv" \
    > "$OUT/eval/$1.log" 2>&1
}

evaluate() {
  mkdir -p "$OUT/eval"
  local jobs=()
  for m in "${MODELS[@]}"; do
    jobs+=("${m}_nonEMA $OUT/$m/checkpoint.pth" "${m}_EMA $OUT/$m/best_model.pth")
  done
  # published T-free weights (sha256 48d69188...), reference only
  jobs+=("published_tfree_nonEMA checkpoints/tfree_trained/checkpoint.pth"
         "published_tfree_EMA checkpoints/tfree_trained/best_model.pth")
  local n=${#GPUS[@]}
  for ((g = 0; g < n; g++)); do
    (
      for ((i = g; i < ${#jobs[@]}; i += n)); do
        eval_one ${jobs[$i]} "${GPUS[$g]}"
      done
    ) &
  done
  wait
}

plot() {
  $PY report/plot_perT.py --eval_dir "$OUT/eval" --out report/figures/perT_v2
}

case ${1:-all} in
  train) train ;;
  eval)  evaluate ;;
  plot)  plot ;;
  all)   train; evaluate; plot ;;
  *) echo "usage: $0 {train|eval|plot|all}"; exit 1 ;;
esac
