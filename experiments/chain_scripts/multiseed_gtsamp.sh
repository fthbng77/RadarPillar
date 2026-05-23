#!/bin/bash
# Multi-seed ablation A1: gt_sampling ON (vod_radarpillar_rot_gtsamp.yaml) — 3 sequential runs.
# Goal: gt_sampling pedestrian-AP effect with mean +/- std, comparable to the
#       paper_faithful_rot baseline 3-seed runs (s1/s2/s3).
# Note: FIX_RANDOM_SEED=False -> each run is a random-init draw (NOT fixed seeds 1/2/3);
#       the loop variable only names the run tag, matching multiseed_v2.sh methodology.
#
# Detached: setsid + nohup + disown. Survives lid close via systemd-inhibit.
# Each run: ~2.5h. Total: ~7.5h.

set -u

REPO="/home/fatih/Xena_Vision/repo/RadarPillar"
CFG="${REPO}/tools/cfgs/vod_models/vod_radarpillar_rot_gtsamp.yaml"
TAG_PREFIX="rot_gtsamp_s"
LOG_DIR="${REPO}/experiments/logs"
CHAIN_LOG="${LOG_DIR}/multiseed_gtsamp_chain.log"

mkdir -p "$LOG_DIR"

echo "[$(date)] gtsamp multi-seed chain started" >> "$CHAIN_LOG"

# Renew suspend-inhibit (8h covers all 3 runs).
setsid nohup systemd-inhibit --what=sleep:idle:handle-lid-switch \
  --who="rp_gtsamp" --why="3-seed gt_sampling training" --mode=block \
  sleep 28800 > /dev/null 2>&1 < /dev/null &
disown
echo "[$(date)] suspend-inhibit armed for 8h" >> "$CHAIN_LOG"

for SEED in 1 2 3; do
  TAG="${TAG_PREFIX}${SEED}"
  LOG_FILE="${LOG_DIR}/${TAG}.log"
  echo "[$(date)] starting run $TAG (FIX_RANDOM_SEED=False random init)" >> "$CHAIN_LOG"

  cd "$REPO"
  CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --cfg_file "$CFG" \
    --batch_size 8 \
    --extra_tag "$TAG" \
    --workers 4 \
    >> "$LOG_FILE" 2>&1
  EXIT=$?
  echo "[$(date)] $TAG finished, exit=$EXIT" >> "$CHAIN_LOG"

  sleep 30
done

echo "[$(date)] gtsamp multi-seed chain complete" >> "$CHAIN_LOG"
