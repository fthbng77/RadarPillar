#!/usr/bin/env bash
set -uo pipefail
export LC_ALL=C
export LANG=C

GPU="0"
EPOCHS="40"
BATCH_SIZE="16"
WORKERS="8"
EXTRA_TAG="ablation_vod"
MODE="all"  # train | eval | all
USE_WANDB="false"
CONTINUE_ON_ERROR="true"
SKIP_EXISTING="true"
CFG_LIST_FILE=""

usage() {
    cat <<USAGE
Usage: bash tools/run_vod_ablation.sh [options]

Options:
  --gpu <id>            CUDA device id (default: 0)
  --epochs <n>          Number of epochs (default: 40)
  --batch-size <n>      Batch size per run (default: 16)
  --workers <n>         Dataloader workers (default: 8)
  --extra-tag <tag>     Output experiment tag (default: ablation_vod)
  --mode <train|eval|all>  Run mode (default: all)
  --cfg-list-file <path>   Optional text file with config paths (one per line)
  --use-wandb           Enable Weights & Biases logging for training runs
  --stop-on-error       Stop script immediately when a config fails
  --no-skip-existing    Force retrain even if checkpoint_best.pth already exists

Examples:
  bash tools/run_vod_ablation.sh --gpu 0 --mode all --extra-tag ablation_v1 --use-wandb
  bash tools/run_vod_ablation.sh --gpu 0 --mode all --cfg-list-file tools/cfgs/vod_models/ablation_sets/vod_box_quality_ablation.txt --extra-tag vod_boxq_v1 --use-wandb
  bash tools/run_vod_ablation.sh --gpu 0 --mode all --extra-tag ablation_v1 --stop-on-error
  bash tools/run_vod_ablation.sh --gpu 0 --mode eval --extra-tag ablation_v1
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --extra-tag)
            EXTRA_TAG="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --cfg-list-file)
            CFG_LIST_FILE="$2"
            shift 2
            ;;
        --use-wandb)
            USE_WANDB="true"
            shift 1
            ;;
        --stop-on-error)
            CONTINUE_ON_ERROR="false"
            shift 1
            ;;
        --no-skip-existing)
            SKIP_EXISTING="false"
            shift 1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ "$MODE" != "train" && "$MODE" != "eval" && "$MODE" != "all" ]]; then
    echo "Invalid --mode: $MODE"
    usage
    exit 1
fi

CFG_LIST=(
    "tools/cfgs/vod_models/vod_radarpillar.yaml"
    "tools/cfgs/vod_models/vod_radarpillar_noflip.yaml"
    "tools/cfgs/vod_models/vod_radarpillar_voxel016.yaml"
    "tools/cfgs/vod_models/vod_radarpillar_voxel016_noflip.yaml"
)

if [[ -n "$CFG_LIST_FILE" ]]; then
    if [[ ! -f "$CFG_LIST_FILE" ]]; then
        echo "CFG list file not found: $CFG_LIST_FILE"
        exit 1
    fi

    mapfile -t CFG_LIST < <(sed -E 's/[[:space:]]*#.*$//' "$CFG_LIST_FILE" | awk 'NF > 0')
    if [[ "${#CFG_LIST[@]}" -eq 0 ]]; then
        echo "CFG list file is empty: $CFG_LIST_FILE"
        exit 1
    fi
fi

echo "Running VoD ablation with mode=$MODE gpu=$GPU epochs=$EPOCHS batch=$BATCH_SIZE workers=$WORKERS tag=$EXTRA_TAG use_wandb=$USE_WANDB continue_on_error=$CONTINUE_ON_ERROR skip_existing=$SKIP_EXISTING cfg_count=${#CFG_LIST[@]}"

mkdir -p output
RESULTS_CSV="output/ablation_summary_${EXTRA_TAG}.csv"
RUN_LOG_DIR="output/ablation_logs/${EXTRA_TAG}"
mkdir -p "$RUN_LOG_DIR"
echo "config_tag,train_status,eval_status,car_3d_moderate_r40,pedestrian_3d_moderate_r40,cyclist_3d_moderate_r40,weighted_score,ckpt,error" > "$RESULTS_CSV"

run_and_log() {
    local log_file="$1"
    shift
    "$@" 2>&1 | tee -a "$log_file"
    return "${PIPESTATUS[0]}"
}

for CFG in "${CFG_LIST[@]}"; do
    CFG_TAG="$(basename "$CFG" .yaml)"
    RUN_LOG="${RUN_LOG_DIR}/${CFG_TAG}.log"
    : > "$RUN_LOG"
    TRAIN_STATUS="not_run"
    EVAL_STATUS="not_run"
    CAR_MR40=""
    PED_MR40=""
    CYC_MR40=""
    WEIGHTED_SCORE=""
    BEST_CKPT=""
    ERROR_MSG=""

    echo ""
    echo "==================== $CFG_TAG ===================="
    echo "[$(date '+%F %T')] Start config=$CFG_TAG" | tee -a "$RUN_LOG"

    if [[ ! -f "$CFG" ]]; then
        ERROR_MSG="cfg_not_found:$CFG"
        printf "%s,%s,%s,,,,,,%s\n" "$CFG_TAG" "$TRAIN_STATUS" "$EVAL_STATUS" "$ERROR_MSG" >> "$RESULTS_CSV"
        echo "[ERROR] Config not found: $CFG" | tee -a "$RUN_LOG"
        echo "[$(date '+%F %T')] End config=$CFG_TAG train_status=$TRAIN_STATUS eval_status=$EVAL_STATUS" | tee -a "$RUN_LOG"
        if [[ "$CONTINUE_ON_ERROR" == "false" ]]; then
            echo "[FATAL] stop-on-error enabled. Exiting."
            exit 1
        fi
        continue
    fi

    CKPT_DIR="output/cfgs/vod_models/$CFG_TAG/$EXTRA_TAG/ckpt"
    BEST_CKPT="$CKPT_DIR/checkpoint_best.pth"

    if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
        if [[ "$SKIP_EXISTING" == "true" && -f "$BEST_CKPT" ]]; then
            TRAIN_STATUS="skipped_existing"
            echo "[INFO] Existing checkpoint found, skip train: $BEST_CKPT" | tee -a "$RUN_LOG"
        else
            TRAIN_CMD=(
                python tools/train.py
                --cfg_file "$CFG"
                --extra_tag "$EXTRA_TAG"
                --fix_random_seed
                --epochs "$EPOCHS"
                --batch_size "$BATCH_SIZE"
                --workers "$WORKERS"
            )
            if [[ "$USE_WANDB" == "true" ]]; then
                TRAIN_CMD+=(--use_wandb)
            fi

            echo "[INFO] Train command: CUDA_VISIBLE_DEVICES=$GPU ${TRAIN_CMD[*]}" | tee -a "$RUN_LOG"
            if run_and_log "$RUN_LOG" env CUDA_VISIBLE_DEVICES="$GPU" "${TRAIN_CMD[@]}"; then
                TRAIN_STATUS="ok"
            else
                rc=$?
                TRAIN_STATUS="failed($rc)"
                ERROR_MSG="train_failed_rc_${rc}"
                echo "[ERROR] Train failed for $CFG_TAG with rc=$rc" | tee -a "$RUN_LOG"
                if [[ "$CONTINUE_ON_ERROR" == "false" ]]; then
                    printf "%s,%s,%s,,,,,,%s\n" "$CFG_TAG" "$TRAIN_STATUS" "$EVAL_STATUS" "$ERROR_MSG" >> "$RESULTS_CSV"
                    echo "[FATAL] stop-on-error enabled. Exiting."
                    exit "$rc"
                fi
            fi
        fi
    fi

    if [[ "$MODE" == "eval" || "$MODE" == "all" ]]; then
        if [[ ! -f "$BEST_CKPT" ]]; then
            LATEST_CKPT="$(ls -t "$CKPT_DIR"/checkpoint_epoch_*.pth 2>/dev/null | head -n 1 || true)"
            if [[ -z "$LATEST_CKPT" ]]; then
                echo "[WARN] No checkpoint found for $CFG_TAG in $CKPT_DIR"
                EVAL_STATUS="missing_ckpt"
                if [[ -z "$ERROR_MSG" ]]; then
                    ERROR_MSG="missing_ckpt"
                else
                    ERROR_MSG="${ERROR_MSG};missing_ckpt"
                fi
                printf "%s,%s,%s,,,,,,%s\n" "$CFG_TAG" "$TRAIN_STATUS" "$EVAL_STATUS" "$ERROR_MSG" >> "$RESULTS_CSV"
                echo "[$(date '+%F %T')] End config=$CFG_TAG train_status=$TRAIN_STATUS eval_status=$EVAL_STATUS" | tee -a "$RUN_LOG"
                continue
            fi
            BEST_CKPT="$LATEST_CKPT"
            echo "[WARN] checkpoint_best.pth not found, using latest: $BEST_CKPT"
        fi

        EVAL_CMD=(
            python tools/test.py
            --cfg_file "$CFG"
            --extra_tag "$EXTRA_TAG"
            --ckpt "$BEST_CKPT"
            --eval_tag best_ckpt
            --workers "$WORKERS"
        )
        echo "[INFO] Eval command: CUDA_VISIBLE_DEVICES=$GPU ${EVAL_CMD[*]}" | tee -a "$RUN_LOG"
        if run_and_log "$RUN_LOG" env CUDA_VISIBLE_DEVICES="$GPU" "${EVAL_CMD[@]}"; then
            EVAL_STATUS="ok"
        else
            rc=$?
            EVAL_STATUS="failed($rc)"
            if [[ -z "$ERROR_MSG" ]]; then
                ERROR_MSG="eval_failed_rc_${rc}"
            else
                ERROR_MSG="${ERROR_MSG};eval_failed_rc_${rc}"
            fi
            echo "[ERROR] Eval failed for $CFG_TAG with rc=$rc" | tee -a "$RUN_LOG"
            if [[ "$CONTINUE_ON_ERROR" == "false" ]]; then
                printf "%s,%s,%s,,,,,%s,%s\n" "$CFG_TAG" "$TRAIN_STATUS" "$EVAL_STATUS" "$BEST_CKPT" "$ERROR_MSG" >> "$RESULTS_CSV"
                echo "[FATAL] stop-on-error enabled. Exiting."
                exit "$rc"
            fi
        fi

        EVAL_LOG="$(ls -t output/cfgs/vod_models/$CFG_TAG/$EXTRA_TAG/eval/epoch_*/val/best_ckpt/log_eval_*.txt 2>/dev/null | head -n 1 || true)"
        if [[ -z "$EVAL_LOG" ]]; then
            EVAL_LOG="$(ls -t output/cfgs/vod_models/$CFG_TAG/$EXTRA_TAG/eval/epoch_no_number/val/best_ckpt/log_eval_*.txt 2>/dev/null | head -n 1 || true)"
        fi

        if [[ -n "$EVAL_LOG" ]]; then
            read -r CAR_MR40 PED_MR40 CYC_MR40 <<< "$(awk '
                /Car AP_R40@/ {cls="car"}
                /Pedestrian AP_R40@/ {cls="ped"}
                /Cyclist AP_R40@/ {cls="cyc"}
                /^3d   AP:/ && cls != "" {
                    split($0, a, ":")
                    gsub(/ /, "", a[2])
                    split(a[2], v, ",")
                    val = (length(v) >= 2) ? v[2] + 0.0 : 0.0
                    if (cls == "car") car = val
                    if (cls == "ped") ped = val
                    if (cls == "cyc") cyc = val
                    cls = ""
                }
                END {
                    if (car == "") car = 0.0
                    if (ped == "") ped = 0.0
                    if (cyc == "") cyc = 0.0
                    printf "%.6f %.6f %.6f", car, ped, cyc
                }
            ' "$EVAL_LOG")"

            WEIGHTED_SCORE="$(awk -v c="$CAR_MR40" -v p="$PED_MR40" -v y="$CYC_MR40" 'BEGIN{printf "%.6f", 0.2*c + 0.3*p + 0.5*y}')"
            printf "%s,%s,%s,%.6f,%.6f,%.6f,%.6f,%s,%s\n" \
                "$CFG_TAG" "$TRAIN_STATUS" "$EVAL_STATUS" "$CAR_MR40" "$PED_MR40" "$CYC_MR40" "$WEIGHTED_SCORE" "$BEST_CKPT" "$ERROR_MSG" >> "$RESULTS_CSV"

            echo "[Summary] $CFG_TAG"
            awk '
                /Car AP_R40@/ {cls="Car"}
                /Pedestrian AP_R40@/ {cls="Pedestrian"}
                /Cyclist AP_R40@/ {cls="Cyclist"}
                /^3d   AP:/ && cls != "" {
                    split($0, a, ":")
                    gsub(/ /, "", a[2])
                    split(a[2], v, ",")
                    if (length(v) >= 2) {
                        printf "  %s_3d/moderate_R40 = %.4f\n", cls, v[2]
                    }
                    cls = ""
                }
            ' "$EVAL_LOG"
        else
            echo "[WARN] Evaluation log not found for $CFG_TAG"
            if [[ -z "$ERROR_MSG" ]]; then
                ERROR_MSG="eval_log_not_found"
            else
                ERROR_MSG="${ERROR_MSG};eval_log_not_found"
            fi
            printf "%s,%s,%s,,,,,%s,%s\n" \
                "$CFG_TAG" "$TRAIN_STATUS" "$EVAL_STATUS" "$BEST_CKPT" "$ERROR_MSG" >> "$RESULTS_CSV"
        fi
    else
        printf "%s,%s,%s,,,,,,%s\n" "$CFG_TAG" "$TRAIN_STATUS" "$EVAL_STATUS" "$ERROR_MSG" >> "$RESULTS_CSV"
    fi
    echo "[$(date '+%F %T')] End config=$CFG_TAG train_status=$TRAIN_STATUS eval_status=$EVAL_STATUS" | tee -a "$RUN_LOG"
done

echo ""
echo "CSV summary: $RESULTS_CSV"
if [[ -f "$RESULTS_CSV" ]]; then
    echo "Cyclist leaderboard (desc):"
    {
        head -n 1 "$RESULTS_CSV"
        tail -n +2 "$RESULTS_CSV" | sort -t',' -k6,6gr
    } | column -s',' -t || cat "$RESULTS_CSV"
fi
echo ""
echo "Ablation run finished."
