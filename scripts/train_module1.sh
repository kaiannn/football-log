#!/usr/bin/env bash
#
# Module 1 fine-tune — one-shot orchestrator.
#
# What this script does (in order):
#   1. Sanity-check the environment (GPU, venv, raw SoccerNet data).
#   2. Dry-run the YOLO converter so you SEE the actual class IDs before
#      committing to a class map.
#   3. Convert SoccerNet → YOLO format (skips if already converted).
#   4. Run baseline eval against the stock yolov8n.pt weights.
#   5. Fine-tune yolov8s.pt on the converted dataset.
#   6. Run the same eval against the fine-tuned best.pt.
#   7. Generate REPORT.md comparing baseline vs module1_v1.
#
# Designed to be RESUMABLE — each step skips its work if the output already
# exists. Re-run after a crash without losing progress.
#
# Usage (on the cloud GPU box, inside tmux):
#
#     cd /root/work/football-log
#     bash scripts/train_module1.sh
#
# Override defaults via environment variables:
#
#     IMGSZ=640 BATCH=32 EPOCHS=30 bash scripts/train_module1.sh
#
# To skip the fine-tune and only run baseline eval (e.g. on the laptop):
#
#     SKIP_TRAIN=1 bash scripts/train_module1.sh
#

set -euo pipefail

# ---- Configuration (override via env) ----------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PYTHON="${VENV_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

# RAW_DIR auto-detect: prefer GSR-2025 (data/gsr2025/extracted) if present,
# fall back to legacy SoccerNet Tracking layout. Both formats are handled by
# scripts/prepare_yolo_dataset.py via --format auto.
if [ -z "${RAW_DIR:-}" ]; then
    if [ -d "${PROJECT_ROOT}/data/gsr2025/extracted" ]; then
        RAW_DIR="${PROJECT_ROOT}/data/gsr2025/extracted"
    elif [ -d "${PROJECT_ROOT}/data/soccernet/raw/tracking" ]; then
        RAW_DIR="${PROJECT_ROOT}/data/soccernet/raw/tracking"
    else
        RAW_DIR="${PROJECT_ROOT}/data/gsr2025/extracted"
    fi
fi
YOLO_DIR="${YOLO_DIR:-${PROJECT_ROOT}/data/soccernet}"
# CLASS_MAP auto: gsr_classes for GSR-2025, soccernet_classes for legacy MOT.
if [ -z "${CLASS_MAP:-}" ]; then
    if [[ "${RAW_DIR}" == *gsr2025* ]]; then
        CLASS_MAP="${PROJECT_ROOT}/football_log/data/gsr_classes.example.yaml"
    else
        CLASS_MAP="${PROJECT_ROOT}/football_log/data/soccernet_classes.example.yaml"
    fi
fi

BASELINE_WEIGHTS="${BASELINE_WEIGHTS:-yolov8n.pt}"
TRAIN_MODEL="${TRAIN_MODEL:-yolov8s.pt}"
IMGSZ="${IMGSZ:-1280}"
BATCH="${BATCH:-8}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
DEVICE="${DEVICE:-0}"

RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs}"
BASELINE_NAME="${BASELINE_NAME:-baseline}"
EXP_NAME="${EXP_NAME:-module1_v1}"

SPLIT_RATIOS="${SPLIT_RATIOS:-0.7,0.15,0.15}"
SEED="${SEED:-42}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

# ---- Helpers -----------------------------------------------------------------

log() { printf "\n\033[1;36m[train_module1]\033[0m %s\n" "$*"; }
warn() { printf "\n\033[1;33m[train_module1] WARN:\033[0m %s\n" "$*"; }
die() { printf "\n\033[1;31m[train_module1] ERROR:\033[0m %s\n" "$*" >&2; exit 1; }

require_file() { [ -e "$1" ] || die "missing required path: $1"; }

cd "${PROJECT_ROOT}"

# ---- Step 0: environment sanity check ----------------------------------------

log "Step 0: environment check"

[ -x "${VENV_PYTHON}" ] || die "venv python not found at ${VENV_PYTHON}. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"

"${VENV_PYTHON}" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0), '|', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB VRAM')"

if [ ! -d "${RAW_DIR}" ]; then
    warn "raw dataset directory not found: ${RAW_DIR}"
    warn "If the download is still running, wait. If it's done, set RAW_DIR to its path."
    warn "GSR-2025: extract train.zip / test.zip into data/gsr2025/extracted/"
    die "cannot proceed without the raw dataset"
fi

mkdir -p "${RUNS_DIR}/${BASELINE_NAME}" "${RUNS_DIR}/${EXP_NAME}"

# ---- Step 1: dry-run the converter -------------------------------------------

DRY_RUN_LOG="${RUNS_DIR}/${EXP_NAME}/converter_dry_run.txt"

log "Step 1: dry-run converter (output → ${DRY_RUN_LOG})"
"${VENV_PYTHON}" scripts/prepare_yolo_dataset.py \
    --source-dir "${RAW_DIR}" \
    --dry-run 2>&1 | tee "${DRY_RUN_LOG}"

log "Class IDs in source data above. Edit ${CLASS_MAP} now if they differ from the default mapping. Press ENTER to continue, Ctrl-C to abort."
if [ "${NONINTERACTIVE:-0}" = "1" ]; then
    log "  NONINTERACTIVE=1 → skipping confirmation"
else
    read -r _
fi

# ---- Step 2: convert SoccerNet → YOLO ----------------------------------------

if [ -f "${YOLO_DIR}/soccernet.yaml" ]; then
    log "Step 2: SKIPPED — converted dataset already exists at ${YOLO_DIR}/soccernet.yaml"
else
    log "Step 2: convert SoccerNet → YOLO format"
    "${VENV_PYTHON}" scripts/prepare_yolo_dataset.py \
        --source-dir "${RAW_DIR}" \
        --output-dir "${YOLO_DIR}" \
        --class-map "${CLASS_MAP}" \
        --split-ratios "${SPLIT_RATIOS}" \
        --seed "${SEED}"

    require_file "${YOLO_DIR}/soccernet.yaml"
    log "Conversion done. Manifest: ${YOLO_DIR}/manifest.json"
fi

DATA_YAML="${YOLO_DIR}/soccernet.yaml"

# ---- Step 3: baseline eval (stock COCO weights) -----------------------------

BASELINE_DET_JSON="${RUNS_DIR}/${BASELINE_NAME}/detection.json"

if [ "${SKIP_BASELINE}" = "1" ]; then
    log "Step 3: SKIPPED — SKIP_BASELINE=1"
elif [ -f "${BASELINE_DET_JSON}" ]; then
    log "Step 3: SKIPPED — baseline already evaluated at ${BASELINE_DET_JSON}"
else
    log "Step 3: baseline eval (${BASELINE_WEIGHTS})"
    # Note: COCO classes ≠ SoccerNet classes, so mAP will be near-zero on
    # SoccerNet val. That's expected — this baseline is for the *report*,
    # not for picking weights.
    "${VENV_PYTHON}" -m football_log.eval.eval_detection \
        --weights "${BASELINE_WEIGHTS}" \
        --data "${DATA_YAML}" \
        --imgsz "${IMGSZ}" \
        --device "${DEVICE}" \
        --out "${BASELINE_DET_JSON}"

    cat > "${RUNS_DIR}/${BASELINE_NAME}/meta.json" <<EOF
{
  "exp_name": "${BASELINE_NAME}",
  "weights": "${BASELINE_WEIGHTS}",
  "note": "Stock COCO yolov8n.pt — included as before-fine-tune reference.",
  "imgsz": ${IMGSZ},
  "device": "${DEVICE}",
  "git_sha": "$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
fi

# ---- Step 4: fine-tune -------------------------------------------------------

BEST_WEIGHTS="${RUNS_DIR}/${EXP_NAME}/weights/best.pt"

if [ "${SKIP_TRAIN}" = "1" ]; then
    log "Step 4: SKIPPED — SKIP_TRAIN=1"
elif [ -f "${BEST_WEIGHTS}" ]; then
    log "Step 4: SKIPPED — best.pt already exists at ${BEST_WEIGHTS}"
    log "  Delete it (or pass a different EXP_NAME) to retrain."
else
    log "Step 4: fine-tune ${TRAIN_MODEL} on ${DATA_YAML}"
    log "  imgsz=${IMGSZ} batch=${BATCH} epochs=${EPOCHS} patience=${PATIENCE} device=${DEVICE}"

    # Use the ultralytics Python API directly so we don't rely on the `yolo`
    # CLI being on PATH (it isn't, in a non-activated venv).
    "${VENV_PYTHON}" - <<PY
from ultralytics import YOLO
model = YOLO("${TRAIN_MODEL}")
model.train(
    data="${DATA_YAML}",
    imgsz=${IMGSZ},
    batch=${BATCH},
    epochs=${EPOCHS},
    patience=${PATIENCE},
    device="${DEVICE}",
    project="${RUNS_DIR}",
    name="${EXP_NAME}",
    exist_ok=True,
)
PY

    require_file "${BEST_WEIGHTS}"
    log "Training done. Best weights: ${BEST_WEIGHTS}"
fi

# ---- Step 5: post-fine-tune eval ---------------------------------------------

EXP_DET_JSON="${RUNS_DIR}/${EXP_NAME}/detection.json"

if [ ! -f "${BEST_WEIGHTS}" ]; then
    warn "Step 5: SKIPPED — no fine-tuned weights to evaluate"
elif [ -f "${EXP_DET_JSON}" ]; then
    log "Step 5: SKIPPED — fine-tuned eval already at ${EXP_DET_JSON}"
else
    log "Step 5: eval fine-tuned weights"
    "${VENV_PYTHON}" -m football_log.eval.eval_detection \
        --weights "${BEST_WEIGHTS}" \
        --data "${DATA_YAML}" \
        --imgsz "${IMGSZ}" \
        --device "${DEVICE}" \
        --out "${EXP_DET_JSON}"

    cat > "${RUNS_DIR}/${EXP_NAME}/meta.json" <<EOF
{
  "exp_name": "${EXP_NAME}",
  "weights": "${BEST_WEIGHTS}",
  "base_model": "${TRAIN_MODEL}",
  "data": "${DATA_YAML}",
  "imgsz": ${IMGSZ},
  "batch": ${BATCH},
  "epochs": ${EPOCHS},
  "device": "${DEVICE}",
  "git_sha": "$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
fi

# ---- Step 6: report ----------------------------------------------------------

log "Step 6: generate comparison report"
"${VENV_PYTHON}" -m football_log.eval.report || warn "report generator returned non-zero (likely missing inputs)"

# ---- Done --------------------------------------------------------------------

log "All steps done."
echo
echo "What to pull back to your laptop:"
echo "  scp -P 11912 root@js3.blockelite.cn:${BEST_WEIGHTS} ./runs/${EXP_NAME}/weights/"
echo "  scp -P 11912 -r root@js3.blockelite.cn:${RUNS_DIR}/${EXP_NAME} ./runs/"
echo "  scp -P 11912 -r root@js3.blockelite.cn:${RUNS_DIR}/${BASELINE_NAME} ./runs/"
echo "  scp -P 11912 ${PROJECT_ROOT}/REPORT.md ./"
echo
echo "Then on the laptop, run inference with the new weights:"
echo "  .venv/bin/python run.py --video YOUR_VIDEO.mp4 \\"
echo "      --model runs/${EXP_NAME}/weights/best.pt \\"
echo "      --player-class-id 0 --ball-class-id 1 --referee-class-id 2 \\"
echo "      --no-ui"
