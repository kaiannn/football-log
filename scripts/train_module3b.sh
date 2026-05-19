#!/usr/bin/env bash
#
# Module 3B fine-tune — team-as-class detector.
#
# Trains a 6-class YOLOv8 model that encodes team identity directly:
#   0 = team_a_player   (field players, left team)
#   1 = team_b_player   (field players, right team)
#   2 = goalkeeper_a    (goalkeeper, left team)
#   3 = goalkeeper_b    (goalkeeper, right team)
#   4 = referee
#   5 = ball
#
# The trained weights replace the HSV / keypoint team classifier entirely:
#   python run.py --video match.mp4 \
#       --team-class-model runs/module3b_v1/weights/best.pt
#
# Steps (same pattern as train_module1.sh):
#   1. Environment check (GPU, venv, raw GSR-2025 data)
#   2. Dry-run the converter to confirm team/role distributions
#   3. Convert GSR-2025 → 6-class YOLO dataset (skips if done)
#   4. Fine-tune yolov8n.pt on the 6-class dataset
#   5. Eval fine-tuned weights
#   6. Generate comparison report
#
# Usage (cloud GPU box, inside tmux):
#
#     cd /root/work/football-log
#     bash scripts/train_module3b.sh
#
# Override defaults:
#
#     IMGSZ=640 BATCH=32 EPOCHS=50 bash scripts/train_module3b.sh
#

set -euo pipefail

# ---- Configuration (override via env) ----------------------------------------

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PYTHON="${VENV_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

if [ -z "${RAW_DIR:-}" ]; then
    if [ -d "${PROJECT_ROOT}/data/gsr2025/extracted" ]; then
        RAW_DIR="${PROJECT_ROOT}/data/gsr2025/extracted"
    else
        RAW_DIR="${PROJECT_ROOT}/data/gsr2025/extracted"
    fi
fi

# Separate output dir — don't clobber the 3-class Module 1 dataset.
YOLO_DIR="${YOLO_DIR:-${PROJECT_ROOT}/data/soccernet_6class}"

TRAIN_MODEL="${TRAIN_MODEL:-yolov8n.pt}"   # smaller base: team classification is simpler
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
DEVICE="${DEVICE:-0}"

RUNS_DIR="${RUNS_DIR:-${PROJECT_ROOT}/runs}"
EXP_NAME="${EXP_NAME:-module3b_v1}"

SPLIT_RATIOS="${SPLIT_RATIOS:-0.7,0.15,0.15}"
SEED="${SEED:-42}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"

# ---- Helpers -----------------------------------------------------------------

log()  { printf "\n\033[1;36m[train_module3b]\033[0m %s\n" "$*"; }
warn() { printf "\n\033[1;33m[train_module3b] WARN:\033[0m %s\n" "$*"; }
die()  { printf "\n\033[1;31m[train_module3b] ERROR:\033[0m %s\n" "$*" >&2; exit 1; }

require_file() { [ -e "$1" ] || die "missing required path: $1"; }

cd "${PROJECT_ROOT}"

# ---- Step 0: environment sanity check ----------------------------------------

log "Step 0: environment check"

[ -x "${VENV_PYTHON}" ] || die "venv python not found at ${VENV_PYTHON}. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"

"${VENV_PYTHON}" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0), '|', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB VRAM')"

if [ ! -d "${RAW_DIR}" ]; then
    die "raw dataset not found: ${RAW_DIR}. Download GSR-2025 first (see scripts/download_gsr2025.sh)"
fi

mkdir -p "${RUNS_DIR}/${EXP_NAME}"

# ---- Step 1: dry-run to confirm team/role distributions ----------------------

DRY_RUN_LOG="${RUNS_DIR}/${EXP_NAME}/converter_dry_run.txt"

log "Step 1: dry-run — verify team/role distributions (→ ${DRY_RUN_LOG})"
"${VENV_PYTHON}" scripts/prepare_yolo_dataset.py \
    --source-dir "${RAW_DIR}" \
    --dry-run 2>&1 | tee "${DRY_RUN_LOG}"

log "Confirm: team_distribution should show both 'left' and 'right' entries."
log "If team_distribution is empty, the data lacks team attributes → 3B cannot run."
if [ "${NONINTERACTIVE:-0}" = "1" ]; then
    log "  NONINTERACTIVE=1 → skipping confirmation"
else
    log "Press ENTER to continue, Ctrl-C to abort."
    read -r _
fi

# ---- Step 2: convert → 6-class YOLO dataset ----------------------------------

if [ -f "${YOLO_DIR}/soccernet.yaml" ]; then
    log "Step 2: SKIPPED — 6-class dataset already exists at ${YOLO_DIR}/soccernet.yaml"
else
    log "Step 2: convert GSR-2025 → 6-class YOLO format"
    "${VENV_PYTHON}" scripts/prepare_yolo_dataset.py \
        --source-dir "${RAW_DIR}" \
        --output-dir "${YOLO_DIR}" \
        --team-classes \
        --split-ratios "${SPLIT_RATIOS}" \
        --seed "${SEED}"

    require_file "${YOLO_DIR}/soccernet.yaml"
    log "Conversion done — manifest: ${YOLO_DIR}/manifest.json"
fi

DATA_YAML="${YOLO_DIR}/soccernet.yaml"

# ---- Step 3: fine-tune -------------------------------------------------------

BEST_WEIGHTS="${RUNS_DIR}/${EXP_NAME}/weights/best.pt"

if [ "${SKIP_TRAIN}" = "1" ]; then
    log "Step 3: SKIPPED — SKIP_TRAIN=1"
elif [ -f "${BEST_WEIGHTS}" ]; then
    log "Step 3: SKIPPED — best.pt already exists at ${BEST_WEIGHTS}"
    log "  Delete it (or set a different EXP_NAME) to retrain."
else
    log "Step 3: fine-tune ${TRAIN_MODEL} on ${DATA_YAML}"
    log "  imgsz=${IMGSZ} batch=${BATCH} epochs=${EPOCHS} patience=${PATIENCE} device=${DEVICE}"

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
  "task": "team-as-class detection (Module 3B)",
  "classes": ["team_a_player","team_b_player","goalkeeper_a","goalkeeper_b","referee","ball"],
  "git_sha": "$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
fi

# ---- Step 4: eval -----------------------------------------------------------

EXP_DET_JSON="${RUNS_DIR}/${EXP_NAME}/detection.json"

if [ ! -f "${BEST_WEIGHTS}" ]; then
    warn "Step 4: SKIPPED — no fine-tuned weights to evaluate"
elif [ -f "${EXP_DET_JSON}" ]; then
    log "Step 4: SKIPPED — eval already at ${EXP_DET_JSON}"
else
    log "Step 4: eval fine-tuned weights on 6-class val split"
    "${VENV_PYTHON}" -m football_log.eval.eval_detection \
        --weights "${BEST_WEIGHTS}" \
        --data "${DATA_YAML}" \
        --imgsz "${IMGSZ}" \
        --device "${DEVICE}" \
        --out "${EXP_DET_JSON}"
fi

# ---- Step 5: report ---------------------------------------------------------

log "Step 5: generate comparison report"
"${VENV_PYTHON}" -m football_log.eval.report || warn "report generator returned non-zero (likely missing inputs)"

# ---- Done -------------------------------------------------------------------

log "All steps done."
echo
echo "What to pull back to your laptop:"
echo "  scp -P 11912 root@js3.blockelite.cn:${BEST_WEIGHTS} ./runs/${EXP_NAME}/weights/"
echo "  scp -P 11912 -r root@js3.blockelite.cn:${RUNS_DIR}/${EXP_NAME} ./runs/"
echo
echo "Then on the laptop, run inference with the new weights:"
echo "  .venv/bin/python run.py --video YOUR_VIDEO.mp4 \\"
echo "      --team-class-model runs/${EXP_NAME}/weights/best.pt"
