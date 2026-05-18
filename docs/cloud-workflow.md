# Cloud-first workflow & low-memory recipes

This project is designed to be run with **data and training on a rented GPU
machine**, while only inference + reporting happens on your laptop. This
document captures the recipe and the knobs available when memory is tight.

## Why cloud-first

| Resource | Local Mac | Cloud GPU box |
|---|---|---|
| SoccerNet Tracking dataset (~100 GB) | ❌ keep off | ✅ lives here |
| Fine-tune training | ❌ no GPU | ✅ runs here |
| `best.pt` weights file (6–50 MB) | ✅ pulled back here | produced here |
| Inference on user videos | ✅ runs here | optional |
| Eval reports for write-up | ✅ generated here | — |

Total Mac footprint stays under ~1 GB (code + weights + a few test videos).

## Step-by-step recipe

### 1. On the cloud GPU machine

```bash
# Clone, install
git clone git@github.com:kaiannn/football-log.git
cd football-log
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install SoccerNet

# Download SoccerNet Tracking
python -c "
from SoccerNet.Downloader import SoccerNetDownloader as D
d = D(LocalDirectory='data/soccernet/raw')
d.password = 'YOUR_SOCCERNET_PASSWORD'
d.downloadDataTask(task='tracking', split=['train', 'test'])
"

# Verify class IDs in this release
python scripts/prepare_yolo_dataset.py --source-dir data/soccernet/raw --dry-run

# Convert (use subsample knobs if disk is tight — see "low-memory recipes" below)
python scripts/prepare_yolo_dataset.py \
    --source-dir data/soccernet/raw \
    --output-dir data/soccernet \
    --class-map football_log/data/soccernet_classes.example.yaml

# Lock in baseline numbers
cp football_log/eval/config.example.yaml eval_config.yaml
# edit: detection.data → data/soccernet/soccernet.yaml
python -m football_log.eval --config eval_config.yaml --exp-name baseline

# Fine-tune (see "GPU-size recipes" below for batch / imgsz)
yolo detect train data=data/soccernet/soccernet.yaml model=yolov8s.pt \
    epochs=50 imgsz=1280 batch=8 patience=10 \
    project=runs name=module1_v1

# Re-evaluate against the new weights
# (edit eval_config.yaml: detection.weights → runs/module1_v1/weights/best.pt)
python -m football_log.eval --config eval_config.yaml --exp-name module1_v1
```

### 2. Pull weights + reports back to Mac

```bash
# From your Mac, into the local football-log clone:
mkdir -p runs/module1_v1/weights
scp <cloud-host>:~/football-log/runs/module1_v1/weights/best.pt \
    runs/module1_v1/weights/

# Optionally pull the eval JSONs too
scp -r <cloud-host>:~/football-log/runs/module1_v1 runs/

# Regenerate the comparison report locally
python -m football_log.eval.report
```

### 3. Use the fine-tuned weights for local inference

```bash
python run.py --video match.mp4 \
    --model runs/module1_v1/weights/best.pt \
    --player-class-id 0 --ball-class-id 1 --referee-class-id 2
```

---

## Low-memory disk recipes

The converter has three subsample knobs that compose. Pick the recipe that
matches your free disk on the cloud box.

| Free disk | Recipe | Approx. dataset size | Recommended for |
|---|---|---|---|
| ≥ 100 GB | full convert, no subsampling | 30–60 GB YOLO output | thesis-quality numbers |
| 50–100 GB | `--frame-stride 2` | 15–30 GB | balanced |
| 20–50 GB | `--frame-stride 5` | 6–12 GB | tight rentals |
| 10–20 GB | `--frame-stride 5 --max-sequences 50` | 2–5 GB | smoke-test runs |
| < 10 GB | `--frame-stride 10 --max-sequences 20 --max-frames-per-sequence 100` | < 1 GB | "does it train at all" |

Subsampling notes:
- `--frame-stride 5` keeps every 5th frame in each sequence. For a 90-min
  match at 25 fps that's 27,000 → 5,400 frames per sequence — still plenty
  for fine-tuning.
- `--max-sequences 50` truncates after the first 50 sequences.
  Combined with stride, this gives you full-resolution images on a small
  subset of matches — better for generalization than seeing every match
  with heavy temporal subsampling.
- `--max-frames-per-sequence` caps after stride. Useful for "give me
  exactly 100 training frames per match."
- All three settings are recorded in `manifest.json` so the dataset's
  provenance is auditable in the report.

The converter symlinks frames by default. If your training environment
will be moved to a different filesystem (e.g. tarballed for upload), pass
`--copy-images` so the images are real files instead of symlinks.

---

## GPU VRAM recipes

If your cloud GPU is small, training will OOM unless you tune `imgsz` and
`batch`. Use the table below as a starting point; bump down if you OOM.

| GPU | VRAM | Model | imgsz | batch | epochs | Wall time (50 ep, 100k images) |
|---|---|---|---|---|---|---|
| RTX 3060 / T4 | 12–16 GB | yolov8n | 640 | 16 | 50 | ~2–3 h |
| RTX 4090 / A10 | 24 GB | yolov8s | 1280 | 8 | 50 | ~3–5 h |
| A100 40 GB | 40 GB | yolov8m | 1280 | 32 | 50 | ~2 h |
| A100 80 GB / H100 | 80 GB | yolov8l | 1536 | 32 | 50 | ~2–3 h |

Rules of thumb:
- **For Module 1, prefer `imgsz=1280` over `imgsz=640`.** Soccer's small
  targets (the ball + far-side players) lose recall at 640. If VRAM is
  tight, drop `batch` first — keep `imgsz` high.
- **Use `cache=ram`** in `yolo detect train` if you have free system RAM —
  it avoids re-reading frames from disk every epoch (~2x speedup).
  Skip this on tight-RAM machines.
- **Use `device=0`** to be explicit about which GPU. Multi-GPU
  (`device=0,1`) helps only if model is the bottleneck, not data loading.

---

## Resume after crash

If conversion or training crashes mid-way:

| What crashed | What to do |
|---|---|
| `prepare_yolo_dataset.py` | Re-run with the same args. The script overwrites; the `--seed` flag keeps splits stable. |
| `yolo train` | Add `resume=True` to the same `yolo detect train` command. ultralytics resumes from `last.pt`. |
| Eval (`football_log.eval`) | Re-run; the orchestrator overwrites the `runs/<exp>/` directory. |

The converter is intentionally idempotent: same inputs + same `--seed` →
same output, regardless of how many times you run it.

---

## What stays on your Mac

The repo plus inference videos. After Module 1 is done:

```
~/football-log/
├── football_log/                  # source code (small)
├── runs/
│   └── module1_v1/
│       ├── weights/best.pt        # ~20 MB, this is the only training artifact
│       ├── meta.json
│       ├── detection.json         # mAP numbers from cloud eval
│       └── tracking.json
├── outputs/
│   └── match_tracks.jsonl         # output of running pipeline on YOUR videos
└── test_videos/                   # your inference inputs
    └── match.mp4
```

No SoccerNet, no `.pt` checkpoints other than `best.pt`, no training image
caches. **Total local footprint: a few hundred MB.**
