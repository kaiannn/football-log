# Setbacks log

A running record of issues encountered during football-log development —
both ones that were resolved and ones parked for later modules. Intended
for the project report's "challenges and lessons learned" section, and as
context for anyone picking up the codebase.

Categories:

1. Codebase audit findings (resolved)
2. Known unresolved issues (parked for upcoming modules)
3. Development environment friction
4. Schema / API quirks worth knowing
5. Open algorithmic questions (sent to advisor)

---

## 1. Codebase audit findings (resolved)

### 1.1 Protocol/implementation contract drift

**What**: `TeamClassifierProto` declared `classify(frame, detection) -> str`, but
the default `TeamClassifier` only implements `instant_label` + `smooth_label`
— and the pipeline calls those, not `classify`. A user implementing the
documented Protocol would silently fail; the default `Detector` would never
call their code.

**Why it's worth recording**: Python `Protocol` is duck-typed at runtime;
nothing forces the protocol declaration to match real call sites. We caught
it by reading both files manually.

**Lesson**: every Protocol needs a `runtime_checkable` test that validates
the default implementation conforms. Locked by
`tests/test_detection_dataclass.py::test_default_runtime_protocol_match_for_team_classifier`.

### 1.2 `datetime.utcnow()` deprecation

**What**: `io/export.py` used `datetime.utcnow().isoformat() + "Z"`. Deprecated
in Python 3.12, scheduled for removal.

**Fix**: switched to `datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")`.
Same output bytes, no deprecation.

### 1.3 Dead code from prior refactors

**What**: `tracker.py::track_frame()`, `runner.py::yolo_tracker` alias, and
`export.py::records_written.setter` had zero callers — leftovers from
abandoned refactors.

**Fix**: deleted. **Lesson**: refactor leftovers accumulate silently; periodic
audits matter even on small codebases.

### 1.4 `requirements.txt` had no version pins

**What**: all five dependencies were unpinned. `ultralytics` ships breaking
changes between minor versions; `numpy 2.x` removed APIs `1.x` had.

**Fix**: added lower bounds with major-version upper bounds
(`ultralytics>=8.1,<9`, `numpy>=1.24,<3`, etc).

### 1.5 LICENSE referenced but missing

**What**: README badge linked to `LICENSE`, file didn't exist.

**Fix**: added MIT LICENSE matching the README badge.

### 1.6 `.DS_Store` tracked despite `.gitignore`

**What**: `.DS_Store` was committed in an early commit, *then* `.gitignore`
was added. `.gitignore` only stops new additions; existing tracked files
keep getting tracked.

**Fix**: `git rm --cached .DS_Store`. **Lesson**: when adding a `.gitignore`
rule for a file already in history, also explicitly untrack it.

### 1.7 README claims didn't match the code

**What**: README said the team classifier used "LAB K-Means"; the code used
HSV. Sample JSONL row in README had a `team` field that doesn't exist in
output. Plugin table referenced an obsolete `classify` method.

**Fix**: rewrote README sections to match actual implementation.

### 1.8 Subprocess injection risk in `experiments/tvcalib_infer.py`

**What**: `f"conda run -n {env_name} ..."` interpolated user input into a
shell string. Low real risk (dev-only tool), but bad pattern.

**Status**: noted as a 🟢-level item in the audit; not yet fixed since the
file is being rewritten anyway during Module 4.

---

## 2. Known unresolved issues (parked)

### 2.1 Skip-frame tracking corrupts ID assignment

**What**: `model.track(persist=True)` with `--detect-every-n > 1` causes
ultralytics' internal tracker to interpret skipped frames as "no detections."
Its Kalman state goes stale, and on the next detected frame the position has
jumped enough that the tracker assigns a new ID. Result: a single player gets
re-counted as multiple track IDs.

**Why parked**: addressed structurally by Module 2 (DeepSORT replaces the
built-in ultralytics tracker entirely, so we control how skipped frames are
handled).

### 2.2 HSV K-Means team classifier collapses on low-saturation kits

**What**: when both teams wear low-saturation kits (black vs white, or two
shades of grey), HSV with only H+S channels can't separate them — saturation
is near zero for both, hue is meaningless. K=2 also assumes only two clusters
exist, but real matches have 5+ (two teams, two goalkeepers, referees).

**Why parked**: Module 3 replaces the entire classifier — keypoint-based color
voting (samples only torso pixels) plus an optional team-as-class detector
trained directly on SoccerNet team labels.

### 2.3 Foot-anchor projection breaks on jumps and occlusions

**What**: world coordinates are computed from the bbox's bottom-center pixel
("foot anchor") under the assumption it lies on the ground plane. False when:
- player jumps (header, save, slide tackle)
- bbox bottom is occluded (advertising boards, other players)
- player is the ball (it flies)

Error spikes to meter-scale at the apex of a jump.

**Why parked**: Module 5 (BEV Kalman filter) handles this — observation
variance scales with bbox confidence and visibility heuristics, and the
filter rejects 3σ-outlier observations entirely.

### 2.4 Gradio default binding is `0.0.0.0`

**What**: `web.py` and `run_web.py` listen on all network interfaces with no
auth. Anyone on the same LAN can hit the UI, upload files, run inference on
the host.

**Why parked**: needs a `--public` flag toggle (default `127.0.0.1`); minor
fix to be done before any public deployment, not while it's a local dev tool.

### 2.5 `print` everywhere instead of `logging`

**What**: pipeline-internal status output uses `print` calls. No way to
control verbosity, no way to redirect to a file, no timestamps.

**Why parked**: pure ergonomics, no functional impact. Will be cleaned up
during Phase 6 polish.

### 2.6 Web UI touches private pipeline methods

**What**: `web.py` calls `pipeline._iter_frames()` and `pipeline._finish()`
— underscore-prefixed methods that signal "internal." CLI uses
`pipeline.run()`. Two paths to the same logic, easy to drift.

**Why parked**: needs an API design pass (promote `iter_frames` / `finish` to
public, or unify both entry points). Phase 6 polish.

---

## 3. Development environment friction

### 3.1 `python` not in PATH on macOS Homebrew

**What**: macOS only ships `python3`. Homebrew installs to
`/opt/homebrew/bin/python3`. Tutorials assuming `python` works will fail.

**Resolution**: always use `.venv/bin/python` for project work. Project README
should make this explicit.

### 3.2 `pytest` not in `requirements.txt`

**What**: a fresh checkout couldn't run tests until `pytest` was installed
manually.

**Resolution**: created `requirements-dev.txt` (pytest, motmetrics) so test
dependencies are documented separately from runtime deps. **Lesson**: split
runtime vs dev requirements from day one.

### 3.3 `pandoc` not available when generating Chinese `.docx` for advisor

**What**: needed to convert a markdown CV-questions doc to `.docx` for an
advisor meeting. Pandoc wasn't installed; installing it would have pulled
in a sizeable LaTeX toolchain.

**Resolution**: wrote a small custom script (`/tmp/build_docx.py`) using
`python-docx`, with explicit Chinese font fallback (PingFang SC for body,
Menlo for code). Far lighter than pandoc + xelatex for a one-off academic
deliverable.

### 3.4 `motmetrics` is heavy and rarely needed

**What**: tracking eval needs `motmetrics` (depends on pandas, scipy, etc.).
Most uses of football-log don't need eval at all.

**Resolution**: `eval_tracking.py` lazy-imports it, and it lives in
`requirements-dev.txt` (not the main runtime deps). The eval orchestrator
skips tracking metrics gracefully if the import fails.

### 3.5 SoccerNet Tracking is ~100 GB — too big for most laptops

**What**: the obvious workflow ("download SoccerNet to my Mac, fine-tune
locally") fails for the simple reason that 100 GB doesn't fit on most
laptops, and the laptop probably has no GPU anyway.

**Resolution**: cloud-first workflow — see `docs/cloud-workflow.md`. The
dataset, training run, and intermediate weights all live on a rented GPU
machine. Only the final `best.pt` (~20 MB) and the eval JSONs come back
to the laptop. Total local footprint stays under ~1 GB.

For tight cloud-disk rentals (~50 GB), the converter ships three subsample
knobs that compose: `--frame-stride`, `--max-sequences`,
`--max-frames-per-sequence`. All three are recorded in the dataset's
`manifest.json` for provenance.

**Lesson**: "build it on your laptop first" doesn't scale to research-grade
CV datasets. Plan the data lifecycle (cloud-only, cloud-then-local, or
hybrid) before writing the training script.

### 3.6 SoccerNet Tracking host (KAUST) unreachable from China

**What**: the SoccerNet `pip install SoccerNet` downloader hard-codes the
KAUST Nextcloud (`exrcsdrive.kaust.edu.sa`) as the file source. From a
mainland-China cloud GPU rental, that endpoint stalls at ~190 B/s — at
that rate the 100 GB tracking split would take 8+ days. We probed the
package source to confirm only **2025 tasks** (`mvfouls-2025`,
`gamestate-2025`, `depth-2025`, `spotting-ball-2025`) have a HuggingFace
fallback path; the older `tracking` task is locked to KAUST.

**Resolution**: switched the entire Module 1 dataset to **SoccerNet
Game-State Reconstruction 2025 (SN-GSR-2025)** —
[https://hf-mirror.com/datasets/SoccerNet/SN-GSR-2025](https://hf-mirror.com/datasets/SoccerNet/SN-GSR-2025).
17 GB instead of 100 GB, and reachable from China via `hf-mirror.com`.
Even there, `huggingface_hub.snapshot_download` was hitting recurring
read-timeouts because the LFS files redirect to AWS-east via Xet. Fixed
by replacing the downloader with `aria2c -x 16 -s 16 --max-tries=0
--retry-wait=10 --timeout=120`, which holds 16 parallel HTTPS connections
to the same S3 URL and resumes through transient resets. Steady **3.5 MB/s**
on this hardware — full train+test in ~80 minutes.

**Side effect (positive)**: GSR-2025 ships *richer* annotations than
SoccerNet Tracking (per-bbox `team` / `role` / `jersey`, plus pitch-line
and camera-pose ground truth in the same JSON). Module 3B's "team-as-class
detector" plan now has direct labels without scraping a separate file,
and Module 4 has true pitch-line GT to validate auto-calibration against.

**Implementation**: `scripts/download_gsr2025.sh` plus a new
`football_log/data/gsr_convert.py` (15 tests) that handles the COCO-style
`Labels-GameState.json` format. `scripts/prepare_yolo_dataset.py` now
auto-detects the source format (`--format auto|mot|gsr`); the legacy MOT
converter is kept for if/when SoccerNet Tracking becomes reachable again.

**Lesson**: (1) when a dataset hosts files in a single foreign region,
verify your training network reaches it before the rental clock starts;
(2) for HuggingFace LFS via mainland-China mirrors, prefer aria2c over
`hf_hub_download` — the multi-connection model is much more tolerant of
the long-haul resets you can't escape.

---

## 4. Schema / API quirks worth knowing

### 4.1 JSONL vs CSV null asymmetry

**What**: when `world_x_m` / `world_y_m` are missing, JSONL writes `null`
but CSV writes `""`. Both are correct for their formats, but downstream
consumers parsing both must handle both encodings.

**Locked by**: `tests/test_export_schema.py::test_jsonl_world_coords_null_when_missing`
and `test_csv_world_coords_blank_when_missing`.

### 4.2 `Detection.from_dict` doesn't restore `extra`

**What**: `Detection.to_dict()` spreads the `extra` dict at the top level
of the output dict, but `Detection.from_dict()` reads only the explicit
fields — anything in `extra` is lost on a round trip.

**Why parked**: no detector populates `extra` yet, so the asymmetry isn't
exercised. If/when a detector starts using `extra`, this needs fixing.
Documented in test comments.

### 4.3 `TrackingDataWriter.write_frame` accepts both `Detection` and `dict`

**What**: dual interface for backward compatibility — legacy callers pass
plain dicts, new code passes `Detection` instances.

**Risk**: dict callers can omit fields the dataclass constructor would have
caught. Errors surface only at write time, not at construction time.

### 4.4 Bbox-foot is mid-bottom for players, mid-center for ball

**What**: `project_foot_to_world` uses `(x + w/2, y + h)` for player labels
and `(x + w/2, y + h/2)` for the ball. Two different anchor conventions in
one function. Makes sense (ball isn't on the ground), but easy to overlook.

**Locked by**: `tests/test_world_homography.py::test_project_foot_to_world_player_uses_foot_anchor`
and `test_project_foot_to_world_ball_uses_bbox_center`.

### 4.5 SoccerNet class IDs are not stable across releases

**What**: SoccerNet Tracking annotation `cls` column has changed numbering
between releases (e.g. ball was `5` in one revision, `1` in another; team-
specific IDs vary). Hard-coding a class map in the converter would silently
mislabel data on a different release.

**Mitigation**: `scripts/prepare_yolo_dataset.py --dry-run` reports the
actual class IDs and counts in the source data before any writes. The
default class map (in `football_log.data.yolo_convert.default_class_map`)
emits a warning when used; the recommended path is a per-dataset
`class_map.yaml`. The converter manifest records which mapping was applied,
so the dataset's provenance is auditable.

**Locked by**: `tests/test_yolo_convert.py::test_dry_run_reports_class_counts`
and `test_custom_class_map_with_only_ball`.

### 4.6 Auto-calibration line classification is unsolved without real video

**What**: Module 4's "real" path is per-frame line detection + classification
(which Hough line is the sideline vs. the goal line vs. the center line).
This is genuinely hard CV that the published baselines (TVCalib, PnLCalib,
SoccerNet calibration challenge) attack with neural-net-based keypoint
detectors trained on labeled pitch lines. Implementing this offline without
a held-out video to validate against would just produce untested code.

**Resolution**: shipped the *starter* approach the plan called for —
hand-labeled keyframes plus Lucas–Kanade optical-flow propagation between
them. The user labels 4–8 points at sparse keyframes (every 5–15 s), and the
runtime tracks those points across intermediate frames. The architecture is
also wired to accept a pre-computed `(N, 3, 3)` homography sequence
(`--homography-sequence`), so when TVCalib output is available it can drop
straight in without code changes.

**Lesson**: when a piece of CV needs validation against real data to be
useful, build the architecture and the validation harness first; defer the
final algorithm choice until you can measure it.

**Locked by**: `tests/test_auto_calibration.py` (23 tests covering homography
fit, EMA smoothing, keyframe loader, projector adapter).

---

## 5. Open algorithmic questions (sent to advisor 2026-05-14)

These are the unresolved CV decisions the project is pending advisor input on:

1. **Single-camera depth recovery** — bbox foot-anchor + ground-plane
   assumption breaks on jumps/occlusion. Is monocular depth estimation
   + pitch prior worth pursuing, or commit to multi-frame triangulation?

2. **Online camera calibration** — TVCalib / PnLCalib / SoccerNet
   Camera Calibration Challenge: which is most realistic to integrate
   given the project's CPU-friendly target?

3. **Team classification dead-ends** — color-space approaches all hit the
   same wall (low-saturation kits, lighting drift). Is moving directly to
   Re-ID embeddings the right next step, or does keypoint-based color
   voting buy enough?

4. **Ball-specific tracking** — is sharing one YOLO between players and
   ball fundamentally wrong? Should there be a dedicated TrackNet-style
   small-target detector?

Full text in `~/Documents/football-log-cv-questions.md`.

---

## How to add to this log

When you hit a new setback, add it to the appropriate section above. Each
entry should answer:

- **What**: one paragraph describing the problem
- **Fix** or **Why parked**: how it was resolved, or why it's deferred
- **Lesson** (optional): the takeaway worth carrying forward

Keep it short. The log is useful only if it stays scannable.
