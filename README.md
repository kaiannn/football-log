# football-log

把足球比赛视频转成结构化轨迹数据：视频 → YOLO 跟踪（ByteTrack/BoT-SORT）→ 可选球衣分队与世界坐标 → JSONL/CSV 导出。

## 项目结构（主包）

```
football-log/
├── football_log/
│   ├── app/            # CLI、Web UI、视频流水线 VideoTrackerPipeline
│   ├── vision/         # YOLO 跟踪、LAB K-Means 自动分队与时序平滑
│   ├── pitch/          # 场地：草地/场线/四边形（OpenCV，可选）
│   ├── world/          # PitchSpec、单应 Homography、针孔+地面 PinholeGroundProjector
│   ├── io/             # JSONL/CSV、meta
│   └── ui/             # OpenCV 叠加绘制
├── run.py                # CLI 入口
├── run_web.py            # Web UI 入口（Gradio）
├── requirements.txt
└── README.md
```

根目录下 `mark_pitch_quadrilateral.py`、`detect_pitch_grass_contour.py`、`video_*.py`、`opencv_smoke_test.py` 等为**独立实验脚本**，不参与主包 import。

**推荐入口：**

- **Web UI（推荐）：** `python3 run_web.py`，浏览器打开 `http://localhost:7860`
- **CLI：** `python3 run.py` 或 `python3 -m football_log.app.cli`

---

## 跑通「整条流水线」：需要提供什么？

### 必须（缺一不可）

| 输入 | 形式 | 说明 |
|------|------|------|
| **比赛视频** | 本地文件路径 | 传给 `--video`，如 `match.mp4`（常见编码 H.264 即可） |

仅此即可跑通：**检测 + 跟踪 + 分队 + 写出轨迹**；`world_x_m` / `world_y_m` 为 `null`（除非下面任选标定）。

### 可选（按需叠加）

| 输入 | 形式 | 作用 |
|------|------|------|
| **YOLO 权重** | `.pt` 文件或 Ultralytics 内置名 | `--model`，默认 `yolov8n.pt`，首次运行会自动下载 |
| **跟踪器配置** | `bytetrack.yaml` / `botsort.yaml` 等 | `--tracker`，与 Ultralytics 一致 |
| **单应矩阵（像素→米）** | `3×3` 的 `.npy`，`float64` | `--homography`，轨迹中写入 `world_x_m` / `world_y_m` |
| **针孔相机 + 地面平面** | `.json` 或 `.yaml` | `--camera-calib`，与单应二选一；若同时提供，**优先针孔** |
| **球场尺寸（元数据）** | 两个数字（米） | `--pitch-length-m`、`--pitch-width-m`，写入 `*.meta.json`，默认 105×68 |
| **场地估计（草皮+线+四边形）** | 无文件，仅开关 | `--pitch-field-detect`，可选 `--pitch-field-every-n`、`--pitch-field-filter-tracks` |

### 一键示例

**最小（只要视频，无界面批处理）：**

```bash
python3 -m football_log.app.cli --video "/path/to/match.mp4" --no-ui --output-dir outputs --output-format both
```

**带世界坐标（二选一）：**

```bash
# 单应：事先用标定得到 H，保存为 3×3 的 homography.npy
python3 -m football_log.app.cli --video "/path/to/match.mp4" --no-ui --homography /path/to/homography.npy

# 或：针孔标定文件（K、dist、R|t 或 R|C，见 football_log/world/pinhole_ground.py）
python3 -m football_log.app.cli --video "/path/to/match.mp4" --no-ui --camera-calib /path/to/camera_calib.json
```

**叠加场地估计与草皮过滤（仍只需视频 + 开关）：**

```bash
python3 -m football_log.app.cli --video "/path/to/match.mp4" --no-ui \
  --pitch-field-detect --pitch-field-every-n 15 --pitch-field-filter-tracks
```

**有界面预览（按 `q` 退出）：**

```bash
python3 -m football_log.app.cli --video "/path/to/match.mp4"
```

### 启发式标定物拟合（多帧/多参考 + 鲁棒）

当已有的 `homography.npy` 存在尺度偏差时，可用已知尺寸矩形做联合拟合。当前支持：

- 交互式：在**多帧**逐帧点击 4 点；
- 文件式：从 `refs-json` 读入多个参考；
- 鲁棒损失：`none` / `huber` / `cauchy`，降低坏点影响。

示例（多帧交互）：

```bash
python3 -m football_log.app.calibrate_reference \
  --video "/path/to/match.mp4" \
  --homography "/path/to/homography.npy" \
  --frame-idxs 100,220,360 \
  --ref-width-m 2.00 \
  --ref-length-m 1.00 \
  --robust-loss huber \
  --robust-delta-m 0.2 \
  --out-homography "/path/to/homography_refit.npy"
```

交互方式：左键点 4 个角（大致 tl,tr,br,bl），`Enter` 确认，`r` 重选，`q` 退出。

`refs-json` 示例：

```json
[
  {
    "image_points_xy": [[100, 200], [220, 205], [225, 260], [95, 255]],
    "width_m": 2.0,
    "length_m": 1.0
  },
  {
    "image_points_xy": [[130, 210], [252, 214], [256, 270], [126, 266]],
    "width_m": 2.0,
    "length_m": 1.0
  }
]
```

```bash
python3 -m football_log.app.calibrate_reference \
  --video "/path/to/match.mp4" \
  --homography "/path/to/homography.npy" \
  --refs-json "/path/to/refs.json" \
  --robust-loss cauchy \
  --out-homography "/path/to/homography_refit.npy"
```

> 注：当前为工程化启发式（坐标下降 + 衰减步长）用于快速修正尺度，后续会替换为更高精度方法。

### 环境与依赖

- Python 3.9+ 建议；需安装 `requirements.txt`（`opencv-python`、`numpy`、`ultralytics`、`pyyaml`、`gradio`）。
- 首次使用 YOLO 需联网下载权重（或使用已下载的 `.pt` 路径）。

---

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## CLI 常用参数（摘要）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--video` | （必填） | 输入视频路径 |
| `--output-dir` | `outputs` | 输出目录 |
| `--output-format` | `both` | `jsonl` / `csv` / `both` |
| `--model` | `yolov8n.pt` | YOLO 权重 |
| `--tracker` | `bytetrack.yaml` | 跟踪配置 |
| `--conf` | `0.3` | 检测置信度 |
| `--imgsz` | `640` | 推理边长 |
| `--detect-every-n` | `1` | 每 N 帧做一次检测（可提速） |
| `--no-ui` | 关 | 无窗口批处理 |
| `--homography` | 无 | `3×3` `.npy`，像素→世界米 |
| `--camera-calib` | 无 | 针孔+地面标定 JSON/YAML |
| `--pitch-length-m` / `--pitch-width-m` | 105 / 68 | 写入 meta |
| `--pitch-field-detect` | 关 | 启用场地估计 |
| `--pitch-field-every-n` | 15 | 场地估计更新间隔 |
| `--pitch-field-filter-tracks` | 关 | 用草皮掩膜过滤检测 |

---

## 输出文件

默认 `outputs/<视频名>_tracks.jsonl|csv` 与 `<视频名>_tracks.meta.json`。

记录字段包括：`frame_idx`、`timestamp_sec`、`track_id`、`label`、`x,y,w,h`、`conf`；若启用标定则还有 `world_x_m`、`world_y_m`。

---

## 说明与调参提示

- 跟踪为 YOLO `track` + ByteTrack（默认）；性能不足时可试 `--detect-every-n 2` 或 `3`。
- 世界坐标依赖标定质量；单应与 `PitchSpec` 的世界轴约定需与标定时选点一致。
- 球衣分队为 LAB 色度 K-Means 自动聚类 + 时序平滑，也可通过 `--team-colors` 手动指定两队 BGR 颜色。

---

## 后续方向（简要）

数据质量与可评估指标、标定时序平滑、事件语义层、工程化与模型升级等，见历史讨论与 issue；优先保证评估与世界坐标约定一致后再扩展功能。
