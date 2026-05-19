<div align="center">

# ⚽ football-log

**视频进，数据出 —— 把足球比赛视频变成可分析的结构化轨迹**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-1.0.0-2EA44F?style=flat-square)](https://github.com/kai/football-log)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-111F68?style=flat-square&logo=yolo&logoColor=white)](https://ultralytics.com)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)

给一段足球比赛视频 → 输出带球员/球标签、track ID、时间戳的 JSONL/CSV 轨迹文件<br>
直接用 Pandas、DuckDB 或任何 BI 工具继续分析

[🚀 快速开始](#-快速开始) · [📖 文档](#-使用方法) · [🔬 研究模块](#-研究模块-v10) · [🔌 插件](#-插件系统) · [🧪 实验分支](#-实验分支) · [📚 相关研究](#-相关研究与开源参考)

</div>

---

## ✨ 核心特性

| | 特性 | 说明 |
|---|---|---|
| 🎬 | **一个视频就能跑** | 不需要标定文件、不需要额外数据集，给视频就出数据 |
| 📊 | **输出格式稳定** | `frame_idx` `track_id` `label` `x,y,w,h` `conf` — 字段固定，升级不突变 |
| 🔍 | **过程透明** | 你始终清楚当前做了检测、跟踪、分队、标定中的哪些步骤 |
| 📐 | **质量分层** | 无标定 → 基础轨迹；有标定 → 世界坐标，两种都有价值 |
| 🖥️ | **双入口** | CLI 批处理 + Gradio Web UI，一行命令启动 |
| 🧩 | **可插拔增强** | TVCalib、Roboflow Sports 等研究成果作为实验层，不影响主线 |
| 🔌 | **插件系统** | 5 个 Protocol 接口，自定义检测器/分队/标定/导出即插即用 |

---

## 🏗️ 技术架构

```
                         ┌──────────────────────────────────────────────────┐
  🧪 实验层              │  TVCalib · Roboflow Sports · TrackLab            │
  experiments/           │  AuxFlow · H-RANSAC · SuperGlue · LK 光流       │
                         │  独立依赖，不影响主线安装                          │
                         ├──────────────────────────────────────────────────┤
  🔬 研究模块（可选）     │  Module 1: YOLOv8s 专项权重 + 裁判类             │
  runs/*/weights/        │  Module 2: DeepSORT + Re-ID 嵌入                │
                         │  Module 3A: 关键点色票投票分队                   │
                         │  Module 3B: 6-class 球队即检测类                 │
                         │  Module 4: 逐帧自动标定 → BEV 坐标              │
                         │  Module 5: BEV Kalman 平滑                      │
                         ├──────────────────────────────────────────────────┤
  🔧 增强层（可选开关）   │  球衣分队 · 世界坐标映射 · 场地估计              │
                         │  Pitch overlay · 启发式标定物拟合                 │
                         ├──────────────────────────────────────────────────┤
  🏠 基础层（默认）       │  YOLO 检测 → ByteTrack/BoT-SORT 跟踪            │
                         │  → JSONL/CSV 导出                                │
                         │  仅需 Python + requirements.txt                  │
                         ├──────────────────────────────────────────────────┤
  🔌 接口层              │  5 个 Protocol：Detector · TeamClassifier         │
  protocols.py           │  PitchEstimator · WorldProjector · Exporter      │
                         │  自定义组件注入 Pipeline，无需改主线代码           │
                         └──────────────────────────────────────────────────┘
```

> **设计原则**：效果提升 10% 但安装复杂度翻 3 倍 → 对主目标未必是好事。<br>
> 判断标准：上手难度 ↓ · 分析适配度 ↑ · 效果稳定性 ↑ · 依赖增量 ↓ · 维护成本 ↓

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/kai/football-log.git
cd football-log
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 30 秒跑通

```bash
# Web UI（推荐，浏览器打开 http://localhost:7860）
python3 run_web.py

# 或 CLI 批处理
python3 run.py --video match.mp4 --no-ui --output-format both
```

只需要一个视频文件，就能输出 **检测 + 跟踪 + 分队 + 轨迹导出**。

---

## 📖 使用方法

### 输入需求

| 类型 | 输入 | 参数 | 说明 |
|---|---|---|---|
| ✅ 必须 | 比赛视频 | `--video` | H.264 编码即可，如 `match.mp4` |
| 🔧 可选 | YOLO 权重 | `--model` | 默认 `yolov8n.pt`，首次自动下载 |
| 🔧 可选 | 跟踪器 | `--tracker` | `bytetrack` / `botsort` / `botsort+reid` / `deepsort` |
| 🔧 可选 | 单应矩阵 | `--homography` | `3×3` `.npy`，像素→世界米 |
| 🔧 可选 | 针孔标定 | `--camera-calib` | JSON/YAML，与单应二选一 |
| 🔧 可选 | 关键帧标定 | `--auto-calibration-keyframes` | JSON，逐帧自动 Homography |
| 🔧 可选 | 团队分类器 | `--team-classifier` | `hsv`（默认）或 `keypoint` |
| 🔧 可选 | Module 3B 权重 | `--team-class-model` | 6-class YOLO，直接从检测结果读取队伍 |

### 常用命令

<details>
<summary>🎬 最小化运行（只要视频）</summary>

```bash
python3 run.py --video match.mp4 --no-ui --output-format both
```

</details>

<details>
<summary>📐 带世界坐标</summary>

```bash
# 方式一：单应矩阵
python3 run.py --video match.mp4 --no-ui --homography homography.npy

# 方式二：针孔标定文件
python3 run.py --video match.mp4 --no-ui --camera-calib camera_calib.json

# 方式三：关键帧自动标定（Module 4）
python3 run.py --video match.mp4 --no-ui \
    --auto-calibration-keyframes keyframes.json \
    --homography-smoothing-alpha 0.3
```

</details>

<details>
<summary>🏃 DeepSORT 跟踪器 + Re-ID（Module 2）</summary>

```bash
# 需要安装 deep-sort-realtime：pip install deep-sort-realtime
python3 run.py --video match.mp4 --no-ui --tracker deepsort
```

遮挡严重或球员挤堆时比 ByteTrack 少切换 ID。

</details>

<details>
<summary>👕 Module 3B：6-class 球队即检测类</summary>

```bash
# 用专项训练的 6-class 权重，队伍标签直接来自检测器，不需要颜色分类器
python3 run.py --video match.mp4 --no-ui \
    --team-class-model runs/module3b_v1/weights/best.pt
```

6 个类别：`team_a_player` / `team_b_player` / `goalkeeper_a` / `goalkeeper_b` / `referee` / `ball`

</details>

<details>
<summary>🗺️ BEV Kalman 平滑（Module 5）</summary>

```bash
python3 run.py --video match.mp4 --no-ui \
    --homography homography.npy \
    --bev-smoothing
```

球员跳起、遮挡或出画面时，世界坐标不再出现米级跳变。

</details>

<details>
<summary>🌿 场地估计 + 草皮过滤</summary>

```bash
# --pitch-field-detect 和 --pitch-field-filter-tracks 默认已开启
# 关闭示例（用于无草皮场景或提速）：
python3 run.py --video match.mp4 --no-ui \
    --no-pitch-field-detect \
    --no-pitch-field-filter-tracks
```

</details>

<details>
<summary>🖥️ 有界面预览（按 q 退出）</summary>

```bash
python3 run.py --video match.mp4
```

</details>

<details>
<summary>📏 启发式标定物拟合（修正尺度偏差）</summary>

当已有的 `homography.npy` 存在尺度偏差时，可用已知尺寸矩形做联合拟合：

```bash
python3 -m football_log.app.calibrate_reference \
  --video match.mp4 \
  --homography homography.npy \
  --frame-idxs 100,220,360 \
  --ref-width-m 2.00 \
  --ref-length-m 1.00 \
  --robust-loss huber \
  --out-homography homography_refit.npy
```

交互方式：左键点 4 个角（tl,tr,br,bl），`Enter` 确认，`r` 重选，`q` 退出。

</details>

### CLI 参数速查

**基础 I/O**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--video` | 必填 | 输入视频路径 |
| `--output-dir` | `outputs` | 输出目录 |
| `--output-format` | `both` | `jsonl` / `csv` / `both` |
| `--no-ui` | — | 无窗口批处理模式 |

**检测与跟踪**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | `yolov8n.pt` | YOLO 权重路径（或 `yolov8s.pt` 等） |
| `--tracker` | `bytetrack` | `bytetrack` / `botsort` / `botsort+reid` / `deepsort` / 自定义 `.yaml` |
| `--conf` | `0.3` | 检测置信度阈值 |
| `--imgsz` | `640` | 推理边长（像素） |
| `--detect-every-n` | `1` | 每 N 帧检测一次（可提速） |

**类别 ID 映射**（使用自定义 YOLO 权重时）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--player-class-id` | `0` | 自定义权重中球员的类别 ID |
| `--ball-class-id` | `32` | 自定义权重中球的类别 ID |
| `--referee-class-id` | 无 | 自定义权重中裁判的类别 ID |

**球队分类**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--team-classifier` | `hsv` | `hsv`（K-Means 颜色）/ `keypoint`（关键点色票投票） |
| `--team-colors` | 无 | 手动指定两队颜色，格式 `B,G,R:B,G,R` |
| `--team-class-model` | 无 | Module 3B：6-class YOLO 权重路径，队伍直接由检测器输出 |

**世界坐标标定**（优先级从高到低）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--auto-calibration-keyframes` | 无 | JSON 关键帧文件，逐帧自动估计 Homography（Module 4） |
| `--homography-sequence` | 无 | `.npy`，每帧一个 3×3 矩阵的数组 |
| `--homography` | 无 | 单个 3×3 `.npy`，全程固定 |
| `--camera-calib` | 无 | 针孔+地面标定 JSON/YAML |
| `--homography-smoothing-alpha` | `0.3` | 逐帧 Homography 的 EMA 平滑系数 |

**场地检测**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--pitch-field-detect` | 开 | 启用草皮/线段检测（`--no-pitch-field-detect` 关闭） |
| `--pitch-field-every-n` | `15` | 场地估计更新间隔（帧） |
| `--pitch-field-filter-tracks` | 开 | 用草皮掩膜过滤场外检测（`--no-pitch-field-filter-tracks` 关闭） |
| `--pitch-length-m` / `--pitch-width-m` | `105` / `68` | 标准场地尺寸，写入 meta |

**BEV 平滑**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--bev-smoothing` | 关 | 启用逐轨迹 BEV Kalman 滤波（Module 5，需先有世界坐标） |

---

## 📤 输出格式

默认输出到 `outputs/<视频名>_tracks.jsonl|csv` 与 `<视频名>_tracks.meta.json`。

```jsonl
{"frame_idx": 120, "timestamp_sec": 4.0, "track_id": 3, "label": "Team A", "x": 342, "y": 188, "w": 48, "h": 112, "conf": 0.87, "world_x_m": 23.4, "world_y_m": -11.2}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `frame_idx` | int | 帧序号 |
| `timestamp_sec` | float | 时间戳（秒） |
| `track_id` | int | 跟踪 ID |
| `label` | str | `Team A` / `Team B` / `Ball` / `Referee` / `Player` |
| `x, y, w, h` | int | 边界框（像素） |
| `conf` | float | 检测置信度 |
| `world_x_m`, `world_y_m` | float? | 世界坐标（米），需标定 |

---

## 🔬 研究模块 (v1.0)

以下 5 个模块在基础层之上叠加，每个都有独立的训练脚本和评估入口。

### Module 1 — 专项检测权重

在 SoccerNet GSR-2025 数据集上微调 YOLOv8s，新增裁判类：

| 类别 | mAP@50 |
|---|---|
| player | 0.978 |
| ball | 0.867 |
| referee | 0.993 |

```bash
# 使用 Module 1 权重
python3 run.py --video match.mp4 \
    --model runs/module1_v1/weights/best.pt \
    --player-class-id 0 --ball-class-id 1 --referee-class-id 2
```

训练脚本：`bash scripts/train_module1.sh`

### Module 2 — DeepSORT + Re-ID

独立 DeepSORT 实现（`deep-sort-realtime`），用外观嵌入辅助重识别，修复 ultralytics 内置跟踪器的跳帧 Kalman 问题：

```bash
pip install deep-sort-realtime
python3 run.py --video match.mp4 --tracker deepsort
```

### Module 3A — 关键点色票投票

用 pose 关键点提取躯干区域像素投票，比纯 HSV 全框聚类更抗干扰：

```bash
python3 run.py --video match.mp4 --team-classifier keypoint
```

### Module 3B — 6-class 球队即检测类

在 SoccerNet GSR-2025 上训练 6-class YOLOv8n，队伍标签直接来自检测器：

| 类别 | mAP@50 |
|---|---|
| team_a_player | 0.764 |
| team_b_player | 0.766 |
| goalkeeper_a | 0.784 |
| goalkeeper_b | 0.844 |
| referee | 0.946 |
| ball | 0.277 |

```bash
python3 run.py --video match.mp4 \
    --team-class-model runs/module3b_v1/weights/best.pt
```

训练脚本：`bash scripts/train_module3b.sh`

### Module 4 — 逐帧自动标定

从视频帧自动估计 Homography，无需手动选点：

```bash
python3 run.py --video match.mp4 \
    --auto-calibration-keyframes keyframes.json
```

### Module 5 — BEV Kalman 平滑

恒速模型 Kalman 滤波器，处理球员跳起/遮挡时的世界坐标跳变：

```bash
python3 run.py --video match.mp4 \
    --homography homography.npy --bev-smoothing
```

---

## 📁 项目结构

```
football-log/
├── football_log/
│   ├── protocols.py          # 插件接口
│   ├── app/                  # CLI、Web UI、VideoTrackerPipeline
│   ├── vision/               # tracker.py / deepsort_tracker.py / team_classifier*.py / pose.py / reid.py
│   ├── pitch/                # 场地：草皮/线/四边形
│   ├── world/                # homography.py / auto_calibration.py / track_filter.py
│   ├── data/                 # gsr_convert.py / yolo_convert.py
│   ├── eval/                 # eval_detection.py / eval_tracking.py / report.py
│   └── io/                   # JSONL/CSV 导出
├── scripts/
│   ├── train_module1.sh      # Module 1 训练脚本
│   ├── train_module3b.sh     # Module 3B 训练脚本
│   └── prepare_yolo_dataset.py
├── experiments/              # 实验分支（独立依赖）
├── tests/                    # 170 个单元测试
├── run.py                    # CLI 入口
└── run_web.py                # Web UI 入口（Gradio）
```

---

## 🔌 插件系统

流水线的 5 个环节均可通过 Protocol 接口替换：

| Protocol | 方法 | 默认实现 | 替换场景 |
|---|---|---|---|
| `Detector` | `detect(frame) → List[Detection]` | YOLO + ByteTrack | RT-DETR、DINO 等 |
| `TeamClassifierProto` | `instant_label(frame, bbox) → str` + `smooth_label(track_id, instant) → str` | HSV K-Means | Re-ID、球衣号码识别 |
| `PitchEstimator` | `estimate(frame) → PitchObservation` | OpenCV HSV+Hough | 学习型场地分割 |
| `WorldProjector` | `project(bbox, label) → (x_m, y_m)` | Homography / 针孔 | TVCalib 等 |
| `Exporter` | `write_frame()` / `close()` | JSONL/CSV | 数据库、API 等 |

```python
from football_log.app.runner import VideoTrackerPipeline

pipeline = VideoTrackerPipeline(
    video_path="match.mp4",
    detector=MyDetector(),
    projector=MyProjector(),
)
pipeline.run()
```

---

## 🧪 实验分支

| 方向 | 脚本 | 状态 |
|---|---|---|
| 🏟️ Roboflow Sports RADAR | `experiments/roboflow_sports_radar.py` | ✅ 已接入 |
| 📷 TVCalib | `experiments/tvcalib_infer.py` | ✅ 已接入 |
| 🌊 AuxFlow | — | 📋 候选 |
| 🎯 H-RANSAC | — | 📋 候选 |
| 🔗 TrackLab | — | 📋 候选 |

---

## 🗺️ 路线图

| 优先级 | 方向 | 状态 |
|---|---|---|
| **P0** | 基础流水线、输出 schema、插件系统 | ✅ 完成 |
| **P1** | Module 1: YOLOv8s 专项检测 + 裁判类 | ✅ 完成 |
| **P2** | Module 2: DeepSORT + Re-ID 跟踪 | ✅ 完成 |
| **P3** | Module 3A: 关键点色票投票分队 | ✅ 完成 |
| **P3** | Module 3B: 6-class 球队即检测类 | ✅ 完成 |
| **P4** | Module 4: 逐帧自动标定 → BEV 坐标 | ✅ 完成 |
| **P5** | Module 5: BEV Kalman 平滑 | ✅ 完成 |

---

## 💡 调参提示

- 🐌 **性能不足**：试试 `--detect-every-n 2` 或 `3`，跳帧检测可显著提速
- 📐 **世界坐标抖动**：加 `--bev-smoothing`，或提高 `--homography-smoothing-alpha`（最大 1.0）
- 👕 **分队不准（HSV）**：用 `--team-colors B,G,R:B,G,R` 手动指定两队颜色，或切换到 `--team-classifier keypoint`
- 👕 **分队不准（根本解法）**：用 Module 3B 权重 `--team-class-model`，队伍由检测器直接输出
- 🏟️ **场外干扰**：`--pitch-field-detect`（默认已开），配合 `--pitch-field-filter-tracks` 过滤观众席

---

## 📚 相关研究与开源参考

<details>
<summary>🏆 SoccerNet 生态</summary>

[SoccerNet](https://www.soccer-net.org) — 比利时列日大学/KAUST 维护的足球视频理解开放基准，每年 CVPR 挑战赛。

| 仓库 | 任务 |
|---|---|
| [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) | 比赛状态重建（检测 + Re-ID + 标定 + 俯视图） |
| [SoccerNet-v2](https://github.com/SoccerNet/SoccerNet) | 动作识别，550 场视频，17 类事件 |
| [sn-tracking](https://github.com/SoccerNet/sn-tracking) | 多目标跟踪，200 片段 |

</details>

<details>
<summary>📷 相机标定 / Homography</summary>

| 项目 | 说明 |
|---|---|
| [TVCalib](https://github.com/mm4spa/tvcalib) (WACV 2023) | 场地配准建模为相机标定，可微优化 [\[paper\]](https://arxiv.org/abs/2207.11709) |
| *AuxFlow* (CVIU 2026) | 锚帧关键点 + 光流时序传播 |
| *H-RANSAC* (2023) | 无特征描述子的点集 Homography 估计 [\[paper\]](https://arxiv.org/abs/2310.04912) |

</details>

<details>
<summary>🏃 球员跟踪框架</summary>

| 项目 | 说明 |
|---|---|
| [TrackLab](https://github.com/TrackingLaboratory/tracklab) | SoccerNet GSR 基线框架，模块化，支持插拔检测器/Re-ID |
| [Roboflow Sports](https://github.com/roboflow/sports) | YOLO + ByteTrack + K-Means + 关键点 + radar，完整 demo |

</details>

---

## 📜 License

MIT License — 详见 [LICENSE](LICENSE) 文件。
