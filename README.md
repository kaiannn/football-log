<div align="center">

# ⚽ football-log

**视频进，数据出 —— 把足球比赛视频变成可分析的结构化轨迹**

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.2.0-2EA44F?style=flat-square)](https://github.com/kai/football-log)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Ultralytics](https://img.shields.io/badge/YOLO-Ultralytics-111F68?style=flat-square&logo=yolo&logoColor=white)](https://ultralytics.com)
[![Gradio](https://img.shields.io/badge/UI-Gradio-FF7C00?style=flat-square&logo=gradio&logoColor=white)](https://gradio.app)

给一段足球比赛视频 → 输出带球员/球标签、track ID、时间戳的 JSONL/CSV 轨迹文件<br>
直接用 Pandas、DuckDB 或任何 BI 工具继续分析

[🚀 快速开始](#-快速开始) · [📖 文档](#-使用方法) · [🔌 插件](#-插件系统) · [🧪 实验分支](#-实验分支) · [📚 相关研究](#-相关研究与开源参考)

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
| 🔧 可选 | 跟踪器配置 | `--tracker` | `bytetrack.yaml` / `botsort.yaml` |
| 🔧 可选 | 单应矩阵 | `--homography` | `3×3` `.npy`，像素→世界米 |
| 🔧 可选 | 针孔标定 | `--camera-calib` | JSON/YAML，与单应二选一 |
| 🔧 可选 | 场地估计 | `--pitch-field-detect` | 草皮+线+四边形检测 |

### 常用命令

<details>
<summary>🎬 最小化运行（只要视频）</summary>

```bash
python3 -m football_log.app.cli \
  --video "/path/to/match.mp4" \
  --no-ui \
  --output-dir outputs \
  --output-format both
```

</details>

<details>
<summary>📐 带世界坐标</summary>

```bash
# 方式一：单应矩阵
python3 -m football_log.app.cli \
  --video "/path/to/match.mp4" --no-ui \
  --homography /path/to/homography.npy

# 方式二：针孔标定文件
python3 -m football_log.app.cli \
  --video "/path/to/match.mp4" --no-ui \
  --camera-calib /path/to/camera_calib.json
```

</details>

<details>
<summary>🌿 场地估计 + 草皮过滤</summary>

```bash
python3 -m football_log.app.cli \
  --video "/path/to/match.mp4" --no-ui \
  --pitch-field-detect \
  --pitch-field-every-n 15 \
  --pitch-field-filter-tracks
```

</details>

<details>
<summary>🖥️ 有界面预览（按 q 退出）</summary>

```bash
python3 -m football_log.app.cli --video "/path/to/match.mp4"
```

</details>

<details>
<summary>📏 启发式标定物拟合（修正尺度偏差）</summary>

当已有的 `homography.npy` 存在尺度偏差时，可用已知尺寸矩形做联合拟合：

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

也可从 JSON 文件读入多个参考：

```bash
python3 -m football_log.app.calibrate_reference \
  --video "/path/to/match.mp4" \
  --homography "/path/to/homography.npy" \
  --refs-json "/path/to/refs.json" \
  --robust-loss cauchy \
  --out-homography "/path/to/homography_refit.npy"
```

</details>

### CLI 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--video` | 必填 | 输入视频路径 |
| `--output-dir` | `outputs` | 输出目录 |
| `--output-format` | `both` | `jsonl` / `csv` / `both` |
| `--model` | `yolov8n.pt` | YOLO 权重 |
| `--tracker` | `bytetrack.yaml` | 跟踪配置 |
| `--conf` | `0.3` | 检测置信度 |
| `--imgsz` | `640` | 推理边长 |
| `--detect-every-n` | `1` | 每 N 帧检测一次（可提速） |
| `--no-ui` | 关 | 无窗口批处理 |
| `--homography` | 无 | `3×3` `.npy`，像素→世界米 |
| `--camera-calib` | 无 | 针孔+地面标定 JSON/YAML |
| `--pitch-length-m` / `--pitch-width-m` | 105 / 68 | 写入 meta |
| `--pitch-field-detect` | 关 | 启用场地估计 |
| `--pitch-field-every-n` | `15` | 场地估计更新间隔 |
| `--pitch-field-filter-tracks` | 关 | 用草皮掩膜过滤检测 |

---

## 📤 输出格式

默认输出到 `outputs/<视频名>_tracks.jsonl|csv` 与 `<视频名>_tracks.meta.json`。

```jsonl
{"frame_idx": 120, "timestamp_sec": 4.0, "track_id": 3, "label": "player", "x": 342, "y": 188, "w": 48, "h": 112, "conf": 0.87, "team": 0, "world_x_m": 23.4, "world_y_m": -11.2}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `frame_idx` | int | 帧序号 |
| `timestamp_sec` | float | 时间戳（秒） |
| `track_id` | int | 跟踪 ID |
| `label` | str | `player` / `ball` |
| `x, y, w, h` | int | 边界框 |
| `conf` | float | 检测置信度 |
| `team` | int | 分队编号（0/1） |
| `world_x_m`, `world_y_m` | float? | 世界坐标（米），需标定 |

---

## 📁 项目结构

```
football-log/
├── football_log/
│   ├── protocols.py    # 插件接口：Detector / TeamClassifier / WorldProjector / Exporter
│   ├── app/            # CLI、Web UI、VideoTrackerPipeline
│   ├── vision/         # YOLO 跟踪、LAB K-Means 分队 + 时序平滑
│   ├── pitch/          # 场地：草皮/线/四边形（OpenCV）
│   ├── world/          # Homography、针孔标定 PinholeGroundProjector
│   ├── io/             # JSONL/CSV 导出、meta
│   └── ui/             # OpenCV 叠加绘制
├── experiments/        # 实验分支（独立依赖）
├── run.py              # CLI 入口
├── run_web.py          # Web UI 入口（Gradio）
└── requirements.txt
```

---

## 🔌 插件系统

流水线的 5 个环节均可通过 Protocol 接口替换，无需继承基类，只需实现对应方法签名：

| Protocol | 方法 | 默认实现 | 替换场景 |
|---|---|---|---|
| `Detector` | `detect(frame) → List[Detection]` | YOLO + ByteTrack | 换用 RT-DETR、DINO 等 |
| `TeamClassifierProto` | `classify(frame, detection) → str` | LAB K-Means | 换用 Re-ID、球衣号码识别 |
| `PitchEstimator` | `estimate(frame) → PitchObservation` | OpenCV HSV+Hough | 换用学习型场地分割 |
| `WorldProjector` | `project(bbox, label) → (x_m, y_m)` | Homography / 针孔 | 换用 TVCalib 等 |
| `Exporter` | `write_frame()` / `close()` | JSONL/CSV | 换成数据库、API 等 |

### 使用方式

**默认（零配置）：**

```python
from football_log.app.runner import VideoTrackerPipeline

pipeline = VideoTrackerPipeline(video_path="match.mp4")
pipeline.run()
```

**注入自定义组件：**

```python
from football_log.app.runner import VideoTrackerPipeline
from football_log.protocols import Detection

class MyDetector:
    def detect(self, frame):
        return [Detection(track_id=1, bbox=(x, y, w, h), label="player", conf=0.9)]

class MyProjector:
    def project(self, bbox, label):
        return (world_x, world_y)

pipeline = VideoTrackerPipeline(
    video_path="match.mp4",
    detector=MyDetector(),
    projector=MyProjector(),
)
pipeline.run()
```

### 数据结构

所有组件之间通过 `Detection` dataclass 传递数据：

```python
@dataclass
class Detection:
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    label: str                        # "Team A" / "Team B" / "Ball" / ...
    conf: float = 0.0
    box_color: Optional[Tuple[int, int, int]] = None
    world_x_m: Optional[float] = None
    world_y_m: Optional[float] = None
```

---

## 🧪 实验分支

外部研究方案的尝试放在 `experiments/` 下，**独立脚本 + 独立依赖**，不污染主流水线。

| 方向 | 脚本 | 状态 | 场景 |
|---|---|---|---|
| 🏟️ Roboflow Sports RADAR | `experiments/roboflow_sports_radar.py` | ✅ 已接入 | 快速对比完整足球分析 demo |
| 📷 TVCalib | `experiments/tvcalib_infer.py` | ✅ 已接入 | 验证更强的相机标定效果 |
| 🌊 AuxFlow | — | 📋 候选 | 时序稳定的 Homography 传播 |
| 🎯 H-RANSAC | — | 📋 候选 | 无特征点时的鲁棒估计 |
| 🔗 TrackLab | — | 📋 候选 | 更完整的 MOT / Re-ID 框架 |

<details>
<summary>🏟️ Roboflow Sports RADAR 详情</summary>

直接包装 `roboflow/sports` 官方 `examples/soccer` 示例，用本项目视频快速试官方 RADAR 流程。

**行为**：首次运行自动 clone → 安装依赖 → 执行 setup → 调用 `main.py --mode RADAR`

```bash
python3 experiments/roboflow_sports_radar.py \
  --source "/path/to/match.mp4" \
  --device mps
```

| 参数 | 说明 |
|---|---|
| `--target` | 输出视频路径，默认 `<source>_roboflow_radar.mp4` |
| `--repo-dir` | 本地缓存仓库位置 |
| `--skip-install` | 跳过 clone / install / setup |

**价值**：最快看到"检测 + 分队 + pitch keypoints + radar"完整效果，便于横向对比。<br>
**风险**：依赖较重，上游可能 breaking change。

</details>

<details>
<summary>📷 TVCalib 详情</summary>

复用官方实现，准备仓库 + conda 环境 + 权重，打开 `inference.ipynb` 供直接跑推理。

**行为**：首次运行自动 clone → 创建 conda 环境 → 下载权重 → 打开 notebook

```bash
python3 experiments/tvcalib_infer.py
```

| 参数 | 说明 |
|---|---|
| `--env-name` | 自定义 conda 环境名 |
| `--weights-path` | 已有权重时直接指定 |
| `--skip-install` | 跳过环境准备 |

**价值**：若标定效果提升明显，可把更稳定的 `homography.npy` 输回主流程。<br>
**约束**：偏研究代码，依赖 conda + notebook，工程整合成本较高。

</details>

<details>
<summary>📋 接入策略</summary>

先做"实验分支"而不是直接改主线：

- **产品侧**：先验证外部成果是否真能带来更好效果，避免过早引入复杂依赖
- **工程侧**：各方案有独立依赖栈，直接揉进 `requirements.txt` 会让轻量流程变重

收敛优先级：

1. 📷 **TVCalib 输出 → 标准 homography / camera-calib 文件** → 接入 `run.py`
2. 🏟️ **Roboflow Sports radar / team classification** → 抽成可复用模块
3. 🔗 **TrackLab / AuxFlow / H-RANSAC** → 评估是否值得第二批实验

对比指标：标定精度（重投影误差 / 跨帧 jitter）· 跟踪质量（ID switch / 遮挡恢复）· 分队纯度 · 安装成本

</details>

---

## 🗺️ 路线图

| 优先级 | 方向 | 状态 |
|---|---|---|
| **P0** | 基础流水线做稳，输出 schema 固定，入口清晰 | ✅ 完成 |
| **P1** | 分队分类、场地过滤、简单标定 | ✅ 完成 |
| **P1.5** | 插件系统（Protocol 接口 + Pipeline 注入） | ✅ 完成 |
| **P2** | TVCalib / Roboflow Sports 实验对比验证 | 🔄 进行中 |
| **P3** | 根据实验结果决定是否吸收进主线 | ⏳ 待评估 |
| **P4** | SuperGlue、LK 光流、AuxFlow 等更多增强 | 📋 候选 |

---

## 📚 相关研究与开源参考

<details>
<summary>🏆 SoccerNet 生态</summary>

[SoccerNet](https://www.soccer-net.org) — 比利时列日大学/KAUST 维护的足球视频理解开放基准，每年 CVPR 挑战赛。

| 仓库 | 任务 | 说明 |
|---|---|---|
| [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) | 比赛状态重建 | 检测 + Re-ID + 标定 + 俯视图，含 GS-HOTA 指标 |
| [SoccerNet-v2](https://github.com/SoccerNet/SoccerNet) | 动作识别 | 550 场视频，17 类事件标注 |
| [sn-tracking](https://github.com/SoccerNet/sn-tracking) | 多目标跟踪 | 200 片段，针对遮挡与外观相似 |

</details>

<details>
<summary>📷 相机标定 / Homography</summary>

| 项目 | 说明 |
|---|---|
| [TVCalib](https://github.com/mm4spa/tvcalib) (WACV 2023) | 场地配准建模为相机标定，可微优化位姿和焦距 [\[paper\]](https://arxiv.org/abs/2207.11709) |
| *AuxFlow* (CVIU 2026) | 锚帧关键点 + 光流时序传播，提升 Homography 时间稳定性 |
| *H-RANSAC* (2023) | 无特征描述子的点集 Homography 估计 [\[paper\]](https://arxiv.org/abs/2310.04912) |

</details>

<details>
<summary>🏃 球员跟踪框架</summary>

| 项目 | 说明 |
|---|---|
| [TrackLab](https://github.com/TrackingLaboratory/tracklab) | SoccerNet GSR 基线框架，模块化，支持插拔检测器/Re-ID |
| [Roboflow Sports](https://github.com/roboflow/sports) | YOLO + ByteTrack + K-Means + 关键点 + radar，完整 demo |

</details>

<details>
<summary>⚡ 动作识别 / 事件检测</summary>

| 项目 | 说明 |
|---|---|
| *FAANTRA* (CVPR CVSports 2025) | Transformer 足球动作预判，提前 5-10 秒预测 [\[paper\]](https://arxiv.org/abs/2504.12021) |

</details>

<details>
<summary>📊 开放数据集</summary>

| 数据集 | 说明 |
|---|---|
| [StatsBomb Open Data](https://github.com/statsbomb/open-data) | 免费结构化赛事数据（传球/射门/位置等 20+ 事件类型），无视频 |

</details>

---

## 💡 调参提示

- 🐌 **性能不足**：试试 `--detect-every-n 2` 或 `3`，跳帧检测可显著提速
- 📐 **世界坐标抖动**：标定质量是关键，单应与 `PitchSpec` 的世界轴约定需与标定时选点一致
- 👕 **分队不准**：默认 LAB K-Means 自动聚类 + 时序平滑；也可 `--team-colors` 手动指定两队 BGR 颜色

---

## 📜 License

MIT License — 详见 [LICENSE](LICENSE) 文件。
