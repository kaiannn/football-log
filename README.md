<div align="center">

# ⚽ football-log

**视频进，数据出 —— 把足球比赛视频变成可分析的结构化轨迹**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://img.shields.io/github/actions/workflow/status/kaiannn/football-log/ci.yml?style=flat-square&label=CI)](https://github.com/kaiannn/football-log/actions)
[![Tests](https://img.shields.io/badge/tests-226-2EA44F?style=flat-square)](#-测试)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

[快速开始](#-快速开始) · [工作流程](#-工作流程) · [使用方法](#-使用方法) · [研究模块](#-研究模块) · [插件系统](#-插件系统)

</div>

---

## 它做什么？

给你一段足球比赛视频，它输出每帧每个球员/球/裁判的 **位置、队伍标签、跟踪 ID、世界坐标**。
格式是标准 JSONL/CSV，直接用 Pandas、DuckDB 或任何 BI 工具继续分析。

```
输入: match.mp4
输出: match_tracks.jsonl

{"frame_idx": 120, "id": 7, "label": "Team A", "bbox": [830,420,45,90], "world_x_m": 34.2, "world_y_m": 51.8}
{"frame_idx": 120, "id": 12, "label": "Team B", "bbox": [1200,380,42,88], "world_x_m": 71.5, "world_y_m": 22.1}
{"frame_idx": 120, "id": -1, "label": "Ball", "bbox": [955,410,12,12], "world_x_m": 52.4, "world_y_m": 34.0}
...
```

拿到数据后，你可以做：热力图、跑动距离、阵型变化、冲刺分析、比赛事件检测…… 随你。

---

## 工作流程

整个管线分六步，逐帧执行：

```
┌─────────────────────────────────────────────────────────────────────┐
│ ① 读帧                                                            │
│   cv2.VideoCapture 逐帧解码视频                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ② 检测 + 跟踪（每 N 帧执行，中间帧复用结果）                        │
│                                                                      │
│   ByteTrack（默认）→ 最快                                           │
│   DeepSORT        → 外观嵌入关联，遮挡时 ID 更稳，慢 2-3x           │
│   6-class 模型    → 直接输出 Team A/B/裁判/球，跳过颜色分队         │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ③ 分队 — 判断每个球员属于哪一队                                     │
│                                                                      │
│   HSV K-Means（默认）→ 提取球衣颜色特征，自动聚类 + 时序投票        │
│   Keypoint + LAB     → 姿态关键点取躯干，LAB 颜色投票，更稳但更慢   │
│   6-class 模型       → 检测器直接输出队伍标签，跳过此步              │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ④ 世界坐标投影 — 像素位置 → 球场上的米制坐标                        │
│                                                                      │
│   需要一个"标定"来建立像素和球场的对应关系：                         │
│   • 手标关键帧 + 光流传播    → 适合移动镜头                         │
│   • 场地关键点模型（32点）   → 全自动，推荐                         │
│   • 静态 homography 文件     → 固定机位                             │
│   • 针孔相机标定             → 精度最高                             │
│                                                                      │
│   + 跳起修正：检测到球员跳起时，锚定到历史地面水平                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ⑤ BEV 卡尔曼平滑（可选）                                           │
│                                                                      │
│   世界坐标空间的常速度卡尔曼滤波，压制单帧投影噪声                   │
│   马氏距离门控拒绝跳变异常值                                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ⑥ 导出                                                             │
│                                                                      │
│   JSONL/CSV 轨迹文件 + meta.json                                    │
│   可选：标注叠加视频 (_overlay.mp4) + 雷达俯视图 (_radar.mp4)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 安装

```bash
git clone https://github.com/kaiannn/football-log.git
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

只需要一个视频文件，不需要标定、不需要额外数据集。

---

## 使用方法

### 典型场景

<details>
<summary><b>只要轨迹，不要世界坐标</b></summary>

```bash
python3 run.py --video match.mp4 --no-ui --output-format both
```

输出每帧每个目标的 bbox、label、track_id、conf。不需要任何标定。

</details>

<details>
<summary><b>要世界坐标（推荐方式：场地关键点模型）</b></summary>

```bash
python3 run.py --video match.mp4 --no-ui \
    --pitch-keypoint-model runs/roboflow/football-pitch-detection.pt \
    --bev-smoothing
```

模型自动检测场地 32 个关键点 → RANSAC 拟合单应矩阵 → 投影到世界坐标 → Kalman 平滑。

</details>

<details>
<summary><b>要世界坐标（手动标定方式）</b></summary>

```bash
# 方式一：单应矩阵文件
python3 run.py --video match.mp4 --no-ui --homography homography.npy

# 方式二：针孔标定文件
python3 run.py --video match.mp4 --no-ui --camera-calib camera_calib.json

# 方式三：关键帧自动标定
python3 run.py --video match.mp4 --no-ui \
    --auto-calibration-keyframes keyframes.json
```

</details>

<details>
<summary><b>遮挡严重，需要更稳定的 ID</b></summary>

```bash
pip install deep-sort-realtime
python3 run.py --video match.mp4 --no-ui --tracker deepsort
```

DeepSORT 用外观嵌入做重识别，争顶/铲球/密集防守时 ID 切换更少。

</details>

<details>
<summary><b>用 6-class 模型跳过颜色分队</b></summary>

```bash
python3 run.py --video match.mp4 --no-ui \
    --team-class-model runs/module3b_v1/weights/best.pt
```

模型直接输出 Team A / Team B / 裁判 / 球，不需要 K-Means 颜色聚类。

</details>

<details>
<summary><b>想看界面预览</b></summary>

```bash
python3 run.py --video match.mp4
```

按 `q` 退出。

</details>

### 性能调优

| 场景 | 做法 |
|------|------|
| 太慢 | `--detect-every-n 2` 或 `3`，跳帧检测可显著提速 |
| 世界坐标抖动 | `--bev-smoothing`，或提高 `--homography-smoothing-alpha` |
| 分队不准 | `--team-colors B,G,R;B,G,R` 手动指定颜色，或用 `--team-class-model` |
| 球检不到 | `--ball-model runs/roboflow/football-ball-detection.pt`，专用球模型 |

### CLI 参数速查

<details>
<summary>展开查看全部参数</summary>

**基础 I/O**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--video` | 必填 | 输入视频路径 |
| `--output-dir` | `outputs` | 输出目录 |
| `--output-format` | `both` | `jsonl` / `csv` / `both` |
| `--no-ui` | — | 无窗口批处理模式 |
| `--save-video` | 关 | 保存标注叠加视频 |
| `--save-radar` | 关 | 保存雷达俯视图视频 |

**检测与跟踪**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | `yolov8n.pt` | YOLO 权重路径 |
| `--tracker` | `bytetrack` | `bytetrack` / `botsort` / `botsort+reid` / `deepsort` / 自定义 `.yaml` |
| `--conf` | `0.3` | 检测置信度阈值 |
| `--imgsz` | `640` | 推理边长（像素） |
| `--detect-every-n` | `1` | 每 N 帧检测一次 |

**类别 ID 映射**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--player-class-id` | `0` | 球员类别 ID |
| `--ball-class-id` | `32` | 球类别 ID |
| `--referee-class-id` | 无 | 裁判类别 ID |

**分队**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--team-classifier` | `hsv` | `hsv` / `keypoint` |
| `--team-colors` | 无 | 手动指定颜色 `B,G,R;B,G,R` |
| `--team-class-model` | 无 | 6-class YOLO 权重路径 |

**世界坐标标定**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--auto-calibration-keyframes` | 无 | 关键帧 JSON（优先级最高） |
| `--homography-sequence` | 无 | 逐帧单应矩阵 `.npy` |
| `--homography` | 无 | 固定单应矩阵 `.npy` |
| `--camera-calib` | 无 | 针孔标定 JSON/YAML |
| `--pitch-keypoint-model` | 无 | 场地关键点模型路径 |
| `--homography-smoothing-alpha` | `0.3` | EMA 平滑系数 |

**BEV 平滑**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--bev-smoothing` | 关 | 启用 BEV Kalman 滤波 |

**其他**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--pitch-length-m` / `--pitch-width-m` | `105` / `68` | 球场尺寸 |
| `--ball-model` | 无 | 专用球检测模型 |
| `--ball-slicer` | 关 | InferenceSlicer 分块检测（更准但更慢） |
| `--save-debug-overlay` | 关 | 叠加草地检测调试层 |

</details>

---

## 输出格式

默认输出到 `outputs/<视频名>_tracks.jsonl|csv` 与 `<视频名>_tracks.meta.json`。

```jsonl
{"frame_idx": 120, "timestamp_sec": 4.0, "track_id": 3, "label": "Team A", "x": 342, "y": 188, "w": 48, "h": 112, "conf": 0.87, "world_x_m": 23.4, "world_y_m": 51.2}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `frame_idx` | int | 帧序号 |
| `timestamp_sec` | float | 时间戳（秒） |
| `track_id` | int | 跟踪 ID（球为 -1） |
| `label` | str | `Team A` / `Team B` / `Ball` / `Referee` / `Player` |
| `x, y, w, h` | int | 边界框（像素） |
| `conf` | float | 检测置信度 |
| `world_x_m`, `world_y_m` | float? | 世界坐标（米），需标定 |
| `world_x_m_smoothed`, `world_y_m_smoothed` | float? | 平滑后世界坐标，需 `--bev-smoothing` |

---

## 研究模块

五个可选模块，在基础管线之上叠加：

### Module 1 — 专项检测权重

SoccerNet GSR-2025 微调 YOLOv8s，新增裁判类：

| 类别 | mAP@50 |
|---|---|
| player | 0.978 |
| ball | 0.867 |
| referee | 0.993 |

```bash
python3 run.py --video match.mp4 \
    --model runs/module1_v1/weights/best.pt \
    --player-class-id 0 --ball-class-id 1 --referee-class-id 2
```

### Module 2 — DeepSORT + Re-ID

独立 DeepSORT，用外观嵌入修复跳帧 Kalman 问题：

```bash
pip install deep-sort-realtime
python3 run.py --video match.mp4 --tracker deepsort
```

### Module 3A — 关键点分队

姿态关键点提取躯干区域 + LAB 颜色投票，比 HSV 全框聚类更抗干扰：

```bash
python3 run.py --video match.mp4 --team-classifier keypoint
```

### Module 3B — 6-class 球队即检测

YOLOv8n 直接输出队伍标签：

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

### Module 4 — 逐帧自动标定

关键帧手标 + Lucas-Kanade 光流传播，适合移动镜头：

```bash
python3 run.py --video match.mp4 \
    --auto-calibration-keyframes keyframes.json
```

### Module 5 — BEV Kalman 平滑

恒速卡尔曼滤波，处理跳起/遮挡导致的世界坐标跳变：

```bash
python3 run.py --video match.mp4 \
    --homography homography.npy --bev-smoothing
```

---

## 插件系统

7 个 Protocol 接口，自定义组件即插即用：

```python
from football_log.app.runner import VideoTrackerPipeline

pipeline = VideoTrackerPipeline(
    video_path="match.mp4",
    detector=MyDetector(),         # 替换检测器
    team_cls=MyClassifier(),       # 替换分队器
    projector=MyProjector(),       # 替换坐标投影
    exporter=MyExporter(),         # 替换导出
)
pipeline.run()
```

| Protocol | 默认实现 | 替换场景 |
|---|---|---|
| `Detector` | YOLO + ByteTrack | RT-DETR、DINO、自定义检测器 |
| `TeamClassifierProto` | HSV K-Means | Re-ID、球衣号码识别 |
| `PitchEstimator` | OpenCV HSV+Hough | 学习型场地分割 |
| `WorldProjector` | Homography / 针孔 | TVCalib 等 |
| `BallDetectorProto` | Roboflow 球模型 | 专用小目标模型 |
| `PitchCalibratorProto` | 32 点关键点模型 | TVCalib、SoccerNet 标定 |
| `Exporter` | JSONL/CSV | 数据库、实时 API |

---

## 项目结构

```
football_log/
├── app/
│   ├── runner.py              # 核心管线 VideoTrackerPipeline
│   ├── cli.py                 # CLI 入口
│   └── web.py                 # Gradio Web UI
├── vision/
│   ├── tracker.py             # ByteTrack 跟踪
│   ├── deepsort_tracker.py    # DeepSORT 跟踪
│   ├── team_classifier.py     # HSV K-Means 分队
│   ├── team_classifier_keypoint.py  # 关键点 + LAB 分队
│   ├── label_utils.py         # 共享标签工具
│   ├── ball_detector.py       # 专用球检测
│   ├── pitch_keypoint_detector.py   # 场地关键点检测
│   └── pose.py                # 姿态估计
├── world/
│   ├── homography.py          # 单应变换
│   ├── auto_calibration.py    # 逐帧自动标定
│   ├── track_filter.py        # BEV 卡尔曼平滑
│   └── pinhole_ground.py      # 针孔相机标定
├── pitch/
│   ├── field_estimator.py     # 草地/场线/四边形检测
│   └── observation.py         # 场地观测数据
├── io/
│   └── export.py              # JSONL/CSV 导出
├── ui/
│   ├── overlay.py             # 标注叠加绘制
│   └── radar.py               # 雷达俯视图渲染
├── protocols.py               # 7 个 Protocol 接口定义
├── configs/                   # 跟踪器 YAML 配置
├── data/                      # 数据集转换工具
└── eval/                      # 检测/跟踪/投影评估
scripts/
├── train_module1.sh           # Module 1 训练
├── train_module3b.sh          # Module 3B 训练
└── prepare_yolo_dataset.py    # 数据集准备
tests/                         # 226 个单元测试
experiments/                   # TVCalib / Roboflow 实验
```

---

## 测试

```bash
pip install -r requirements-dev.txt
pytest tests/ -v

# 226 passed
```

覆盖核心算法（单应变换、卡尔曼滤波、分队聚类、坐标标定、数据导出）。
CI 在 Python 3.10 + 3.12 上自动运行。

---

## 相关研究

<details>
<summary>SoccerNet 生态</summary>

[SoccerNet](https://www.soccer-net.org) — 足球视频理解开放基准，每年 CVPR 挑战赛。

| 仓库 | 任务 |
|---|---|
| [sn-gamestate](https://github.com/SoccerNet/sn-gamestate) | 比赛状态重建（检测 + Re-ID + 标定 + 俯视图） |
| [SoccerNet-v2](https://github.com/SoccerNet/SoccerNet) | 动作识别，550 场视频 |
| [sn-tracking](https://github.com/SoccerNet/sn-tracking) | 多目标跟踪 |

</details>

<details>
<summary>相机标定 / Homography</summary>

| 项目 | 说明 |
|---|---|
| [TVCalib](https://github.com/mm4spa/tvcalib) (WACV 2023) | 场地配准建模为相机标定 [\[paper\]](https://arxiv.org/abs/2207.11709) |
| [H-RANSAC](https://arxiv.org/abs/2310.04912) (2023) | 无特征描述子的 Homography 估计 |

</details>

<details>
<summary>球员跟踪框架</summary>

| 项目 | 说明 |
|---|---|
| [TrackLab](https://github.com/TrackingLaboratory/tracklab) | SoccerNet GSR 基线框架 |
| [Roboflow Sports](https://github.com/roboflow/sports) | YOLO + ByteTrack + K-Means + radar 完整 demo |

</details>

---

## License

MIT — 详见 [LICENSE](LICENSE)。
