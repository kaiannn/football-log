# football-log

一个用于把足球比赛视频转成结构化轨迹数据的原型项目。

当前版本重点是先跑通数据管线，并提升识别效率：视频输入 -> YOLO 跟踪(ByteTrack/BoT-SORT) -> 标签平滑 -> JSONL/CSV 导出。

## 项目结构

```
football-log/
├── football_log/
│   ├── world/          # 世界坐标：球场尺寸模型、单应矩阵、像素↔米
│   ├── vision/         # 人物/球检测与跟踪、HSV 分队与时序平滑
│   ├── io/             # 轨迹 JSONL/CSV 与 meta 导出
│   ├── ui/             # OpenCV 窗口与叠加绘制
│   └── app/            # CLI、视频流水线编排
├── run.py              # 根目录入口（与 football_log.app.cli 等价）
├── mark_pitch_quadrilateral.py   # 手工点选球场四边形（旧实验脚本）
├── detect_pitch_grass_contour.py   # 草地 HSV 轮廓估计场地（旧实验脚本）
├── video_player_step_frames.py     # 可暂停/逐帧播放（旧实验脚本）
├── video_play_simple.py            # 简单连续播放（旧实验脚本）
├── opencv_smoke_test.py            # OpenCV 环境自检（旧实验脚本）
└── requirements.txt
```

推荐入口：`python3 -m football_log.app.cli`（与 `python3 run.py` 等价）。

## 1. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 运行

### 无界面批处理（推荐，先拿数据）

```bash
python3 -m football_log.app.cli --video "/path/to/match.mp4" --no-ui --output-dir outputs --output-format both
```

### 有界面交互模式

```bash
python3 -m football_log.app.cli --video "/path/to/match.mp4"
```

可选参数：

- `--model`: 权重，如 `yolov8n.pt` / `yolov8s.pt`
- `--tracker`: `bytetrack.yaml`（默认）或 `botsort.yaml`
- `--detect-every-n`: 每 N 帧更新一次检测/跟踪（可提速）
- `--imgsz`: 推理输入尺寸（越大越准，越慢）
- `--conf`: 检测置信度阈值
- `--output-format`: `jsonl` / `csv` / `both`（默认 `both`）
- `--homography`: `3×3` 单应矩阵 `.npy`（像素 → 世界米）；提供后轨迹中会写入 `world_x_m` / `world_y_m`
- `--pitch-length-m` / `--pitch-width-m`: 球场尺寸（米），写入 `*.meta.json`，默认 105×68

## 3. 输出文件

默认输出到 `outputs/` 目录，文件名格式为 `<视频文件名>_tracks.*`：

- `*.jsonl`: 每行一个目标记录，适合后续流式处理
- `*.csv`: 表格格式，方便直接分析
- `*.meta.json`: 运行元数据（视频路径、fps、输出格式等）

记录字段：

- `frame_idx`
- `timestamp_sec`
- `track_id`
- `label`
- `x`, `y`, `w`, `h`
- `conf`
- `world_x_m`, `world_y_m`（未标定或无单应时为 `null` / CSV 中空）

## 4. 说明

- 跟踪方案已改为 YOLO `track`（默认 ByteTrack），相比多 CSRT 更稳定且更快。
- 若机器性能有限，优先尝试 `--detect-every-n 2` 或 `--detect-every-n 3`。
- 若首次运行 YOLO，可能会自动下载权重文件。

## 5. 后续发展方向

当前管线是「视频 → 轨迹 →（可选）世界坐标」。下面按优先级列出可演进方向；**尤其建议先做第 1、2 点**，否则后续事件统计与产品化容易建立在不可量化的结果上。

### 5.1 数据质量与可评估性（优先）

在扩大功能之前，先能**说清楚现在有多好、改完有没有变好**：

- **小规模真值标注**：例如若干片段上标注球是否可见、关键球员框、或 track_id 是否应保持不变，用于回归对比。
- **指标体系**：至少覆盖检测（球/人）、跟踪（ID 切换、断跟）、若启用世界坐标则再加投影误差或线与真实场地的偏差。
- **置信度与质检字段**：在输出中显式带上 `track_confidence`、`homography_valid` 等，下游只消费高置信片段，避免「全信」导致错误放大。
- **时间基准一致**：统一 `timestamp_sec` 与帧索引规则；对丢帧、变 FPS、抽帧（`detect_every_n`）要有文档化约定，避免多源数据对齐出错。

### 5.2 标定与世界坐标（优先）

从「能投一个点到地面」到「转播镜头下长期可用」：

- **自动 / 半自动单应**：用场线分割或线段检测 + 与标准球场模型匹配 + RANSAC 估计 `H`，减少对手工点选三、四个像素的依赖。
- **时序平滑**：镜头推拉、平移时单应会抖；对 `H` 或等价相机参数做卡尔曼滤波、滑动窗口优化，减少 `world_x_m` / `world_y_m` 跳变。
- **可见区域不足时的策略**：仅半场、少线时单应可能多解或不稳定；需定义降级策略（例如仅输出图像坐标、或输出带不确定度的世界坐标）。
- **与 `PitchSpec` 一致**：世界坐标轴与标定使用的世界点顺序必须一致，并在 `meta` 或单独 schema 中写清约定，便于多人协作与复现。

### 5.3 事件与语义层（轨迹之上）

在轨迹与世界坐标相对稳定后，再叠：控球/触球候选、传球与推进的粗事件、热点区与球队级统计等（依赖规则与阈值调参）。

### 5.4 工程与产品化

配置化（YAML）、批处理与任务队列、日志与失败重试、以及带时间轴与纠错的轻量界面，便于给他人使用或长期维护。

### 5.5 模型与算法升级

专用检测/分割、Re-ID 或球衣号码、多机位融合等，适合在评估体系建立后再投入，避免「感觉变好」却无法对比版本。
