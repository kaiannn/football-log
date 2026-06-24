# 测试视频说明

## 现有视频

| 文件 | 分辨率 | 帧率 | 时长 | 适配度 | 说明 |
|------|--------|------|------|--------|------|
| `manutd_30s.mp4` | 1920×1080 | 25fps | 30s | ★★★★★ | 主转播机位，单镜头全场，最适合测试 |
| `brighton_v_liverpool.mp4` | 1280×720 | 25fps | 30s | ★★★★☆ | 720p，其余条件相同 |
| `Manchester United 3-2 Nottingham Forest.mp4` | 1920×1080 | 25fps | 2min | ★★★☆☆ | 集锦视频，含镜头切换和回放，不适合世界坐标投影 |

**快速测试推荐**：用 `manutd_30s.mp4`，30 秒内跑完全流程。

---

## 什么样的视频最适合

```
✅ 最佳输入
├── 格式: MP4 (H.264)
├── 分辨率: 1920×1080
├── 帧率: 25/30 fps
├── 镜头: 转播标准半场高角度机位
├── 内容: 单镜头连续拍摄，不切镜头
└── 无回放/慢动作/字幕叠加

❌ 不适合
├── 集锦视频（多机位切换）
├── 特写镜头（只拍 1-2 个球员）
├── 回放/慢动作（帧率突变）
├── <720p 分辨率
└── 航拍/底线/极端角度
```

**为什么要求单镜头全场**：
- 世界坐标投影需要看到足够的场地标记点来拟合 homography
- 跟踪器需要连续帧才能维持 track ID
- 集锦视频的镜头切换会导致跟踪 ID 全部重置

---

## 获取更多测试视频

### 方式一：SoccerNet Tracking（推荐）

200 个 30 秒标注片段，全部是主转播机位，带跟踪 GT：

```bash
pip install SoccerNet

python3 -c "
from SoccerNet.Downloader import SoccerNetDownloader
dl = SoccerNetDownloader(LocalDirectory='data/soccernet')
dl.downloadDataTask(task='tracking', split=['train'])
"
```

数据量：~17GB（含视频 + 标注）。下载后视频在 `data/soccernet/train/` 目录。

每个片段的 GT 格式（MOT CSV）：
```
frame_id, track_id, x, y, w, h, conf, -1, -1, -1
```

### 方式二：SoccerNet GSR-2025

如果你已下载 GSR-2025 数据集（`scripts/prepare_yolo_dataset.py` 用的那个），其中的完整比赛视频也可以截取主转播机位片段。

### 方式三：从完整比赛录像截取

用 ffmpeg 从完整比赛录像中截取单镜头片段：

```bash
# 从第 120 秒开始截 30 秒，不重新编码
ffmpeg -i full_match.mp4 -ss 120 -t 30 -c copy test_clip.mp4

# 如果需要重新编码为 H.264
ffmpeg -i full_match.mp4 -ss 120 -t 30 -c:v libx264 -crf 23 test_clip.mp4
```

**截取技巧**：
- 选择**比赛进行中**的连续画面（不是死球/暂停）
- 避开场边广告牌切换、记分牌弹出的时间段
- 确保画面中能看到**整个半场**（至少能看到中圈和一个罚球区）
- 30-60 秒足够测试，不需要截太长

### 方式四：YouTube 转播完整场

搜索 "Premier League full match" 或 "La Liga full match"，用 yt-dlp 下载：

```bash
# 安装 yt-dlp
pip install yt-dlp

# 下载（选择 1080p）
yt-dlp -f "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]" \
    "https://www.youtube.com/watch?v=VIDEO_ID" -o full_match.mp4

# 截取 30 秒片段
ffmpeg -i full_match.mp4 -ss 300 -t 30 -c copy test_clip.mp4
```

---

## 检查视频属性

```bash
# 用 ffprobe 查看视频信息
ffprobe -v quiet -print_format json -show_streams test_clip.mp4

# 或用 Python
python3 -c "
import cv2
cap = cv2.VideoCapture('test_clip.mp4')
print(f'分辨率: {int(cap.get(3))}x{int(cap.get(4))}')
print(f'帧率: {cap.get(5):.1f} fps')
print(f'总帧数: {int(cap.get(7))}')
cap.release()
"
```

---

## 典型测试命令

```bash
# 最简测试（只有轨迹，无世界坐标）
python run.py --video test_videos/manutd_30s.mp4 --no-ui --output-format both

# 带世界坐标（关键点模型自动标定）
python run.py --video test_videos/manutd_30s.mp4 --no-ui \
    --pitch-keypoint-model runs/roboflow/football-pitch-detection.pt \
    --bev-smoothing --save-video --save-radar

# 带性能分析
python run.py --video test_videos/manutd_30s.mp4 --no-ui \
    --save-video  # 结束后会打印各组件耗时占比
```
