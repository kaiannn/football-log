import cv2
import numpy as np
import os

# --- 检查并导入 ultralytics ---
try:
    from ultralytics import YOLO
    
    class YoloDetector:
        """使用YOLO模型进行通用目标检测"""
        def __init__(self):
            print("正在加载 YOLOv8n 模型...")
            self.model = YOLO('yolov8n.pt') 
            self.person_class_id = 0
            self.ball_class_id = 32
            self.target_classes = [self.person_class_id, self.ball_class_id]
            print("YOLOv8n 模型加载完成。")

        def detect(self, frame):
            results = self.model(frame, classes=self.target_classes, conf=0.3, verbose=False)
            detections = []
            
            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy() # x1, y1, x2, y2
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls in zip(boxes, confs, clss):
                    # 转换为 (x, y, w, h)
                    x1, y1, x2, y2 = map(int, box)
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(((x1, y1, w, h), conf, cls))
            return detections

except ImportError:
    print("错误：未安装 'ultralytics' 库。请运行 'pip install ultralytics' 安装。")
    # 如果没有 YOLO，为了程序能运行，提供一个占位符类
    class YoloDetector:
        def __init__(self): 
            print("YOLO 模型未加载，跟踪功能将无法初始化。请安装 ultralytics 库。")
        def detect(self, frame): 
            return []

# =================================================================
# 1. ObjectTrackerManager 类 (目标跟踪器管理)
# =================================================================
class ObjectTrackerManager:
    def __init__(self):
        # 存储当前正在跟踪的所有目标：{ID: {'tracker': tracker, 'bbox': bbox, 'label': label}}
        self.trackers = {}
        self.next_id = 0

    def init_tracker(self, frame, bbox, label):
        """为新检测到的目标初始化一个跟踪器"""
        # 推荐使用 CSRT，性能和准确度更好
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        
        obj_id = self.next_id
        self.trackers[obj_id] = {'tracker': tracker, 'bbox': bbox, 'label': label}
        self.next_id += 1
        return obj_id

    def update_trackers(self, frame):
        """更新所有活动的跟踪器"""
        updated_trackers = {}
        
        for obj_id, data in self.trackers.items():
            tracker = data['tracker']
            label = data['label']
            
            success, bbox = tracker.update(frame)

            if success:
                # 跟踪成功
                updated_trackers[obj_id] = {'tracker': tracker, 'bbox': bbox, 'label': label}
            
        self.trackers = updated_trackers
        # 返回当前跟踪列表的深拷贝信息
        return [{'id': k, 'bbox': v['bbox'], 'label': v['label']} for k, v in self.trackers.items()]

    def get_tracked_bboxes(self):
        """返回所有跟踪目标的边界框和标签"""
        return [{'id': k, 'bbox': v['bbox'], 'label': v['label']} for k, v in self.trackers.items()]

    def reset(self):
        """清除所有跟踪器"""
        self.trackers = {}
        self.next_id = 0


# =================================================================
# 2. VideoMarkerAndDetector 类 (主程序逻辑)
# =================================================================
class VideoTrackerApp:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise SystemExit("无法打开视频文件")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.delay = int(1000 / self.fps)
        self.paused = False
        self.frame = None
        self.frame_idx = 0

        # --- 跟踪模块 ---
        self.object_tracker_manager = ObjectTrackerManager()
        self.yolo_detector = YoloDetector() # YOLO 检测器
        self.detection_triggered = False 
        self.tracking_enabled = False 

        # --- 颜色检测阈值（仅用于 YOLO 后处理，区分队伍） ---
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_blue = np.array([100, 70, 50])
        self.upper_blue = np.array([130, 255, 255])
        self.lower_ball = np.array([20, 100, 100]) # 亮黄色
        self.upper_ball = np.array([30, 255, 255])
        
        # --- 手动修正属性 ---
        self.manual_reselect_mode = False 
        self.reselect_bbox_start = None   
        self.reselect_bbox_end = None     
        self.target_id_to_fix = 0 # 要修正的目标 ID

        cv2.namedWindow('Video Tracker')
        cv2.setMouseCallback('Video Tracker', self.mouse_callback)
        
        # 创建一个用于选择 ID 的 Trackbar，简化为 Target ID
        cv2.createTrackbar('Target ID to Fix (C)', 'Video Tracker', 0, 100, lambda x: self._set_target_id(x))
        
        self._set_initial_frame()
        
    def _set_target_id(self, val):
        """Trackbar 回调函数，用于设置要修正的 ID"""
        self.target_id_to_fix = val

    def _set_initial_frame(self):
        """读取并设置第一帧"""
        ret, self.frame = self.cap.read()
        if not ret:
            raise SystemExit("无法读取视频帧")
        
        self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.current_frame = self.frame.copy()

    # --- 按照颜色分队，现在只有红色和蓝色 ---
    def _analyze_team_color(self, frame, bbox):
        """对给定边界框内的区域进行颜色分析，确定队伍标签。"""
        x, y, w, h = bbox
        
        # 裁剪边界框区域，分析上半身或中间部分 (简化处理)
        y_center = y + int(h * 0.25)
        h_analyze = int(h * 0.5)
        
        if y_center + h_analyze > frame.shape[0] or x + w > frame.shape[1] or x < 0 or y_center < 0:
             return 'Unknown Player'

        analysis_area = frame[y_center:y_center + h_analyze, x:x + w]
        if analysis_area.size == 0:
            return 'Unknown Player'

        hsv_area = cv2.cvtColor(analysis_area, cv2.COLOR_BGR2HSV)
        
        # 计算红色像素数量 (使用两个阈值范围)
        red_mask = cv2.inRange(hsv_area, self.lower_red1, self.upper_red1) | \
                   cv2.inRange(hsv_area, self.lower_red2, self.upper_red2)
        red_count = np.sum(red_mask > 0)

        # 计算蓝色像素数量
        blue_mask = cv2.inRange(hsv_area, self.lower_blue, self.upper_blue)
        blue_count = np.sum(blue_mask > 0)
        
        MIN_PIXEL_COUNT = 50 
        
        if red_count > blue_count and red_count > MIN_PIXEL_COUNT:
            return 'Red Player' 
        elif blue_count > red_count and blue_count > MIN_PIXEL_COUNT:
            return 'Blue Player'
        
        return 'Unknown Player'

    # --- 初始检测逻辑 (YOLO) ---
    def _run_initial_detection(self, frame):
        """
        使用 YOLO 检测目标，并使用颜色后处理确定队伍，最后应用 24 个目标的限制。
        返回：[(bbox, label), ...]
        """
        yolo_detections = self.yolo_detector.detect(frame)
        
        all_targets = []
        
        for bbox, conf, cls in yolo_detections:
            if cls == self.yolo_detector.person_class_id: 
                label = self._analyze_team_color(frame, bbox)
                if label != 'Unknown Player':
                    all_targets.append((bbox, conf, label))
            elif cls == self.yolo_detector.ball_class_id: 
                all_targets.append((bbox, conf, 'Ball'))

        # 排序：优先选择置信度最高的检测结果
        all_targets.sort(key=lambda x: x[1], reverse=True) 
        
        final_targets = []
        player_count = 0
        ball_count = 0
        
        # 限制总数 24 (23 球员/裁判 + 1 球)
        for bbox, conf, label in all_targets:
            if len(final_targets) >= 24:
                break
                
            if 'Player' in label:
                if player_count < 23:
                    final_targets.append((bbox, label))
                    player_count += 1
            elif label == 'Ball':
                if ball_count < 1:
                    final_targets.append((bbox, label))
                    ball_count += 1
        
        print(f"YOLO + 颜色检测到 {len(final_targets)} 个目标。")
        return [(bbox, label) for bbox, label in final_targets]

    # --- 绘制跟踪结果 ---
    def _draw_tracking_results(self, display_frame):
        """在帧上绘制所有跟踪目标的 ID 和边界框"""
        tracked_objects = self.object_tracker_manager.get_tracked_bboxes()
        
        for obj in tracked_objects:
            obj_id = obj['id']
            x, y, w, h = [int(v) for v in obj['bbox']]
            label = obj['label']

            # 绘制边界框
            color = (0, 0, 255) if 'Red' in label else (255, 0, 0) if 'Blue' in label else (0, 255, 255) # 红/蓝/黄
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # 绘制 ID 和标签
            text = f"ID {obj_id}: {label}"
            cv2.putText(display_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    # --- 鼠标回调 (仅用于手动修正) ---
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数，仅用于手动修正跟踪目标"""
        
        # ---------------- 1. 跟踪手动修正 ----------------
        if self.tracking_enabled and self.manual_reselect_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.reselect_bbox_start = (x, y)
                self.reselect_bbox_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
                self.reselect_bbox_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                x1, y1 = self.reselect_bbox_start
                x2, y2 = self.reselect_bbox_end
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                new_bbox = (min(x1, x2), min(y1, y2), w, h)
                
                if w > 10 and h > 10: 
                    target_id = self.target_id_to_fix
                    
                    if target_id in self.object_tracker_manager.trackers:
                        # 重新初始化跟踪器
                        tracker_data = self.object_tracker_manager.trackers[target_id]
                        tracker = cv2.TrackerCSRT_create() 
                        try:
                             tracker.init(self.frame, new_bbox)
                             # 更新管理器中的数据
                             self.object_tracker_manager.trackers[target_id]['tracker'] = tracker
                             self.object_tracker_manager.trackers[target_id]['bbox'] = new_bbox
                             print(f"成功修正 ID {target_id} 的跟踪框到 {new_bbox}")
                        except cv2.error:
                             print(f"修正 ID {target_id} 失败：边界框无效。")
                    else:
                        print(f"警告：未找到 ID {target_id} 的目标进行修正。")

                self.reselect_bbox_start = None
                self.reselect_bbox_end = None
                self.manual_reselect_mode = False 
                self.paused = True 

    # --- 主循环 ---
    def run(self):
        """主循环，处理视频播放和用户输入"""
        print("\n=== 控制台说明 ===")
        print("- 'p'/' ': 暂停/播放 (或单帧步进)")
        print("- 'r': 倒退5帧")
        print("- 't': 切换跟踪模式 (Tracker ON/OFF)")
        print("- 'c': 切换手动修正模式 (需暂停，并使用 Trackbar 选择 ID)")
        print("- 'q': 退出")
        
        while True:
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break
                self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.current_frame = self.frame.copy()
            
            display_frame = self.current_frame.copy()
            
            # 1. 跟踪逻辑
            if self.tracking_enabled:
                if not self.paused or self.frame_idx > self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1:
                    if not self.detection_triggered:
                        print("首次目标检测：正在初始化跟踪器...")
                        initial_targets = self._run_initial_detection(self.frame)
                        
                        if initial_targets:
                            for bbox, label in initial_targets:
                                self.object_tracker_manager.init_tracker(self.frame, bbox, label)
                            self.detection_triggered = True
                            print(f"已初始化 {len(initial_targets)} 个跟踪目标。")
                        else:
                            print("未检测到初始目标，请检查 YOLO 模型或颜色阈值。")
                            self.tracking_enabled = False 
                    
                    if self.detection_triggered:
                        self.object_tracker_manager.update_trackers(self.frame)
                
                # 绘制跟踪结果
                self._draw_tracking_results(display_frame)
            
            # 2. 绘制手动修正时的临时矩形
            if self.manual_reselect_mode and self.reselect_bbox_start and self.reselect_bbox_end:
                pt1 = self.reselect_bbox_start
                pt2 = self.reselect_bbox_end
                cv2.rectangle(display_frame, pt1, pt2, (255, 255, 0), 2)
                cv2.putText(display_frame, f"Re-selecting BBox for ID {self.target_id_to_fix}. Click to finalize.", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 3. 绘制状态信息
            h, w, _ = display_frame.shape
            cv2.putText(display_frame, f"Frame: {self.frame_idx}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display_frame, f"PAUSED: {'YES' if self.paused else 'NO'}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display_frame, f"TRACKER: {'ON' if self.tracking_enabled else 'OFF'} (T)", (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if self.tracking_enabled else (0, 0, 255), 2)


            cv2.imshow('Video Tracker', display_frame)
            
            key = cv2.waitKey(0 if self.paused else self.delay) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'): 
                self.paused = not self.paused
            elif key == ord(' '):
                if self.paused:
                    ret, self.frame = self.cap.read()
                    if not ret:
                        break
                    self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.current_frame = self.frame.copy()
            elif key == ord('r'):
                self.paused = True 
                N = 5
                target = max(0, self.frame_idx - N)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, self.frame = self.cap.read()
                if ret:
                    self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.current_frame = self.frame.copy()
                else:
                    self._set_initial_frame() 

            elif key == ord('t'): 
                self.tracking_enabled = not self.tracking_enabled
                if self.tracking_enabled:
                    self.object_tracker_manager.reset()
                    self.detection_triggered = False
                    print("跟踪模式开启。按空格或播放开始跟踪。")
                else:
                    self.object_tracker_manager.reset()
                    self.detection_triggered = False
                    print("跟踪模式关闭。")
                    
            elif key == ord('c'): # 切换手动修正模式
                if self.tracking_enabled:
                    self.manual_reselect_mode = not self.manual_reselect_mode
                    if self.manual_reselect_mode:
                        self.paused = True 
                        print(f"进入手动修正模式。请在 'Target ID to Fix' Trackbar 选择 ID {self.target_id_to_fix}，然后在视频窗口中拖动鼠标绘制新的边界框。")
                    else:
                        print("退出手动修正模式。")
                else:
                    print("请先按 't' 开启跟踪模式。")

        self.cap.release()
        cv2.destroyAllWindows()


# 示例使用
if __name__ == "__main__":
    # 请将此路径替换为您的视频文件路径
    video_file_path = "/Users/kai/Downloads/3d模型/output.mp4" 
    
    if not os.path.exists(video_file_path):
        print(f"警告: 视频文件未找到在 {video_file_path}")
        print("请将代码中的 video_file_path 变量修改为您的视频文件路径。")
    
    try:
        app = VideoTrackerApp(video_file_path)
        app.run()
    except SystemExit as e:
        print(e)
    except Exception as e:
        print(f"发生错误: {e}")