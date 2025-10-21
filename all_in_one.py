import cv2
import numpy as np
import os # 导入 os 库用于检查文件是否存在

# =================================================================
# 1. ObjectTrackerManager 类 (只负责跟踪器管理)
# =================================================================
class ObjectTrackerManager:
    def __init__(self):
        # 存储当前正在跟踪的所有目标：{ID: (tracker, bbox, label)}
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
        tracked_objects_info = [] # 存储 ID, 坐标, 标签

        for obj_id, data in self.trackers.items():
            tracker = data['tracker']
            label = data['label']
            
            success, bbox = tracker.update(frame)

            if success:
                # 跟踪成功
                updated_trackers[obj_id] = {'tracker': tracker, 'bbox': bbox, 'label': label}
                
                # Bounding box 转换为整数
                x, y, w, h = [int(v) for v in bbox]
                tracked_objects_info.append({
                    'id': obj_id, 
                    'bbox': (x, y, w, h), 
                    'label': label
                })
            else:
                # 跟踪失败，可以认为目标丢失
                # print(f"ID {obj_id} ({label}) 丢失跟踪。") # 频繁打印可能影响性能
                pass

        self.trackers = updated_trackers
        return tracked_objects_info

    def get_tracked_bboxes(self):
        """返回所有跟踪目标的边界框和标签"""
        # 返回当前跟踪列表的深拷贝信息
        return [{'id': k, 'bbox': v['bbox'], 'label': v['label']} for k, v in self.trackers.items()]

    def reset(self):
        """清除所有跟踪器"""
        self.trackers = {}
        self.next_id = 0


# =================================================================
# 2. VideoMarkerAndDetector 类 (主程序逻辑和视图)
# =================================================================
class VideoMarkerAndDetector:
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

        # --- 手动/自动标记属性 ---
        self.points = []
        self.original_frame = None
        self.current_frame = None
        self.drawing = False
        self.mode = 'manual'
        self.show_birds_eye = False
        self.M = None
        
        # --- 球场检测 (绿色) 属性 ---
        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([85, 255, 255])
        self.avg_detected_points = None

        # --- 目标跟踪模块（正确初始化） ---
        self.object_tracker_manager = ObjectTrackerManager()
        self.detection_triggered = False 
        self.tracking_enabled = False 
        
        # --- 新增：手动修正模式 ---
        self.manual_reselect_mode = False # 标记是否处于手动重选状态
        self.reselect_bbox_start = None   # 鼠标开始点
        self.reselect_bbox_end = None     # 鼠标结束点
        
        # --- 颜色检测阈值 (球员/球，请根据视频调整) ---
        self.lower_red1 = np.array([0, 70, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 70, 50])
        self.upper_red2 = np.array([180, 255, 255])
        self.lower_blue = np.array([100, 70, 50])
        self.upper_blue = np.array([130, 255, 255])
        self.lower_ball = np.array([20, 100, 100]) # 亮黄色
        self.upper_ball = np.array([30, 255, 255])
        
        #创建窗口并设置鼠标回调
        cv2.namedWindow('Video Marker & Detector')
        cv2.setMouseCallback('Video Marker & Detector', self.mouse_callback)

        #创建颜色调整窗口
        cv2.namedWindow('Color Adjust',cv2.WINDOW_NORMAL)
        self._create_trackbars()
        cv2.setMouseCallback('Video Maker & Detector', self.mouse_callback)
        self._set_initial_frame()
        
    def _set_initial_frame(self):
        ret, self.frame = self.cap.read()
        if not ret:
            raise SystemExit("无法读取视频帧")
        
        self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.set_frame(self.frame)

    # --- 颜色检测和边界框提取 ---
    def _color_detect_initial_targets(self, frame):
            """
            在某一帧进行颜色分割和轮廓检测，找到球员和球的初始边界框，并应用 24 个目标的限制。
            返回：[(bbox, label), ...]
            """
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 1. 实时获取颜色阈值 (确保使用用户调整后的最新阈值)
            # 在 _get_hsv_thresholds 中已更新 self.lower_blue/upper_blue
            
            # 2. 球场 Mask (排除球场背景)
            green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            
            # 3. 颜色分割 (获取所有潜在目标)
            red_mask = cv2.inRange(hsv, self.lower_red1, self.upper_red1) | cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            red_mask = cv2.subtract(red_mask, green_mask) 
            
            blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
            blue_mask = cv2.subtract(blue_mask, green_mask)
            
            ball_mask = cv2.inRange(hsv, self.lower_ball, self.upper_ball)
            
            # --- 轮廓检测和边界框提取 ---
            all_player_candidates = []
            all_ball_candidates = []
            
            # 球员轮廓检测 (形态学处理)
            kernel = np.ones((5, 5), np.uint8)
            
            # Red Players (最小面积 500)
            red_processed = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            red_processed = cv2.morphologyEx(red_processed, cv2.MORPH_CLOSE, kernel, iterations=3)
            # _find_and_append_bboxes 返回 [(bbox, area)] 列表
            red_cands = self._find_and_append_bboxes(red_processed, 'Red Player', [], min_area=500, max_area=np.inf)
            all_player_candidates.extend([(bbox, area, 'Red Player') for bbox, area in red_cands])

            # Blue Players (最小面积 500)
            blue_processed = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            blue_processed = cv2.morphologyEx(blue_processed, cv2.MORPH_CLOSE, kernel, iterations=3)
            blue_cands = self._find_and_append_bboxes(blue_processed, 'Blue Player', [], min_area=500, max_area=np.inf)
            all_player_candidates.extend([(bbox, area, 'Blue Player') for bbox, area in blue_cands])

            # Ball (最小面积 50, 最大面积 500)
            ball_processed = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            ball_cands = self._find_and_append_bboxes(ball_processed, 'Ball', [], min_area=50, max_area=500)
            all_ball_candidates.extend([(bbox, area, 'Ball') for bbox, area in ball_cands])


            # --- 4. 限制目标数量（核心逻辑） ---
            
            final_targets = []
            
            # 4a. 球员/裁判限制：选择面积最大的 23 个目标 (22 球员 + 1 裁判)
            # 按面积降序排序
            all_player_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 选取最大的 23 个目标
            selected_players = all_player_candidates[:23]
            
            for bbox, area, label in selected_players:
                final_targets.append((bbox, label))
                
            # 4b. 球限制：选择面积最大的 1 个球 (如果有的话)
            if all_ball_candidates:
                # 球的面积通常很小，但可能是多个噪点，选最大的一个
                all_ball_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # 检查这个最大的球边界框是否与任何选定的球员边界框重叠
                # (避免将球员脚下的亮光误识别为球)
                is_new_ball = True
                ball_bbox, _, _ = all_ball_candidates[0]
                bx, by, bw, bh = ball_bbox
                
                # 简单重叠检查 (Jaccard 相似度或简单中心点距离都可以)
                ball_center = (bx + bw/2, by + bh/2)
                
                # 如果球的中心点不在任何球员边界框内，则认为它是合法的球
                for player_bbox, _, _ in selected_players:
                    px, py, pw, ph = player_bbox
                    if px < ball_center[0] < px + pw and py < ball_center[1] < py + ph:
                        is_new_ball = False
                        break

                if is_new_ball:
                    final_targets.append((ball_bbox, 'Ball'))

            print(f"检测到 {len(final_targets)} 个目标 (限制最大 24 个)。")
            return final_targets
        
    def _find_and_append_bboxes(self, mask, label, results_list, min_area, max_area=np.inf):
        """
        辅助函数：查找轮廓并返回符合面积要求的边界框列表。
        不再直接修改 results_list，而是返回 [(bbox, area)] 列表。
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_bboxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area and area < max_area:
                x, y, w, h = cv2.boundingRect(c)
                found_bboxes.append(((x, y, w, h), area)) # 返回 bbox 和面积
                
        return found_bboxes

    # --- 绘制跟踪结果 ---
    def _draw_tracking_results(self, display_frame):
        """在帧上绘制所有跟踪目标的 ID 和边界框"""
        tracked_objects = self.object_tracker_manager.get_tracked_bboxes()
        
        for obj in tracked_objects:
            obj_id = obj['id']
            x, y, w, h = obj['bbox']
            label = obj['label']

            # 绘制边界框
            color = (0, 0, 255) if 'Red' in label else (255, 0, 0) if 'Blue' in label else (0, 255, 255) # 红/蓝/黄
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # 绘制 ID 和标签
            text = f"ID {obj_id}: {label}"
            cv2.putText(display_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 打印到控制台 (输出：每帧的球员/球的唯一 ID + 坐标)
            # print(f"Frame {self.frame_idx} | ID {obj_id} ({label}): Pixel Pos (x={x}, y={y})")

    # --- 核心新增功能：透视变换 ---
    def get_birds_eye_view(self, frame, src_points):
        """
        执行透视变换，将四边形区域转换为矩形俯视图。
        """
        pts1 = np.float32(src_points)
        widthA = np.sqrt(((pts1[2][0] - pts1[3][0]) ** 2) + ((pts1[2][1] - pts1[3][1]) ** 2))
        widthB = np.sqrt(((pts1[1][0] - pts1[0][0]) ** 2) + ((pts1[1][1] - pts1[0][1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((pts1[1][0] - pts1[2][0]) ** 2) + ((pts1[1][1] - pts1[2][1]) ** 2))
        heightB = np.sqrt(((pts1[0][0] - pts1[3][0]) ** 2) + ((pts1[0][1] - pts1[3][1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        pts2 = np.float32([
            [0, 0],                         
            [maxWidth - 1, 0],              
            [maxWidth - 1, maxHeight - 1],  
            [0, maxHeight - 1]              
        ])
        self.M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(frame, self.M, (maxWidth, maxHeight))
        return warped
    
    def _update_perspective_view(self):
        """根据当前模式获取点并尝试计算俯视图"""
        src_points = None
        if self.mode == 'manual' and self.drawing:
            src_points = self.sort_points(self.points)
        elif self.mode == 'auto' and self.avg_detected_points is not None:
            src_points = self.avg_detected_points
        
        if src_points and len(src_points) == 4:
            birds_eye_frame = self.get_birds_eye_view(self.frame, src_points)
            cv2.imshow('Birds-Eye View (Perspective Transform)', birds_eye_frame)
        else:
            cv2.destroyWindow('Birds-Eye View (Perspective Transform)')
            self.show_birds_eye = False

    # --- 视频播放控制和主循环 ---
    def run(self):
        """主循环，处理视频播放和用户输入"""
        print("\n=== 控制台说明 ===")
        print("- 'p'/' ': 暂停/播放 (或单帧步进)")
        print("- 'r': 倒退5帧")
        print("- 'm': 切换模式 (manual/auto)")
        print("- 'b': 切换俯视图显示 (Birds-Eye View)")
        print("- 't': 切换跟踪模式 (Tracker ON/OFF)")
        print("- 'c': 切换手动修正模式 (需暂停)") # 新增按键
        print("- 'q': 退出")
        
        while True:
            # 1. 实时获取颜色阈值
            self._get_hsv_thresholds()

            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break
                self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.set_frame(self.frame)
            
            # 2. 绘制：根据当前模式进行绘制
            display_frame = self.current_frame.copy()
            if self.mode == 'manual':
                self._update_manual_drawing()
                self._draw_manual_tips(display_frame)
            elif self.mode == 'auto':
                self._draw_auto_detection(display_frame)
                self._draw_auto_tips(display_frame)
            
            # 3. 跟踪逻辑
            if self.tracking_enabled:
                # 仅在播放或单帧步进（即帧发生变化）时进行跟踪更新
                if not self.paused or self.frame_idx > self.cap.get(cv2.CAP_PROP_POS_FRAMES) - 1:
                    if not self.detection_triggered:
                        print("首次目标检测：正在初始化跟踪器...")
                        initial_targets = self._color_detect_initial_targets(self.frame)
                        
                        if initial_targets:
                            for bbox, label in initial_targets:
                                self.object_tracker_manager.init_tracker(self.frame, bbox, label)
                            self.detection_triggered = True
                            print(f"已初始化 {len(initial_targets)} 个跟踪目标。")
                        else:
                            print("未检测到初始目标，请调整颜色阈值或尝试其他帧。")
                            self.tracking_enabled = False # 如果没检测到，自动关闭跟踪模式
                    
                    if self.detection_triggered:
                        self.object_tracker_manager.update_trackers(self.frame)
                
                # 绘制跟踪结果（无论是否暂停）
                self._draw_tracking_results(display_frame)
            
            # 4. 新增：绘制手动修正时的临时矩形
            if self.manual_reselect_mode and self.reselect_bbox_start and self.reselect_bbox_end:
                pt1 = self.reselect_bbox_start
                pt2 = self.reselect_bbox_end
                cv2.rectangle(display_frame, pt1, pt2, (255, 255, 0), 2)
                cv2.putText(display_frame, "Drag to re-select BBox for ID (Check 'Color Adjust' window)", 
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Video Marker & Detector', display_frame)
            
            # 3. 俯视图显示
            if self.show_birds_eye:
                self._update_perspective_view()
            else:
                cv2.destroyWindow('Birds-Eye View (Perspective Transform)')

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
                    self.set_frame(self.frame)
            elif key == ord('r'):
                self.paused = True 
                N = 5
                target = max(0, self.frame_idx - N)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ret, self.frame = self.cap.read()
                if ret:
                    self.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.set_frame(self.frame)
                else:
                    self._set_initial_frame() 

            elif key == ord('m'): 
                self.mode = 'auto' if self.mode == 'manual' else 'manual'
                print(f"模式已切换到: {self.mode}")
                if self.mode == 'auto':
                    print("正在计算稳定的自动检测点...")
                    self.avg_detected_points = self._calculate_stable_detection()
                    print("稳定的检测点已计算完成。")
                self.set_frame(self.frame)
                
            elif key == ord('b'): 
                self.show_birds_eye = not self.show_birds_eye
                print(f"俯视图显示: {'开启' if self.show_birds_eye else '关闭'}")
                if not self.show_birds_eye:
                    cv2.destroyWindow('Birds-Eye View (Perspective Transform)')
                    
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
                        self.paused = True # 必须暂停才能精确修正
                        print("进入手动修正模式。请在 'Color Adjust' 窗口设置目标 ID，然后在视频窗口中拖动鼠标绘制新的边界框。")
                    else:
                        print("退出手动修正模式。")
                else:
                    print("请先按 't' 开启跟踪模式。")


        self.cap.release()
        cv2.destroyAllWindows()
    
    # --- 帧和状态管理 ---
    def set_frame(self, frame):
        """设置当前帧，读取新帧或重置时调用"""
        self.original_frame = frame.copy()
        self.current_frame = frame.copy()
        # 仅在模式切换或帧跳转时才清空手动点，否则保持当前点不变
        if self.mode == 'manual' and len(self.points) < 4:
            self.points = [] 
            self.drawing = False 

    # --- 手动标记相关方法（部分省略以保持简洁，使用原先的逻辑） ---
    def mouse_callback(self, event, x, y, flags, param):
        ''''鼠标回调函数，集成手动标记和手动修正'''
        #------------1. 手动标记球场-------------------------
        if self.mode != 'manual': return
        if event == cv2.EVENT_LBUTTONDOWN: 
            if len(self.points) < 4:
                self.points.append((x, y))
                if len(self.points) == 4:
                    self.drawing = True
        elif event == cv2.EVENT_RBUTTONDOWN: 
            self.points = []
            self.drawing = False
            self.current_frame = self.original_frame.copy()
    # -------------2. 跟踪手动修正 -------------------------
        if self.tracking_enabled and self.manual_reselect_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.reselect_bbox_start = (x, y)
                self.reselect_bbox_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
                self.reselect_bbox_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                # 鼠标抬起，完成绘制，进行修正
                x1, y1 = self.reselect_bbox_start
                x2, y2 = self.reselect_bbox_end
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                
                # Bounding Box: (x, y, w, h)
                new_bbox = (min(x1, x2), min(y1, y2), w, h)
                
                if w > 10 and h > 10: # 确保绘制了一个有效的矩形
                    target_id = cv2.getTrackbarPos('Target_ID', 'Color Adjust')
                    
                    # 查找要修正的跟踪器
                    if target_id in self.object_tracker_manager.trackers:
                        # 重新初始化跟踪器
                        tracker_data = self.object_tracker_manager.trackers[target_id]
                        tracker = cv2.TrackerCSRT_create() # 重新创建 CSRT 实例
                        tracker.init(self.frame, new_bbox)
                        
                        # 更新管理器中的数据
                        self.object_tracker_manager.trackers[target_id]['tracker'] = tracker
                        self.object_tracker_manager.trackers[target_id]['bbox'] = new_bbox
                        
                        print(f"成功修正 ID {target_id} 的跟踪框到 {new_bbox}")
                    else:
                        print(f"警告：未找到 ID {target_id} 的目标进行修正。")

                self.reselect_bbox_start = None
                self.reselect_bbox_end = None
                self.manual_reselect_mode = False # 修正完成后退出模式
                self.paused = True # 保持暂停状态

    def _update_manual_drawing(self):
        self.current_frame = self.original_frame.copy()
        for point in self.points:
            cv2.circle(self.current_frame, point, 5, (0, 255, 0), -1)
        if self.drawing:
            self._draw_quadrilateral(self.points)

    def _draw_quadrilateral(self, points_list):
        if len(points_list) == 4:
            sorted_points = self.sort_points(points_list)
            pts = np.array(sorted_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(self.current_frame, [pts], True, (0, 0, 255), 2)
            overlay = self.current_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, self.current_frame, 0.7, 0, self.current_frame)

    def sort_points(self, points):
        points = np.array(points)
        center = points.mean(axis=0)
        angles = [np.arctan2(p[1] - center[1], p[0] - center[0]) for p in points]
        sorted_indices = np.argsort(angles)
        return points[sorted_indices].tolist()    

    def save_coordinates(self):
        if len(self.points) == 4:
            sorted_points = self.sort_points(self.points)
            with open('quadrilateral_coordinates.txt', 'w') as f:
                for i, point in enumerate(sorted_points):
                    f.write(f"Point {i+1}: {point}\n")
            np.save('quadrilateral_coordinates.npy', np.array(sorted_points))

    # --- 自动检测相关方法 ---
    def _process_frame_for_green(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 10000:
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                return box.astype(int)
        return None

    def _draw_auto_detection(self, display_frame):
        if self.avg_detected_points is not None:
            pts = np.array(self.avg_detected_points, np.int32)
            cv2.drawContours(display_frame, [pts], 0, (0, 0, 255), 3)
            overlay = display_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)

    def _calculate_stable_detection(self, num_frames=10):
        collected_points = []
        current_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if not ret: break
            box = self._process_frame_for_green(frame)
            if box is not None: collected_points.append(box)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        
        if len(collected_points) > 0:
            return np.mean(collected_points, axis=0).astype(int).tolist()
        return None

    # --- 提示信息绘制 ---
    def _draw_manual_tips(self, display_frame):
        cv2.putText(display_frame, f"MODE: MANUAL (M)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "L/R Click: Point/Clear", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Points: {len(self.points)}/4. Press 's' to save", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self._draw_status_info(display_frame)

    def _draw_auto_tips(self, display_frame):
        cv2.putText(display_frame, f"MODE: AUTO (M)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display_frame, "Auto Green Detection", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Detection {'READY' if self.avg_detected_points else 'NOT READY'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        self._draw_status_info(display_frame)

    def _draw_status_info(self, display_frame):
        # 统一绘制底部状态信息
        h, w, _ = display_frame.shape
        cv2.putText(display_frame, f"Frame: {self.frame_idx}", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, f"PAUSED: {'YES' if self.paused else 'NO'}", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, f"TRACKER: {'ON' if self.tracking_enabled else 'OFF'} (T)", (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if self.tracking_enabled else (0, 0, 255), 2)
        cv2.putText(display_frame, f"BIRDS-EYE: {'ON' if self.show_birds_eye else 'OFF'} (B)", (w - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255) if self.show_birds_eye else (255, 255, 255), 2)

    def _create_trackbars(self):
        """创建用于实时调整颜色的Trackbars"""
        
        # 简化演示：只创建一组蓝色队服的 HSV 调整条
        # 实际应用中，您可能需要为红队和球创建多组或切换机制
        def on_trackbar(val):
            pass # 回调函数为空，值直接从 trackbar 获取

        cv2.createTrackbar('H_Min_Blue', 'Color Adjust', self.lower_blue[0], 179, on_trackbar)
        cv2.createTrackbar('S_Min_Blue', 'Color Adjust', self.lower_blue[1], 255, on_trackbar)
        cv2.createTrackbar('V_Min_Blue', 'Color Adjust', self.lower_blue[2], 255, on_trackbar)
        cv2.createTrackbar('H_Max_Blue', 'Color Adjust', self.upper_blue[0], 179, on_trackbar)
        cv2.createTrackbar('S_Max_Blue', 'Color Adjust', self.upper_blue[1], 255, on_trackbar)
        cv2.createTrackbar('V_Max_Blue', 'Color Adjust', self.upper_blue[2], 255, on_trackbar)
        
        # 为了演示手动修正，添加一个用于选择目标的 Trackbar
        cv2.createTrackbar('Target_ID', 'Color Adjust', 0, 100, on_trackbar)
        print("\n**提示：开启跟踪模式后，使用 'Color Adjust' 窗口的 Trackbar 调整颜色阈值。**")

    def _get_hsv_thresholds(self):
        """从Trackbars读取当前的HSV阈值"""
        h_min = cv2.getTrackbarPos('H_Min_Blue', 'Color Adjust')
        s_min = cv2.getTrackbarPos('S_Min_Blue', 'Color Adjust')
        v_min = cv2.getTrackbarPos('V_Min_Blue', 'Color Adjust')
        h_max = cv2.getTrackbarPos('H_Max_Blue', 'Color Adjust')
        s_max = cv2.getTrackbarPos('S_Max_Blue', 'Color Adjust')
        v_max = cv2.getTrackbarPos('V_Max_Blue', 'Color Adjust')
        
        # 实时更新蓝色阈值
        self.lower_blue = np.array([h_min, s_min, v_min])
        self.upper_blue = np.array([h_max, s_max, v_max])

# 示例使用
if __name__ == "__main__":
    # 请将此路径替换为您的视频文件路径
    video_file_path = "/Users/kai/Downloads/3d模型/output.mp4" 
    
    if not os.path.exists(video_file_path):
        print(f"警告: 视频文件未找到在 {video_file_path}")
        print("请将代码中的 video_file_path 变量修改为您的视频文件路径。")
    
    try:
        marker_detector = VideoMarkerAndDetector(video_file_path)
        marker_detector.run()
    except SystemExit as e:
        print(e)
    except Exception as e:
        print(f"发生错误: {e}")