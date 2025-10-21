import cv2
import numpy as np

class QuadrilateralMarker:  #四边形
    def __init__(self):
        self.points = []  # 存储四个点的坐标
        self.current_frame = None
        self.drawing = False #标记是否绘制四边形
        self.original_frame = None
        self.current_frame = None

    def set_frame(self, frame):
        """设置当前帧，读取新帧时调用"""
        self.original_frame = frame.copy()
        self.current_frame = frame.copy()
        self.points = [] #重制点
        self.drawing = False #重制绘制状态

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN: #左键选择点
            if len(self.points) < 4:
                self.points.append((x, y))
                print(f"点 {len(self.points)}: ({x}, {y})")
                
                # 重新加载原始图像清除之前的绘制
                self.current_frame = self.original_frame.copy()
                
                # 重新绘制所有已有点
                for point in self.points:
                    cv2.circle(self.current_frame, point, 5, (0, 255, 0), -1)
                
                # 如果已经点了4个点，自动连接成四边形
                if len(self.points) == 4:
                    self.draw_quadrilateral()
                    self.drawing = True
                    
        elif event == cv2.EVENT_RBUTTONDOWN: #右键清除点
            # 右键清除所有点重新开始
            self.points = []
            self.drawing = False
            # 重新加载原始图像清除所有绘制
            self.current_frame = self.original_frame.copy()
            print("已清除所有点，请重新点击")

    def draw_quadrilateral(self):
        """连接四个点形成四边形"""
        if len(self.points) == 4:
            # 对点进行排序，确保正确的连接顺序
            sorted_points = self.sort_points(self.points)
            
            # 将点转换为numpy数组
            pts = np.array(sorted_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # 绘制四边形
            cv2.polylines(self.current_frame, [pts], True, (0, 0, 255), 2)
            
            # 填充四边形（半透明）
            overlay = self.current_frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, self.current_frame, 0.7, 0, self.current_frame)
            
            print("四边形已绘制完成！")
            print("坐标点:", sorted_points)

    def sort_points(self, points):
        """对四个点进行排序：左上→右上→右下→左下"""
        # 将点转换为numpy数组便于处理
        points = np.array(points)
        
        # 计算中心点
        center = points.mean(axis=0)
        
        # 计算每个点相对于中心点的角度
        angles = []
        for point in points:
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            angle = np.arctan2(dy, dx) #返回弧度制
            angles.append(angle)
        
        # 按角度排序
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]
        
        return sorted_points.tolist()    
        # 还可以两次排序，先y（上下）后x（左右）；连接前两个点和后两个点

    def mark_video_frame(self, video_path):
        """在视频的某一帧上标记四边形"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("无法打开视频文件")
            return
        
        # 读取第一帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取视频帧")
            return
        
        # 初始化帧
        self.set_frame(frame)  # 使用 set_frame 而不是直接赋值
        
        # 创建窗口并设置鼠标回调
        cv2.namedWindow('Mark Quadrilateral')
        cv2.setMouseCallback('Mark Quadrilateral', self.mouse_callback)
        
        print("使用说明:")
        print("- 左键点击: 添加点 (共需要4个点)")
        print("- 右键点击: 清除所有点重新开始")
        print("- 按 's': 保存四边形坐标")
        print("- 按 'q': 退出")
        print("- 按 'n': 跳到下一帧")
        
        while True:
            # 显示当前帧（包含标记）
            display_frame = self.current_frame.copy()
            
            # 显示操作提示
            cv2.putText(display_frame, "Left click: add point, Right click: clear", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 's' to save, 'q' to quit, 'n' for next frame", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Points: {len(self.points)}/4", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Mark Quadrilateral', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and self.drawing:
                self.save_coordinates()
            elif key == ord('n'):
                # 跳到下一帧并重置标记
                ret, frame = cap.read()
                if ret:
                    self.set_frame(frame)  # 重要：使用 set_frame 重置状态
                else:
                    print("已到达视频末尾")
                    break
        
        cap.release()
        cv2.destroyAllWindows()   

    def mark_image(self, image_path):
        """在图片上标记四边形"""
        image = cv2.imread(image_path)
        if image is None:
            print("无法加载图片")
            return
        
        self.current_frame = image.copy()
        
        cv2.namedWindow('Mark Quadrilateral')
        cv2.setMouseCallback('Mark Quadrilateral', self.mouse_callback)
        
        print("使用说明:")
        print("- 左键点击: 添加点 (共需要4个点)")
        print("- 右键点击: 清除所有点重新开始")
        print("- 按 's': 保存四边形坐标")
        print("- 按 'q': 退出")
        
        while True:
            display_frame = self.current_frame.copy()
            
            # 显示操作提示
            cv2.putText(display_frame, "Left click: add point, Right click: clear", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 's' to save, 'q' to quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Points: {len(self.points)}/4", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Mark Quadrilateral', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and self.drawing:
                self.save_coordinates()
                break
        
        cv2.destroyAllWindows()
        return self.points if self.drawing else None
    
    def save_coordinates(self):
        """保存坐标到文件"""
        if len(self.points) == 4:
            with open('quadrilateral_coordinates.txt', 'w') as f:
                for i, point in enumerate(self.points):
                    f.write(f"Point {i+1}: {point}\n")
            print("坐标已保存到 quadrilateral_coordinates.txt")
            
            # 也可以保存为numpy格式
            np.save('quadrilateral_coordinates.npy', np.array(self.points))
            print("坐标已保存到 quadrilateral_coordinates.npy")

# 使用示例
if __name__ == "__main__":
    marker = QuadrilateralMarker()
    
    # 方式1: 在视频帧上标记
    path = "/Users/kai/Downloads/3d模型/ecc7c86352f76011ea88a7f7ff54c61f.mp4"
    marker.mark_video_frame(path)
    
    # 方式2: 在图片上标记
    points = marker.mark_image('your_image.jpg')
    
    if points:
        print("最终四边形坐标:")
        for i, point in enumerate(points):
            print(f"点 {i+1}: {point}")