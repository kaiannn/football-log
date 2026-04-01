import cv2
import numpy as np

video_path = "/Users/kai/Downloads/3d模型/output.mp4"
cap = cv2.VideoCapture(video_path)

# HSV 范围 - 可调（草地偏亮就调高 V）
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

# 收集几帧的四个点，最后平均
collected_points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 去噪
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # 找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 10000:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = box.astype(int)

            collected_points.append(box)
            cv2.drawContours(frame, [box], 0, (0, 0, 255), 3)

    # 显示（可选）
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
        break

cap.release()
cv2.destroyAllWindows()

# 平均四个点（取前10帧）
if len(collected_points) > 0:
    avg_points = np.mean(collected_points[:10], axis=0).astype(int)
    print("平均四个点（稳定输出）:")
    print(avg_points.tolist())
