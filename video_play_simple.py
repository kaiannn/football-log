import cv2
#mp4播放
path = "/Users/kai/Downloads/3d模型/ecc7c86352f76011ea88a7f7ff54c61f.mp4"
cap = cv2.VideoCapture(path)  # replace with your video path

if not cap.isOpened():
    print("Error: cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break

    cv2.imshow('Video', frame)

    # 30 ms delay per frame (about 33 fps)
    key = cv2.waitKey(300) & 0xFF
    if key == ord('q'):   # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
