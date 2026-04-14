import cv2
#mp4播放
path = "path/to/match.mp4"
cap = cv2.VideoCapture(path)

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
