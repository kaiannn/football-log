import cv2

path = "/Users/kai/Downloads/3d模型/ecc7c86352f76011ea88a7f7ff54c61f.mp4"
cap = cv2.VideoCapture(path)  # replace with your video path

if not cap.isOpened():
    raise SystemExit("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
delay = int(1000 / fps)

paused = False
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    cv2.imshow('Video', frame)
    key = cv2.waitKey(0 if paused else delay) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('p'):          # toggle pause/play
        paused = not paused
    elif key == ord(' '):          # space = step forward one frame when paused
        if paused:
            ret, frame = cap.read()
            if not ret:
                break
    elif key == ord('r'):          # rewind a few frames (seek backward)
        # subtract N frames; clamp to 0
        N = 5
        target = max(0, frame_idx - N)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        frame_idx = target
        ret, frame = cap.read()
        paused = True

cap.release()
cv2.destroyAllWindows()
