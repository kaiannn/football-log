import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

img = np.zeros((300, 300, 3), dtype='uint8')
cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)
cv2.imshow("Test Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
