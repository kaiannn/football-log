import cv2
import numpy as np  # 👈 add this line

print("OpenCV version:", cv2.__version__)

# Create a simple black image
img = np.zeros((300, 300, 3), dtype='uint8')

# Draw a red circle
cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)

# Show the image
cv2.imshow("Test Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
