import cv2
import numpy as np

img = cv2.imread("uploads/prd.jpg")
print(img.dtype, img.shape)  # should be uint8 and 3 channels

# Convert if needed
if len(img.shape) == 2:  # grayscale
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
elif img.shape[2] == 4:  # RGBA
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
else:  # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
