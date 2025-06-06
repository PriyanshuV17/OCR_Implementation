import cv2
import numpy as np

# Read the input color image
color_img = cv2.imread('input.jpg')
if color_img is None:
    print("Error: Could not read the image file")
    exit()

# Method 1: Average Method
average_gray = np.mean(color_img, axis=2).astype(np.uint8)

# Method 2: Luminosity Method (Weighted Average)
# OpenCV uses BGR order by default
weights = [0.114, 0.587, 0.299]  # BGR weights corresponding to the luminosity formula
luminosity_gray = np.dot(color_img[..., :3], weights).astype(np.uint8)

# Method 3: Desaturation Method
max_channel = np.max(color_img, axis=2)
min_channel = np.min(color_img, axis=2)
desaturation_gray = ((max_channel + min_channel) / 2).astype(np.uint8)

# Method 4: OpenCV's built-in conversion (for reference)
opencv_gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

# Save all images
cv2.imwrite('gray_average.jpg', average_gray)
cv2.imwrite('gray_luminosity.jpg', luminosity_gray)
cv2.imwrite('gray_desaturation.jpg', desaturation_gray)
cv2.imwrite('gray_opencv.jpg', opencv_gray)

# Display all images (optional)
cv2.imshow('Original', color_img)
cv2.imshow('Average Method', average_gray)
cv2.imshow('Luminosity Method', luminosity_gray)
cv2.imshow('Desaturation Method', desaturation_gray)
cv2.imshow('OpenCV Built-in', opencv_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Grayscale images saved successfully:")
print("- gray_average.jpg (Average method)")
print("- gray_luminosity.jpg (Luminosity method)")
print("- gray_desaturation.jpg (Desaturation method)")
print("- gray_opencv.jpg (OpenCV's built-in method)")