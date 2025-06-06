# preprocessing/deskewing.py

import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import matplotlib.pyplot as plt

def determine_skew(gray_img):
    """
    Detect skew angle using projection profile analysis.
    Args:
        gray_img: Grayscale image as NumPy array
    Returns:
        float: Skew angle in degrees
    """
    # Ensure binary image with text as white
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    angles = np.arange(-15, 15, 0.5)
    scores = []

    for angle in angles:
        rotated = inter.rotate(binary, angle, reshape=False, order=0)
        hist = np.sum(rotated, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        scores.append(score)

    best_angle = angles[np.argmax(scores)]
    return best_angle


def deskew_image(image, angle=None, border_value=255):
    """
    Deskew the image based on detected or provided angle.
    Args:
        image: Input grayscale image (NumPy array)
        angle: Skew angle in degrees (auto-calculated if None)
        border_value: Background fill value (default: white)
    Returns:
        NumPy array: Deskewed image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if angle is None:
        angle = determine_skew(image)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    deskewed = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return deskewed


def show_comparison(original, deskewed, title="Deskewed Result"):
    """
    Display original and processed images side-by-side using matplotlib.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(deskewed, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Optional testing entry point
if __name__ == "__main__":
    test_img = cv2.imread("skewed_document.jpg", cv2.IMREAD_GRAYSCALE)
    if test_img is None:
        print("Error: Image not found.")
        exit()

    deskewed = deskew_image(test_img)
    show_comparison(test_img, deskewed)
    cv2.imwrite("deskewed_output.jpg", deskewed)
