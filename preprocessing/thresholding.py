# preprocessing/thresholding.py

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from matplotlib import pyplot as plt

def apply_global_threshold(gray_img, show_result=False):
    """
    Apply global Otsu thresholding to a grayscale image.

    Parameters:
    - gray_img: Grayscale image (numpy array)
    - show_result: bool, whether to plot results

    Returns:
    - binary_img: Binary thresholded image
    """
    if gray_img is None:
        raise ValueError("Input image is None in apply_global_threshold()")

    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if show_result:
        _display_results(gray_img, binary_img, "Global (Otsu) Thresholding")

    return binary_img


def apply_adaptive_threshold(gray_img, block_size=11, C=2, show_result=False):
    """
    Apply adaptive Gaussian thresholding to a grayscale image.

    Parameters:
    - gray_img: Grayscale image (numpy array)
    - block_size: Size of pixel neighborhood (must be odd)
    - C: Constant subtracted from mean or weighted mean
    - show_result: bool, whether to plot results

    Returns:
    - binary_img: Binary thresholded image
    """
    if gray_img is None:
        raise ValueError("Input image is None in apply_adaptive_threshold()")

    binary_img = cv2.adaptiveThreshold(
        gray_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        C
    )

    if show_result:
        _display_results(gray_img, binary_img, "Adaptive Thresholding")

    return binary_img


def apply_sauvola_threshold(gray_img, window_size=25, k=0.2, show_result=False):
    """
    Apply Sauvola thresholding to a grayscale image.

    Parameters:
    - gray_img: Grayscale image (numpy array)
    - window_size: Size of the window to calculate threshold
    - k: Parameter to tune threshold bias
    - show_result: bool, whether to plot results

    Returns:
    - binary_img: Binary thresholded image
    """
    if gray_img is None:
        raise ValueError("Input image is None in apply_sauvola_threshold()")

    thresh = threshold_sauvola(gray_img, window_size=window_size, k=k)
    binary_img = (gray_img > thresh).astype(np.uint8) * 255

    if show_result:
        _display_results(gray_img, binary_img, "Sauvola's Thresholding")

    return binary_img


def _display_results(original, processed, title):
    """
    Display original and processed images side by side.

    Parameters:
    - original: Original grayscale image
    - processed: Binary processed image
    - title: Title for the processed image subplot
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title(title)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# Optional test run
if __name__ == "__main__":
    import sys
    import cv2

    input_path = sys.argv[1] if len(sys.argv) > 1 else "your_image.jpg"
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Error: Unable to load image {input_path}")
        sys.exit(1)

    # Test all thresholding methods with visualization
    binary_global = apply_global_threshold(gray, show_result=True)
    binary_adaptive = apply_adaptive_threshold(gray, show_result=True)
    binary_sauvola = apply_sauvola_threshold(gray, show_result=True)

    # Save the Sauvola result by default
    cv2.imwrite("binary_output_sauvola.jpg", binary_sauvola)


def apply_threshold(image, method="otsu"):
    """
    Wrapper to apply selected thresholding method.

    Parameters:
    - image: Grayscale image
    - method: 'otsu', 'adaptive', or 'sauvola'

    Returns:
    - binary image
    """
    if method == "otsu":
        return apply_global_threshold(image)
    elif method == "adaptive":
        return apply_adaptive_threshold(image)
    elif method == "sauvola":
        return apply_sauvola_threshold(image)
    else:
        raise ValueError(f"Unsupported thresholding method: {method}")
