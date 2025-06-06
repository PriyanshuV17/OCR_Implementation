# preprocessing/noise_reduction.py

import cv2
import numpy as np

def reduce_noise(gray_img):
    """
    Applies noise removal techniques to a grayscale image.

    Parameters:
    - gray_img: Grayscale input image (numpy array).

    Returns:
    - denoised_img: Image after noise removal and binarization (numpy array).
    """

    if gray_img is None:
        raise ValueError("Input image is None in reduce_noise()")

    # Gaussian Blur (reduces high-frequency noise)
    gaussian_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Median Blur (removes salt-and-pepper noise)
    median_blur = cv2.medianBlur(gaussian_blur, 3)

    # Adaptive Thresholding (binarizes text/background)
    denoised_img = cv2.adaptiveThreshold(
        median_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # or cv2.THRESH_BINARY depending on text/background colors
        11,
        2
    )

    return denoised_img


# Optional test block
if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "input_grayscale.png"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "denoised_output.png"

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image from {input_path}")
        sys.exit(1)

    denoised = reduce_noise(img)
    cv2.imwrite(output_path, denoised)
    print(f"Denoised image saved to: {output_path}")

    cv2.imshow("Original Grayscale", img)
    cv2.imshow("Denoised Image", denoised)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
