# preprocessing/grayscale.py

import cv2
import numpy as np

def grayscale_opencv(image):
    """
    Convert a BGR color image to grayscale using OpenCV's built-in method.

    Parameters:
    - image: Input image in BGR format.

    Returns:
    - Grayscale image (uint8 numpy array).
    """
    if image is None:
        raise ValueError("Input image is None in grayscale_opencv()")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def grayscale_all_methods(image):
    """
    Apply different grayscale methods: Average, Luminosity, Desaturation, OpenCV.

    Parameters:
    - image: Input image (BGR).

    Returns:
    - Dictionary containing all grayscale versions:
        {
            'average': ...,
            'luminosity': ...,
            'desaturation': ...,
            'opencv': ...
        }
    """
    if image is None:
        raise ValueError("Input image is None in grayscale_all_methods()")

    # Method 1: Average Method
    average_gray = np.mean(image, axis=2).astype(np.uint8)

    # Method 2: Luminosity Method (BGR weights)
    weights = [0.114, 0.587, 0.299]  # BGR order
    luminosity_gray = np.dot(image[..., :3], weights).astype(np.uint8)

    # Method 3: Desaturation Method
    max_channel = np.max(image, axis=2)
    min_channel = np.min(image, axis=2)
    desaturation_gray = ((max_channel + min_channel) / 2).astype(np.uint8)

    # Method 4: OpenCV Built-in
    opencv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return {
        'average': average_gray,
        'luminosity': luminosity_gray,
        'desaturation': desaturation_gray,
        'opencv': opencv_gray
    }


# Optional: Standalone test (not needed in main pipeline, but good for dev)
if __name__ == "__main__":
    import os

    input_path = 'input.jpg'
    output_dir = 'output_grayscale'

    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        print("Error: Could not read input image.")
        exit()

    # Get all grayscale versions
    results = grayscale_all_methods(img)

    for method, gray_img in results.items():
        cv2.imwrite(os.path.join(output_dir, f'gray_{method}.jpg'), gray_img)

    print(f"[INFO] Grayscale images saved to {output_dir}")
