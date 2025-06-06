# preprocessing/dilation_erosion.py

import cv2
import numpy as np

def apply_morph_operations(image, operation="dilation", kernel_size=3, iterations=1):
    """
    Apply morphological operations to a binary or grayscale image.
    
    Args:
        image (np.ndarray): Input image (grayscale or binary).
        operation (str): One of ["dilation", "erosion", "opening", "closing"].
        kernel_size (int): Size of structuring element.
        iterations (int): Number of iterations to apply operation.

    Returns:
        np.ndarray: Processed image (binary with black text on white background).
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to binary with text in white
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Structuring element (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply selected morphological operation
    if operation == "dilation":
        processed = cv2.dilate(binary, kernel, iterations=iterations)
    elif operation == "erosion":
        processed = cv2.erode(binary, kernel, iterations=iterations)
    elif operation == "opening":
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "closing":
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError("Operation must be one of: 'dilation', 'erosion', 'opening', 'closing'.")

    # Invert back to original format: black text on white
    result = cv2.bitwise_not(processed)
    return result


# Optional visual test
if __name__ == "__main__":
    import os

    image = cv2.imread("document.jpg", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: 'document.jpg' not found.")
        exit()

    ops = ["dilation", "erosion", "opening", "closing"]
    for op in ops:
        output = apply_morph_operations(image, operation=op)
        output_path = f"{op}.jpg"
        cv2.imwrite(output_path, output)
        print(f"Saved {output_path}")

        cv2.imshow(f"{op.capitalize()}", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
