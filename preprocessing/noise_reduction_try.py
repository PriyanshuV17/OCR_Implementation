import cv2
import numpy as np

def apply_noise_removal(input_path, output_path):
    """
    Applies noise removal techniques to a grayscale image.
    Steps:
        1. Read the image in grayscale.
        2. Apply Gaussian Blur (for general noise).
        3. Apply Median Blur (for salt-and-pepper noise).
        4. Apply Adaptive Thresholding (for binarization).
        5. Save the cleaned image.
    """
    # Step 1: Read the image (already in grayscale)
    gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        print("Error: Image not found or invalid path!")
        return

    # Step 2: Gaussian Blur (reduces high-frequency noise)
    gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 3: Median Blur (removes salt-and-pepper noise)
    median_blur = cv2.medianBlur(gaussian_blur, 3)

    # Step 4: Adaptive Thresholding (binarizes text/background)
    denoised_image = cv2.adaptiveThreshold(
        median_blur, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV,  # Use THRESH_BINARY for white text on black
        11, 
        2
    )

    # Step 5: Save the result
    cv2.imwrite(output_path, denoised_image)
    print(f"Denoised image saved to: {output_path}")

    # (Optional) Display results
    cv2.imshow("Original Grayscale", gray_image)
    cv2.imshow("Denoised Image", denoised_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
input_image_path = "input_grayscale.png"  # Replace with your image path
output_image_path = "denoised_output.png"
apply_noise_removal(input_image_path, output_image_path)