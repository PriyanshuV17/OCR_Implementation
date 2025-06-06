import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def determine_skew(image):
    """
    Calculate the skew angle of a document image using projection profile analysis
    Args:
        image: numpy array (binary or grayscale image)
    Returns:
        float: detected skew angle in degrees
    """
    # Convert to binary if needed and invert (text should be white)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Test range of angles and find the one with maximum variance
    angles = np.arange(-15, 15, 0.5)
    scores = []
    for angle in angles:
        rotated = inter.rotate(binary, angle, reshape=False, order=0)
        hist = np.sum(rotated, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        scores.append(score)
        
    return angles[np.argmax(scores)]

def deskew_image(image, angle=None, border_value=255):
    """
    Deskew an image by rotating it
    Args:
        image: numpy array (input image)
        angle: float (skew angle in degrees). If None, will auto-detect
        border_value: int (value to use for background padding)
    Returns:
        numpy array: deskewed image
    """
    if angle is None:
        angle = determine_skew(image)
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                           flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=border_value)
    return rotated

def show_comparison(original, processed, title="Deskewed Result"):
    """Display original and processed images side by side"""
    import matplotlib.pyplot as plt
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

# Example usage
if __name__ == "__main__":
    # Load your document image
    img_path = "skewed_document.jpg"
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # or IMREAD_COLOR
    
    # Option 1: Auto-detect and correct skew
    deskewed_auto = deskew_image(image)
    show_comparison(image, deskewed_auto, "Auto-deskewed")
    
    # Option 2: Manual angle correction
    # deskewed_manual = deskew_image(image, angle=2.5)
    
    # Save the result
    cv2.imwrite("deskewed_output.jpg", deskewed_auto)