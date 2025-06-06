import cv2
import numpy as np

def rescale_image(image, scale_factor=None, target_height=None, target_width=None, interpolation=cv2.INTER_CUBIC):
    """
    Rescale an image for optimal OCR processing.
    
    Args:
        image: Input image (numpy array or file path)
        scale_factor: Optional scaling multiplier (e.g., 2.0 = 2x enlargement)
        target_height: Desired output height (maintains aspect ratio)
        target_width: Desired output width (maintains aspect ratio)
        interpolation: Method for resizing (default: cv2.INTER_CUBIC for enlargement)
        
    Returns:
        Resized image (numpy array)
    """
    if isinstance(image, str):
        image = cv2.imread(image)
    
    h, w = image.shape[:2]
    
    # Calculate new dimensions
    if scale_factor:
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    elif target_height and target_width:
        new_w, new_h = target_width, target_height
    elif target_height:
        new_w = int(w * (target_height / h))
        new_h = target_height
    elif target_width:
        new_h = int(h * (target_width / w))
        new_w = target_width
    else:
        raise ValueError("Must specify scale_factor, target_height, or target_width")
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized

# Example Usage
if __name__ == "__main__":
    # Load image
    img = cv2.imread("document.jpg")
    
    # Option 1: Scale by factor (2x larger)
    upscaled = rescale_image(img, scale_factor=2.0)
    
    # Option 2: Scale to target height (600px, keeps aspect ratio)
    height_scaled = rescale_image(img, target_height=600)
    
    # Option 3: Scale to exact dimensions (may stretch)
    exact_scaled = rescale_image(img, target_width=800, target_height=600)
    
    # Save results
    cv2.imwrite("upscaled.jpg", upscaled)
    cv2.imwrite("height_scaled.jpg", height_scaled)
    cv2.imwrite("exact_scaled.jpg", exact_scaled)
    
    # Display comparisons
    def show_comparison(original, resized, title):
        cv2.imshow("Original", original)
        cv2.imshow(title, resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    show_comparison(img, upscaled, "2x Upscaled (INTER_CUBIC)")
    show_comparison(img, height_scaled, "Height=600px (Aspect Ratio Kept)")