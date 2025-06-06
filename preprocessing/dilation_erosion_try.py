import cv2
import numpy as np

def apply_morph_operations(image, operation="dilation", kernel_size=3, iterations=1):
    """
    Apply dilation/erosion to a binary image (black text on white background).
    
    Args:
        image: Binary input image (numpy array)
        operation: "dilation", "erosion", "opening", "closing"
        kernel_size: Size of the structuring element (odd integer)
        iterations: Number of times to apply the operation
        
    Returns:
        Processed image (numpy array)
    """
    # Ensure binary image (text=black, background=white)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV)
    
    # Define kernel (structuring element)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply operation
    if operation == "dilation":
        result = cv2.dilate(binary, kernel, iterations=iterations)
    elif operation == "erosion":
        result = cv2.erode(binary, kernel, iterations=iterations)
    elif operation == "opening":
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "closing":
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError("Operation must be 'dilation', 'erosion', 'opening', or 'closing'")
    
    # Revert to original format (black text on white)
    result = cv2.bitwise_not(result)
    return result

# Example Usage
if __name__ == "__main__":
    # Load image (grayscale or binary)
    image = cv2.imread("document.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Apply operations
    dilated = apply_morph_operations(image, "dilation", kernel_size=3, iterations=1)
    eroded = apply_morph_operations(image, "erosion", kernel_size=3, iterations=1)
    opened = apply_morph_operations(image, "opening", kernel_size=3, iterations=1)
    closed = apply_morph_operations(image, "closing", kernel_size=3, iterations=1)
    
    # Save results
    cv2.imwrite("dilated.jpg", dilated)
    cv2.imwrite("eroded.jpg", eroded)
    cv2.imwrite("opened.jpg", opened)
    cv2.imwrite("closed.jpg", closed)
    
    # Display comparisons
    def show_comparison(original, processed, title):
        cv2.imshow("Original", original)
        cv2.imshow(title, processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    show_comparison(image, dilated, "Dilation")
    show_comparison(image, eroded, "Erosion")
    show_comparison(image, opened, "Opening (Erosion → Dilation)")
    show_comparison(image, closed, "Closing (Dilation → Erosion)")