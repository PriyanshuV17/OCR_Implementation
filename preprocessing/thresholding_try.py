import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from matplotlib import pyplot as plt

class OCRThresholding:
    def __init__(self, image_path):
        """Initialize with the image path"""
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
        
    def apply_global_threshold(self, show_result=False):
        """Apply global thresholding using Otsu's method"""
        _, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if show_result:
            self._display_results(self.gray, binary, "Global (Otsu) Thresholding")
        return binary
    
    def apply_adaptive_threshold(self, block_size=11, C=2, show_result=False):
        """Apply adaptive thresholding"""
        binary = cv2.adaptiveThreshold(self.gray, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, block_size, C)
        if show_result:
            self._display_results(self.gray, binary, "Adaptive Thresholding")
        return binary
    
    def apply_sauvola_threshold(self, window_size=25, k=0.2, show_result=False):
        """Apply Sauvola's thresholding"""
        thresh = threshold_sauvola(self.gray, window_size=window_size, k=k)
        binary = (self.gray > thresh).astype('uint8') * 255
        if show_result:
            self._display_results(self.gray, binary, "Sauvola's Thresholding")
        return binary
    
    def _display_results(self, original, processed, title):
        """Display original and processed images side by side"""
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
    # Initialize with your image path
    thresholder = OCRThresholding('your_image.jpg')
    
    # Choose the appropriate method based on your image:
    # 1. For clean documents with uniform background
    binary_global = thresholder.apply_global_threshold(show_result=True)
    
    # 2. For documents with uneven lighting
    binary_adaptive = thresholder.apply_adaptive_threshold(show_result=True)
    
    # 3. For noisy or poor quality documents
    binary_sauvola = thresholder.apply_sauvola_threshold(show_result=True)
    
    # Save the best result
    cv2.imwrite('binary_output.jpg', binary_sauvola)  # Change to your preferred method