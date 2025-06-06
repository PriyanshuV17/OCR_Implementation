import cv2
import numpy as np
from matplotlib import pyplot as plt

class TextDetector:
    def __init__(self, preprocessed_img):
        """
        Initialize with a preprocessed binary image (black text on white background)
        :param preprocessed_img: Binary numpy array from Step 2 preprocessing
        """
        self.image = preprocessed_img
        self.boxes = []
        
    def detect_text_edges(self, canny_low=50, canny_high=150):
        """Detect text regions using Canny edge detection"""
        edges = cv2.Canny(self.image, canny_low, canny_high)
        return edges
    
    def find_text_contours(self, min_area=50, max_area=5000, aspect_ratio_range=(0.1, 10)):
        """
        Find contours and filter for text-like regions
        :param min_area: Minimum contour area to consider
        :param max_area: Maximum contour area to consider
        :param aspect_ratio_range: Valid width/height ratio range for text
        """
        # Find contours (OpenCV 4.x syntax)
        contours, _ = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        self.boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(cnt)
            
            # Text-like region filters
            if (aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1] and
                min_area < area < max_area):
                self.boxes.append((x, y, x+w, y+h))
        
        return self.boxes
    
    def merge_overlapping_boxes(self, overlap_thresh=0.3):
        """Non-maximum suppression to merge overlapping boxes"""
        if len(self.boxes) == 0:
            return []
        
        boxes = np.array(self.boxes)
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)  # Sort by bottom y-coordinate
        
        pick = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            suppress = [last]
            for pos in range(0, last):
                j = idxs[pos]
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y1[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])
                
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)
                overlap = float(w * h) / areas[j]
                
                if overlap > overlap_thresh:
                    suppress.append(pos)
            
            idxs = np.delete(idxs, suppress)
        
        self.boxes = boxes[pick]
        return self.boxes
    
    def sort_boxes(self):
        """Sort boxes in reading order (top-to-bottom, left-to-right)"""
        self.boxes = sorted(self.boxes, key=lambda b: (b[1], b[0]))
        return self.boxes
    
    def visualize(self, output_path=None, show=True):
        """Draw bounding boxes on the original image"""
        if len(self.image.shape) == 2:  # Convert grayscale to BGR for visualization
            vis = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        else:
            vis = self.image.copy()
        
        for (x1, y1, x2, y2) in self.boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis)
        
        if show:
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        
        return vis

# Example Usage
if __name__ == "__main__":
    # Load and preprocess image (Step 1 & 2)
    img = cv2.imread('document.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Initialize detector with preprocessed image
    detector = TextDetector(binary)
    
    # Step 1: Find text regions
    detector.find_text_contours(min_area=30, aspect_ratio_range=(0.2, 8))
    
    # Step 2: Merge overlapping boxes
    detector.merge_overlapping_boxes()
    
    # Step 3: Sort in reading order
    detector.sort_boxes()
    
    # Visualize and save results
    detector.visualize(output_path='text_regions.jpg')
    
    # Optional: Extract individual text regions for OCR
    for i, (x1, y1, x2, y2) in enumerate(detector.boxes):
        roi = img[y1:y2, x1:x2]
        cv2.imwrite(f'text_region_{i}.jpg', roi)