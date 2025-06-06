# preprocessing/text_detection.py

import cv2
import numpy as np
from matplotlib import pyplot as plt

class TextDetector:
    def __init__(self, preprocessed_img):
        """
        Initialize with a preprocessed binary image (text = white, background = black)
        :param preprocessed_img: Binary image (np.ndarray)
        """
        self.image = preprocessed_img
        self.boxes = []
        
    def detect_text_edges(self, canny_low=50, canny_high=150):
        """Detect edges using Canny (optional step before contours)"""
        edges = cv2.Canny(self.image, canny_low, canny_high)
        return edges

    def find_text_contours(self, min_area=50, max_area=5000, aspect_ratio_range=(0.1, 10)):
        """
        Detect text-like contours in the binary image.
        """
        contours, _ = cv2.findContours(self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.boxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            aspect_ratio = w / float(h)

            if min_area < area < max_area and aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1]:
                self.boxes.append((x, y, x + w, y + h))

        return self.boxes

    def merge_overlapping_boxes(self, overlap_thresh=0.3):
        """
        Apply non-maximum suppression (NMS) to merge overlapping bounding boxes.
        """
        if not self.boxes:
            return []

        boxes = np.array(self.boxes)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)  # Bottom-right Y

        pick = []
        while len(idxs) > 0:
            last = idxs[-1]
            pick.append(last)
            suppress = [last]

            for pos in idxs[:-1]:
                xx1 = max(x1[last], x1[pos])
                yy1 = max(y1[last], y1[pos])
                xx2 = min(x2[last], x2[pos])
                yy2 = min(y2[last], y2[pos])

                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)
                overlap = float(w * h) / areas[pos]

                if overlap > overlap_thresh:
                    suppress.append(pos)

            idxs = np.setdiff1d(idxs, suppress)

        self.boxes = boxes[pick].tolist()
        return self.boxes

    def sort_boxes(self):
        """Sort boxes in reading order (top to bottom, then left to right)."""
        self.boxes = sorted(self.boxes, key=lambda b: (b[1], b[0]))
        return self.boxes

    def visualize(self, output_path=None, show=True, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes on image.
        """
        vis = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR) if len(self.image.shape) == 2 else self.image.copy()

        for (x1, y1, x2, y2) in self.boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        if output_path:
            cv2.imwrite(output_path, vis)

        if show:
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

        return vis

    def extract_text_regions(self, original_image, output_dir=None):
        """
        Extract detected text regions from the original image.
        :param original_image: Original BGR image
        :param output_dir: Directory to save the cropped images
        :return: List of ROI images
        """
        rois = []
        for i, (x1, y1, x2, y2) in enumerate(self.boxes):
            roi = original_image[y1:y2, x1:x2]
            rois.append(roi)
            if output_dir:
                cv2.imwrite(f"{output_dir}/text_region_{i}.jpg", roi)
        return rois


# Example usage
if __name__ == "__main__":
    img = cv2.imread("document.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    detector = TextDetector(binary)
    detector.find_text_contours(min_area=30, aspect_ratio_range=(0.2, 8))
    detector.merge_overlapping_boxes()
    detector.sort_boxes()
    detector.visualize(output_path="text_regions.jpg")

    # Optional: extract regions
    detector.extract_text_regions(img, output_dir="output")
