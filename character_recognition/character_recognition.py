# character_recognition/character_recognition.py

import easyocr
import cv2

class OCRRecognizer:
    def __init__(self, lang_list=['en'], use_gpu=False):
        """
        Initialize the EasyOCR reader.
        :param lang_list: List of language codes (default: ['en'])
        :param use_gpu: Whether to use GPU (default: False)
        """
        self.reader = easyocr.Reader(lang_list, gpu=use_gpu)

    def recognize_text_from_bboxes(self, image, bboxes, conf_threshold=0.0, save_debug=False):
        """
        Perform text recognition on specified bounding boxes.

        Parameters:
        - image: Input image (BGR or grayscale)
        - bboxes: List of bounding boxes [(x1, y1, x2, y2), ...]
        - conf_threshold: Minimum confidence to include result
        - save_debug: If True, saves cropped regions for inspection

        Returns:
        - results: List of dicts with bbox, text, and confidence
        """
        results = []
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        for idx, (x1, y1, x2, y2) in enumerate(bboxes):
            roi = image_gray[y1:y2, x1:x2]

            # OCR on the cropped region
            ocr_result = self.reader.readtext(roi)

            for _, text, conf in ocr_result:
                if conf >= conf_threshold:
                    results.append({
                        'bbox': (x1, y1, x2, y2),
                        'text': text,
                        'confidence': conf
                    })

            if save_debug:
                cv2.imwrite(f"debug/cropped_{idx}.jpg", roi)

        return results


# Example usage
if __name__ == "__main__":
    import json
    import os
    from preprocessing.text_detection import TextDetector

    # Ensure debug folder exists
    os.makedirs("debug", exist_ok=True)

    # Step 1: Load image
    img = cv2.imread("document.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 2: Detect text boxes
    detector = TextDetector(binary)
    detector.find_text_contours(min_area=30, aspect_ratio_range=(0.2, 8))
    detector.merge_overlapping_boxes()
    detector.sort_boxes()

    # Step 3: Recognize text
    recognizer = OCRRecognizer(use_gpu=False)
    ocr_results = recognizer.recognize_text_from_bboxes(img, detector.boxes, conf_threshold=0.5, save_debug=True)

    # Print or save results
    print(json.dumps(ocr_results, indent=2))
