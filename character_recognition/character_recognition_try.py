# character_recognition/character_recognition.py

import easyocr
import cv2

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

def recognize_text_from_bboxes(image, bboxes):
    """
    Performs character/word recognition from detected text regions.

    Parameters:
    - image: Original or preprocessed image (BGR or grayscale)
    - bboxes: List of bounding boxes [(x1, y1, x2, y2), ...]

    Returns:
    - results: List of dicts with text and confidence scores
    """
    results = []

    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cropped = image[y1:y2, x1:x2]

        # OCR the cropped region
        ocr_result = reader.readtext(cropped)

        for _, text, conf in ocr_result:
            results.append({
                'bbox': bbox,
                'text': text,
                'confidence': conf
            })

    return results
