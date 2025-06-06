import cv2
from text_detection.text_detection import detect_text_regions
from character_recognition.character_recognition import recognize_text_from_bboxes

# Load preprocessed image
image_path = "path_to_preprocessed_image.png"
image = cv2.imread(image_path)

# Detect text bounding boxes
bboxes = detect_text_regions(image)  # Should return [(x1, y1, x2, y2), ...]

# Recognize text from these regions
results = recognize_text_from_bboxes(image, bboxes)

# Print the output
for r in results:
    print(f"Text: {r['text']}, Confidence: {r['confidence']:.2f}, Box: {r['bbox']}")
