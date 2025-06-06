# main.py

import cv2
import os
from preprocessing.grayscale import grayscale_opencv
from preprocessing.noise_reduction import reduce_noise
from preprocessing.thresholding import apply_threshold
from preprocessing.deskewing import deskew_image
from preprocessing.dilation_erosion import apply_morph_operations
from preprocessing.rescaling import rescale_image
from text_detection.text_detection import TextDetector
from character_recognition.character_recognition import OCRRecognizer

def main(image_path):
    # Step 1: Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image from path: {image_path}")
        return

    # Step 2: Preprocessing Pipeline
    gray = grayscale_opencv(image)
    denoised = reduce_noise(gray)
    thresholded = apply_threshold(denoised)
    deskewed = deskew_image(thresholded)
    morphed = apply_morph_operations(deskewed, operation="closing", kernel_size=3, iterations=1)
    resized = rescale_image(morphed, scale_factor=2.0)

    # Step 3: Text Detection
    detector = TextDetector(resized)
    detector.find_text_contours(min_area=30, aspect_ratio_range=(0.2, 8))
    detector.merge_overlapping_boxes()
    detector.sort_boxes()
    bboxes = detector.boxes

    # Step 4: OCR Recognition
    recognizer = OCRRecognizer(use_gpu=False)
    results = recognizer.recognize_text_from_bboxes(resized, bboxes)

    # Step 5: Output Results
    print("\n📄 OCR Results:\n" + "-"*40)
    for i, res in enumerate(results):
        print(f"{i+1}. Text: {res['text']} | Confidence: {res['confidence']:.2f} | Box: {res['bbox']}")

    # Optional: Visualize
    annotated = detector.visualize(show=False)
    output_img = "output_annotated.jpg"
    cv2.imwrite(output_img, annotated)
    print(f"\n✅ Annotated image saved as {output_img}")

if __name__ == "__main__":
    image_path = "/home/cslab2/Desktop/priyanshu_verma/OCR_Implementation/input.jpg"  # <- replace this with your actual input
    main(image_path)

