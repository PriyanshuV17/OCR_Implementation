README.txt
==========

Project Title: OCR Implementation Pipeline
--------------------------------------------

Overview:
------------
This project is a modular Optical Character Recognition (OCR) pipeline built using Python, OpenCV, and PyTorch. It extracts text from images by combining image preprocessing, text detection, and deep learning-based character recognition.

It can be used in applications like:
- Number plate detection
- Document digitization
- Automated form reading
- Scene text extraction

Key Features:
----------------
- Grayscale conversion
- Noise reduction using Gaussian blur
- Thresholding (Otsu, Adaptive, or Sauvola)
- Deskewing (rotation correction)
- Morphological operations (dilation, erosion, closing)
- Rescaling image to improve OCR accuracy
- Bounding box-based text detection using contouring
- OCR using pre-trained PyTorch model
- Confidence scores for each recognized character
- Annotated output image saved with bounding boxes and recognized text
- Clean extracted text printed at the end in terminal

Folder Structure:
---------------------
OCR_Implementation/
├── main.py
├── input.jpg
├── output_annotated.jpg
├── preprocessing/
│   ├── grayscale.py
│   ├── noise_reduction.py
│   ├── thresholding.py
│   ├── deskewing.py
│   ├── dilation_erosion.py
│   └── rescaling.py
├── text_detection/
│   └── text_detection.py
├── character_recognition/
│   └── character_recognition.py

How to Run:
---------------
1. Install required libraries:
   pip install opencv-python torch torchvision numpy matplotlib scikit-image

2. Place your input image at the specified location (`input.jpg`) or update the path in `main.py`.

3. Run the script:
   python3 main.py

4. Terminal Output:
   - Each detected text segment with bounding box, confidence, and location
   - Final extracted clean text at the end
   - Annotated image saved as output_annotated.jpg

Sample Terminal Output:
---------------------------
📄 OCR Results:
----------------------------------------
1. Text: Hello | Confidence: 0.95 | Box: (100, 110, 200, 150)
2. Text: World | Confidence: 0.91 | Box: (210, 110, 310, 150)

📝 Extracted Text:
-------------------------
Hello World

Notes:
---------------------
- Ensure the image quality is good for better accuracy.
- GPU acceleration is supported (set use_gpu=True), but code falls back to CPU if unavailable.
- Easily extendable with Tesseract or transformer-based OCR backends.

Contact:
-----------
Developed by:

Priyanshu Verma  
Mobile: +91 7376021218
