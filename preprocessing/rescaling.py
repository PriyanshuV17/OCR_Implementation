# preprocessing/rescale_image.py

import cv2
import numpy as np

def rescale_image(image, scale_factor=None, target_height=None, target_width=None, interpolation=cv2.INTER_CUBIC):
    """
    Rescale an image for optimal OCR preprocessing.

    Args:
        image (np.ndarray or str): Input image (grayscale, BGR, or path).
        scale_factor (float): Optional scale multiplier (e.g., 2.0).
        target_height (int): Resize to this height while keeping aspect ratio.
        target_width (int): Resize to this width while keeping aspect ratio.
        interpolation (cv2 constant): Interpolation method, default is INTER_CUBIC.

    Returns:
        np.ndarray: Rescaled image.
    """
    # Load from path if a string is passed
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image}")

    h, w = image.shape[:2]

    # Compute new size
    if scale_factor:
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    elif target_height and target_width:
        new_w, new_h = target_width, target_height
    elif target_height:
        scale = target_height / float(h)
        new_h = target_height
        new_w = int(w * scale)
    elif target_width:
        scale = target_width / float(w)
        new_w = target_width
        new_h = int(h * scale)
    else:
        raise ValueError("Must specify scale_factor, target_height, or target_width")

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized


# Optional direct test
if __name__ == "__main__":
    import os

    image_path = "document.jpg"
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        exit()

    options = {
        "upscaled": rescale_image(img, scale_factor=2.0),
        "height_scaled": rescale_image(img, target_height=600),
        "exact_scaled": rescale_image(img, target_width=800, target_height=600)
    }

    for label, output_img in options.items():
        output_file = f"{label}.jpg"
        cv2.imwrite(output_file, output_img)
        print(f"Saved: {output_file}")
        cv2.imshow(label, output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
