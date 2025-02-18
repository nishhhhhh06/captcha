import os
import cv2
import logging
import numpy as np
from paddleocr import PaddleOCR

logging.getLogger("ppocr").setLevel(logging.ERROR)
ocr = PaddleOCR(use_angle_cls=True, lang="en")

DATA_DIR = "test_images"
OUTPUT_FILE = "extracted_captchas.txt"

def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    result = ocr.ocr(img, cls=True)
    
    extracted_text = ""
    for line in result:
        for word_info in line:
            extracted_text += word_info[1][0]
            
    return extracted_text.strip()

def process_captchas():
    extracted_data = []
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".jpeg"):
            img_path = os.path.join(DATA_DIR, filename)
            extracted_text = extract_text_from_image(img_path)
            extracted_data.append(f"{filename}: {extracted_text}")
            print(f"Extracted from {filename}: {extracted_text}")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(extracted_data))

    print(f"\n✅ Extracted CAPTCHA texts saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_captchas()
