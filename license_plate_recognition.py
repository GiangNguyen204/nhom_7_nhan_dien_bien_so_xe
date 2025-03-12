import cv2
import numpy as np
import easyocr
import re
import os
from ultralytics import YOLO

# Định nghĩa model nhận diện biển số
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# Khởi tạo EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Nhãn biển số hợp lệ
license_plate_pattern = r"[A-Z0-9\-]{5,10}"

# Tạo thư mục lưu biển số
os.makedirs("plates", exist_ok=True)

def preprocess_license_plate(image):
    """Cải thiện chất lượng ảnh biển số trước khi OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    contrast = cv2.equalizeHist(blur)
    binary = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return processed

def format_license(text):
    """Chuẩn hóa biển số xe"""
    text = text.upper().replace(' ', '')
    replacements = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
    corrected_text = ''.join(replacements.get(c, c) for c in text)
    return corrected_text if re.match(license_plate_pattern, corrected_text) else None

def read_license_plate(license_plate_crop, plate_id):
    """Nhận diện biển số từ ảnh cắt"""
    processed_plate = preprocess_license_plate(license_plate_crop)

    # Lưu ảnh để kiểm tra
    plate_path = f"plates/plate_{plate_id}.png"
    cv2.imwrite(plate_path, processed_plate)

    detections = reader.readtext(processed_plate)
    for bbox, text, score in detections:
        print(f"OCR phát hiện: {text} (Score: {score:.2f})")
        formatted_text = format_license(text)
        if formatted_text:
            return formatted_text, score
    return None, None

def detect_license_plate(frame, vehicle_box):
    """Tìm vùng biển số trong xe"""
    x1, y1, x2, y2 = vehicle_box
    vehicle_crop = frame[y1:y2, x1:x2]
    plate_results = license_plate_detector(vehicle_crop, conf=0.4)

    for plate in plate_results:
        for box in plate.boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            return plate_crop, (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
    return None, None
