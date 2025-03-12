import cv2
import numpy as np
import easyocr
import re
import os
from ultralytics import YOLO

# Định nghĩa đường dẫn model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LICENSE_PLATE_MODEL_PATH = os.path.join(BASE_DIR, "models", "license_plate_detector.pt")
VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

# Kiểm tra xem model có tồn tại không
if not os.path.exists(LICENSE_PLATE_MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model biển số: {LICENSE_PLATE_MODEL_PATH}")

if not os.path.exists(VEHICLE_MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model phương tiện: {VEHICLE_MODEL_PATH}")

# Khởi tạo EasyOCR và YOLO
reader = easyocr.Reader(['en'], gpu=False)
vehicle_model = YOLO(VEHICLE_MODEL_PATH)  # Model nhận diện phương tiện
plate_model = YOLO(LICENSE_PLATE_MODEL_PATH)  # Model nhận diện biển số

# Nhãn phương tiện từ COCO dataset
coco_labels = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Biểu thức regex cho biển số xe
license_plate_pattern = r"^[0-9A-Z]{2,3}-?[0-9]{4,5}$"

def preprocess_license_plate(image):
    """Xử lý ảnh biển số trước khi OCR"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return processed

def format_license(text):
    """Chuẩn hóa biển số xe, sửa lỗi OCR"""
    text = text.upper().replace(' ', '')
    replacements = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
    corrected_text = ''.join(replacements.get(c, c) for c in text)
    if re.match(license_plate_pattern, corrected_text):
        return corrected_text
    return None

def read_license_plate(license_plate_crop):
    """Nhận diện biển số xe từ ảnh cắt"""
    processed_plate = preprocess_license_plate(license_plate_crop)
    detections = reader.readtext(processed_plate)
    for bbox, text, score in detections:
        text = format_license(text)
        if text:
            return text, score
    return None, None

def detect_license_plate(frame, vehicle_box):
    """Tìm vùng biển số trong vùng phương tiện"""
    x1, y1, x2, y2 = vehicle_box
    vehicle_crop = frame[y1:y2, x1:x2]
    plate_results = plate_model(vehicle_crop, conf=0.4)
    for plate in plate_results:
        for box in plate.boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            plate_crop = vehicle_crop[py1:py2, px1:px2]
            return plate_crop, (x1 + px1, y1 + py1, x1 + px2, y1 + py2)
    return None, None

def detect_traffic_violation(video_path):
    """Nhận diện phương tiện vi phạm đèn đỏ và đọc biển số"""
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * 0.42)  # Vị trí vạch đèn đỏ
    violators = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = vehicle_model(frame, conf=0.3)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                track_id = box.id[0] if box.id is not None else None
                if cls in coco_labels:
                    y_center = (y1 + y2) // 2
                    if y1 < line_y and y_center > line_y:
                        vehicle_type = coco_labels[cls]
                        if track_id and track_id in violators:
                            continue
                        plate_crop, plate_box = detect_license_plate(frame, (x1, y1, x2, y2))
                        plate_text, plate_score = None, None
                        if plate_crop is not None:
                            plate_text, plate_score = read_license_plate(plate_crop)
                        cv2.putText(frame, "Da vuot den do!", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, f"Loai: {vehicle_type}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        if not plate_text:
                            plate_text = "Khong nhan dien duoc"
                        cv2.putText(frame, f"Bien so: {plate_text}", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        print(f"Phuong tien {vehicle_type} co bien so {plate_text} da vi pham!")
                        if track_id:
                            violators.add(track_id)
                        else:
                            violators.add((x1, y1, x2, y2))
                        if plate_box:
                            px1, py1, px2, py2 = plate_box
                            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                            cv2.putText(frame, f"OCR Score: {plate_score:.2f}", (px1, py1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "demo2.mp4"
    detect_traffic_violation(video_path)
