import cv2
import csv
import os
import requests
from ultralytics import YOLO
from license_plate_recognition import detect_license_plate, read_license_plate

# Khởi tạo model nhận diện phương tiện
VEHICLE_MODEL_PATH = "yolov8n.pt"
vehicle_model = YOLO(VEHICLE_MODEL_PATH)

# Nhãn phương tiện từ COCO dataset
coco_labels = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Thư mục lưu ảnh vi phạm
os.makedirs("vehicles_violation", exist_ok=True)
os.makedirs("plates_detected", exist_ok=True)

# API Token Telegram Bot và Chat ID
# TELEGRAM_BOT_TOKEN = "7413376592:AAEEWYDNOT2SL3CHW3M71mVOJNtrrykf2no"  # API Token của bot của bạn
# TELEGRAM_CHAT_ID = "-4679083892"  # Chat ID của bạn

# def send_to_telegram(image_path, message):
#     """Gửi ảnh về Telegram kèm theo thông báo"""
#     url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
#     with open(image_path, 'rb') as photo:
#         response = requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'caption': message}, files={'photo': photo})
#     return response

def detect_traffic_violation(video_path):
    """Nhận diện phương tiện vi phạm và đọc biển số"""
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * 0.42)  # Vị trí vạch đèn đỏ
    violators = set()
    results_csv = []

    frame_id = 0  # Biến đếm frame để debug
    while True:
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
                    vehicle_type = coco_labels[cls]
                    y_center = (y1 + y2) // 2
                    
                    # Kiểm tra phương tiện có vượt đèn đỏ không
                    if y1 < line_y and y_center > line_y:
                        if track_id and track_id in violators:
                            continue

                        print(f"[DEBUG] Phát hiện {vehicle_type} vượt đèn đỏ tại frame {frame_id}")
                        
                        plate_crop, plate_box = detect_license_plate(frame, (x1, y1, x2, y2))
                        plate_text, plate_score = None, None
                        plate_filename = None  # Đảm bảo rằng plate_filename luôn được khai báo

                        if plate_crop is not None:
                            plate_text, plate_score = read_license_plate(plate_crop, frame_id)
                            plate_filename = f"plates_detected/plate_{frame_id}.png"
                            cv2.imwrite(plate_filename, plate_crop)
                            print(f"[DEBUG] Đã lưu ảnh biển số: {plate_filename}")
                        
                        # Lưu ảnh phương tiện vi phạm
                        vehicle_filename = f"vehicles_violation/vehicle_{frame_id}.png"
                        cv2.imwrite(vehicle_filename, frame[y1:y2, x1:x2])
                        print(f"[DEBUG] Đã lưu ảnh phương tiện: {vehicle_filename}")
                        
                        if not plate_text:
                            plate_text = "Khong nhan dien duoc"
                        
                        cv2.putText(frame, "Vuot den do!", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, f"Loai: {vehicle_type}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        
                        # Hiển thị biển số xe trên video chỉ nếu có biển số
                        if plate_text != "Khong nhan dien duoc":
                            cv2.putText(frame, f"Bien so: {plate_text}", (x1, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                        print(f"[DEBUG] Phuong tien {vehicle_type} - Bien so: {plate_text}")

                        if track_id:
                            violators.add(track_id)
                        else:
                            violators.add((x1, y1, x2, y2))

                        if plate_box:
                            px1, py1, px2, py2 = plate_box
                            cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2)
                            if plate_score is not None:
                                cv2.putText(frame, f"OCR: {plate_score:.2f}", (px1, py1 - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                        results_csv.append([vehicle_type, plate_text])

                        # Gửi thông báo vi phạm và ảnh về Telegram
                        # violation_message = f"Phương tiện vi phạm: {vehicle_type}\nBiển số: {plate_text}\nVượt đèn đỏ!"
                        # send_to_telegram(vehicle_filename, violation_message)  # Gửi ảnh phương tiện vi phạm về Telegram
                        
                        # if plate_filename:  # Chỉ gửi ảnh biển số nếu có
                        #     send_to_telegram(plate_filename, violation_message)  # Gửi ảnh biển số xe về Telegram
        
                        # cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        
        # Hiển thị video trên màn hình laptop
        cv2.imshow("Video", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        frame_id += 1
    
    cap.release()
    cv2.destroyAllWindows()

    # Xuất kết quả ra CSV
    with open("violators.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Loai Phuong Tien", "Bien So"])
        writer.writerows(results_csv)
    print("Đã lưu kết quả vào violators.csv")

if __name__ == "__main__":
    video = "7.mp4"  # Đường dẫn video của bạn
    if not os.path.exists(video):
        print(f"[ERROR] Không tìm thấy video: {video}")
    else:
        detect_traffic_violation(video)
