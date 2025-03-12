import cv2
import easyocr
import numpy as np

# Load video
video_path = "demo2.mp4"
cap = cv2.VideoCapture(video_path)

# Force EasyOCR to use CPU
reader = easyocr.Reader(['en'], gpu=False)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # Possible license plate
            x, y, w, h = cv2.boundingRect(cnt)
            plate = frame[y:y+h, x:x+w]
            
            # OCR recognition
            result = reader.readtext(plate)
            
            for (bbox, text, prob) in result:
                if prob > 0.5:  # Confidence threshold
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Show result
    cv2.imshow("License Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()