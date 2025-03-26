from ultralytics import YOLO
import cv2
import torch

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" or "yolov8m.pt" for better accuracy

# Set device to CPU explicitly
device = 'cpu'
model.to(device)

# Open a webcam feed
cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform real-time object detection
    results = model(frame)
    
    # Visualize results on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index
            label = f"{model.names[cls]}: {conf:.2f}"
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("YOLOv8 Real-Time Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
