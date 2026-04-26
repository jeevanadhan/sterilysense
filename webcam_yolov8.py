import cv2
import time

try:
    from ultralytics import YOLO
except Exception as e:
    print("ultralytics not installed. Install with: pip install ultralytics")
    raise

MODEL_PATH = 'yolov8n.pt'
CAM_INDEX = 0
IMG_SIZE = 640
CONF_THRESH = 0.25

print('Loading YOLO model...')
model = YOLO(MODEL_PATH)

try:
    model.names = {0: 'bacteria'}
except Exception:
    pass

print('Model loaded. Opening webcam...')

cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)

if not cap.isOpened():
    print(f"Cannot open camera index {CAM_INDEX}. Try changing CAM_INDEX in the script.")
    raise SystemExit(1)

prev_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to grab frame')
        break

    frame_count += 1
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    results = model.predict(source=img, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
    r = results[0]

    if hasattr(r, 'boxes') and len(r.boxes) > 0:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = model.names[int(cls)] if hasattr(model, 'names') else str(int(cls))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    fps = 1.0 / (time.time() - prev_time) if frame_count > 1 else 0.0
    prev_time = time.time()
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('YOLOv8 Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Done')
