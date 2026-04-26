import cv2
import os
import time
import random
import json
from threading import Thread, Lock
from ultralytics import YOLO
from flask import Flask, Response, jsonify
from flask_cors import CORS

# -------------------- Flask App --------------------
app = Flask(__name__)
CORS(app)

# -------------------- Load YOLO --------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, 'weights', 'best.pt')

if not os.path.exists(weights_path):
    weights_path = os.path.join(script_dir, '..', 'runs', 'train_stage3', 'stage2_surface', 'weights', 'best.pt')

if not os.path.exists(weights_path):
    raise FileNotFoundError(
        f"Model weights not found. Place best.pt in {os.path.join(script_dir, 'weights')} "
        "or update the path in webcam_bact_detection.py"
    )

model = YOLO(weights_path)

# -------------------- Load Bacteria DB --------------------
with open(os.path.join(script_dir, 'bacteria_db.json'), 'r', encoding='utf-8') as f:
    BACTERIA_DB = json.load(f)

# -------------------- Camera --------------------
WEBCAM_INDEX = 0
MOBILE_CAM_URL = 'http://10.191.116.40:4747/video'


def get_camera():
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print('⚠ Webcam not found. Switching to mobile cam...')
        cap = cv2.VideoCapture(MOBILE_CAM_URL)
        if not cap.isOpened():
            raise Exception('❌ Camera not accessible')
    return cap


cap = get_camera()
print('✅ Camera Connected')

# -------------------- Shared Memory --------------------
latest_detected = []
latest_boxes = []
data_lock = Lock()

ALL_BACTERIA = list(BACTERIA_DB.keys())


# -------------------- Species Classifier --------------------
def bacteria_species_classifier(cropped_img):
    return random.choice(ALL_BACTERIA)


# -------------------- BACKGROUND YOLO LOOP --------------------
def detection_loop():
    global cap, latest_detected, latest_boxes

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb, imgsz=640, conf=0.25, verbose=False)

        detected = []
        boxes_for_draw = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                name = bacteria_species_classifier(crop)
                info = BACTERIA_DB.get(name, {})

                detected.append({
                    'name': name,
                    'type': info.get('type', 'Unknown'),
                    'family': info.get('family', ''),
                    'transmission': info.get('transmission', ''),
                    'diseases': info.get('diseases', ''),
                    'solution': info.get('solution', ''),
                    'harmful_percent': info.get('harmful_percent', 0)
                })

                boxes_for_draw.append({
                    'box': (x1, y1, x2, y2),
                    'label': f"{name} {conf:.2f}",
                    'type': info.get('type', 'Unknown')
                })

        with data_lock:
            latest_detected = detected
            latest_boxes = boxes_for_draw

        time.sleep(0.3)


# -------------------- VIDEO STREAM (DRAW BOXES HERE) --------------------
def generate_frames():
    global cap

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        with data_lock:
            draw_boxes = list(latest_boxes)

        for item in draw_boxes:
            x1, y1, x2, y2 = item['box']
            label = item['label']
            tag = item['type']

            color = (0, 255, 0) if tag == 'Good' else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )


# -------------------- API --------------------
@app.route('/analysis')
def analysis():
    with data_lock:
        return jsonify(latest_detected)


@app.route('/')
def index():
    return '''
    <h2>Bacteria Detection Stream</h2>
    <img src="/video_feed" width="720"/>
    '''


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# -------------------- RUN --------------------
if __name__ == '__main__':
    Thread(target=detection_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)
