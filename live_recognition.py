import cv2
import os
from roboflow import Roboflow

# -------------------------
# 1️⃣ Initialize Roboflow ID detection
# -------------------------
rf = Roboflow(api_key="4XvTksb7csTKkWdqpMfy")
project = rf.workspace().project("id-detectionn-2")
model = project.version(1).model

# -------------------------
# 2️⃣ Load OpenCV face detector
# -------------------------
cascade_path = r"C:\Users\jamip\OneDrive\Desktop\Face csec\haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    print("Error: Haar cascade file not found!")
    exit()
face_cascade = cv2.CascadeClassifier(cascade_path)

# -------------------------
# 3️⃣ Initialize webcam
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting live ID & Face Detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    id_detected = False
    face_boxes = []

    # -------------------------
    # 4️⃣ Run ID detection
    # -------------------------
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model.predict(rgb_frame).json()

    for pred in result.get("predictions", []):
        if pred["class"] == "id_wearing" and pred["confidence"] > 0.7:
            id_detected = True
            x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
            # Draw ID box in green
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            break

    # -------------------------
    # 5️⃣ Run face detection
    # -------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_boxes.append((x, y, w, h))

    # -------------------------
    # 6️⃣ Draw face boxes
    # -------------------------
    for (x, y, w, h) in face_boxes:
        if id_detected:
            color = (0, 255, 0)  # Green if ID detected
        else:
            color = (0, 0, 255)  # Red if no ID detected
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, "Student ID: 22341A4214", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ------------------------- 
    # 7️⃣ Show alert if no ID and no face
    # -------------------------
    if not id_detected and len(face_boxes) == 0:
        cv2.putText(frame, "Student missing ID!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # -------------------------
    # 8️⃣ Show frame
    # -------------------------
    cv2.imshow("Live ID & Face Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------
# 9️⃣ Release resources
# -------------------------
cap.release()
cv2.destroyAllWindows()
