import cv2
from inference_sdk import InferenceHTTPClient
from face import FaceRecognition

# ---------------- Roboflow Client ----------------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="6PWvEvjPfoJxLbMy8Csh"
)

# ---------------- Face Recognition ----------------
face_recog = FaceRecognition(known_faces_dir="known_faces")

# ---------------- Load Image ----------------
image_path = r"uploads/test1.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print("❌ Could not load image. Check the path:", image_path)
    exit()

# ---------------- Ensure 8-bit RGB ----------------
if frame.dtype != 'uint8':
    frame = (frame * 255).astype('uint8')
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ---------------- Run Roboflow model ----------------
try:
    result = CLIENT.infer(image_path, model_id="id-fqivr-kkbzf/1")  # your single model
    print("✅ Roboflow detection done.")
except Exception as e:
    print("❌ Roboflow error:", e)
    result = None

# ---------------- Detect Faces ----------------
faces = face_recog.recognize(frame_rgb)

# ---------------- Draw Roboflow Detections ----------------
if result:
    for pred in result.get("predictions", []):
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        label = pred.get("class", "Object")
        conf = pred.get("confidence", 0) * 100
        # ID = yellow, Shoe = cyan
        color = (255, 255, 0) if "id" in label.lower() else (0, 255, 255)
        cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 2)
        cv2.putText(frame, f"{label} ({conf:.1f}%)", (x - w//2, y - h//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ---------------- Draw Face Detections ----------------
for (top, right, bottom, left, name) in faces:
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ---------------- Save and Show ----------------
cv2.imwrite("output.jpg", frame)
print("✅ Output saved as output.jpg")

cv2.imshow("Detection Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
