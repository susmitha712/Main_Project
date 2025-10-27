# from flask import Flask, request, render_template, send_from_directory,jsonify
# from flask import redirect, url_for
# import os
# import cv2
# import numpy as np
# import faiss
# import sqlite3
# from insightface.app import FaceAnalysis
# from inference_sdk import InferenceHTTPClient  # Roboflow client
# from roboflow import Roboflow


# app = Flask(__name__)

# # -------------------- Paths --------------------
# UPLOAD_FOLDER = "uploads"
# OUTPUT_FOLDER = "output"
# DB_PATH = r"Facelive/database/student.db"
# FAISS_INDEX_PATH = r"C:\Users\Sushmitha\OneDrive\Desktop\Face csec3\Face csec\Facelive\database\test1.faiss"
# FAISS_MAPPING_PATH = r"C:\Users\Sushmitha\OneDrive\Desktop\Face csec3\Face csec\Facelive\database\embedding_id_to_student.npy"  # mapping array

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # -------------------- Initialize models --------------------
# # face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# # face_app.prepare(ctx_id=-1, det_size=(640, 640))
# face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 → first GPU

# faiss_index = faiss.read_index(FAISS_INDEX_PATH)
# embedding_to_student = np.load(FAISS_MAPPING_PATH)  # array mapping FAISS index → student ID

# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cursor = conn.cursor()

# # Roboflow client for object detection
# CLIENT = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key="6PWvEvjPfoJxLbMy8Csh")

# # -------------------- Helper functions --------------------
# def get_face_embedding_from_image(img):
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     faces = face_app.get(img_rgb)
#     if len(faces) == 0:
#         return None
#     face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
#     return face.normed_embedding.astype(np.float32)

# # def find_student(embedding):
# #     if embedding is None:
# #         return "Unknown"
# #     embedding = embedding.reshape(1, -1)
# #     D, I = faiss_index.search(embedding, k=1)
# #     nearest_index = int(I[0][0])
# #     try:
# #         student_id = embedding_to_student[nearest_index]
# #         return student_id
# #     except IndexError:
# #         return "Unknown"

# def find_student(embedding, threshold=0.8):
#     """
#     embedding: np.array of shape (512,) or (1, 512)
#     threshold: maximum allowed distance for a valid match
#     """
#     if embedding is None:
#         return "Unknown"
    
#     embedding = embedding.reshape(1, -1)
#     D, I = faiss_index.search(embedding, k=1)
    
#     nearest_index = int(I[0][0])
#     distance = float(D[0][0])
    
#     if distance > threshold:
#         return "Unknown"
    
#     try:
#         student_id = embedding_to_student[nearest_index]
#         return student_id
#     except IndexError:
#         return "Unknown"

# # -------------------- Serve uploaded/output images --------------------
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

# @app.route('/output/<filename>')
# def output_file(filename):
#     return send_from_directory(OUTPUT_FOLDER, filename)

# # -------------------- Common detection logic --------------------
# # def process_image(img, filename):
# #     # Save temporary input
# #     temp_path = os.path.join(UPLOAD_FOLDER, filename)
# #     cv2.imwrite(temp_path, img)

# #     # -------- Object detection (Roboflow) --------
# #     try:
# #         result = CLIENT.infer(temp_path, model_id="id-fqivr-kkbzf/1")
# #         rf_predictions = [p for p in result.get("predictions", []) if p.get("confidence", 0) > 0.5]
# #     except Exception as e:
# #         rf_predictions = []
# #         print("❌ Roboflow error:", e)

# #     # Determine detected objects
# #     detected_classes = [pred.get("class", "").lower() for pred in rf_predictions]
# #     id_detected = any("id" in cls for cls in detected_classes)
# #     shoe_detected = any("shoe" in cls for cls in detected_classes)

# #     # -------- Face recognition (always run to get student ID) --------
# #     embedding = get_face_embedding_from_image(img)
# #     student_id = find_student(embedding)

# #     # -------- Draw object detection boxes --------
# #     for pred in rf_predictions:
# #         x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
# #         label = pred.get("class", "Object")
# #         conf = pred.get("confidence", 0) * 100
# #         color = (255, 255, 0) if "id" in label.lower() else (0, 255, 255)
# #         cv2.rectangle(img, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 3)
# #         cv2.putText(img, f"{label} ({conf:.1f}%)", (x - w//2, y - h//2 - 15),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

# #     # Draw face label
# #     if student_id and student_id != "Unknown":
# #         cv2.putText(img, f"Face: {student_id}", (10, 30),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

# #     # Save output image
# #     output_path = os.path.join(OUTPUT_FOLDER, filename)
# #     cv2.imwrite(output_path, img)

# #     # -------- Detection message --------
# #     if id_detected and shoe_detected:
# #         detection_message = f"✅ ID and Shoe detected. Face → {student_id}"
# #     elif id_detected:
# #         detection_message = f"⚠ Only ID detected. Face → {student_id}" if student_id != "Unknown" else "⚠ Only ID detected. Face not recognized / Unknown person"
# #     elif shoe_detected:
# #         detection_message = f"⚠ Only Shoe detected. Face → {student_id}" if student_id != "Unknown" else "⚠ Only Shoe detected. Face not recognized / Unknown person"
# #     else:
# #         detection_message = f"❌ No objects detected. Face → {student_id}"

# #     return detection_message


# def process_image(img, filename):
#     temp_path = os.path.join(UPLOAD_FOLDER, filename)
#     cv2.imwrite(temp_path, img)

#     try:
#         result = CLIENT.infer(temp_path, model_id="id-fqivr-kkbzf/1")
#         rf_predictions = result.get("predictions", [])
#     except Exception as e:
#         print("❌ Roboflow error:", e)
#         rf_predictions = []

#     img_h, img_w = img.shape[:2]
#     allowed_classes = ["id", "id card", "id_card", "shoe", "shoes"]
#     filtered_predictions = []

#     for p in rf_predictions:
#         label = p.get("class", "").strip().lower()
#         conf = float(p.get("confidence", 0))
#         box_area = p["width"] * p["height"]
#         frame_area = img_w * img_h

#         # Ignore irrelevant labels
#         if label not in allowed_classes:
#             continue

#         # Ignore walls or backgrounds
#         if "wall" in label or "background" in label:
#             continue

#         # Ignore weird box sizes
#         if box_area > 0.6 * frame_area or box_area < 0.002 * frame_area:
#             continue

#         # Apply confidence threshold
#         if conf < 0.55:
#             continue

#         filtered_predictions.append(p)

#     # Now determine what was actually detected
#     id_detected = any(p["class"].lower() in ["id", "id card", "id_card"] for p in filtered_predictions)
#     shoe_detected = any("shoe" in p["class"].lower() for p in filtered_predictions)

#     # Draw boxes for remaining predictions
#     for p in filtered_predictions:
#         x, y, w, h = int(p["x"]), int(p["y"]), int(p["width"]), int(p["height"])
#         label = p["class"]
#         conf = p["confidence"] * 100
#         color = (255, 255, 0) if "id" in label.lower() else (0, 255, 255)
#         cv2.rectangle(img, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 3)
#         cv2.putText(img, f"{label} ({conf:.1f}%)", (x - w//2, y - h//2 - 15),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

#     # Face recognition
#     embedding = get_face_embedding_from_image(img)
#     student_id = find_student(embedding)
#     if student_id and student_id != "Unknown":
#         cv2.putText(img, f"Face: {student_id}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

#     output_path = os.path.join(OUTPUT_FOLDER, filename)
#     cv2.imwrite(output_path, img)

#     # Message logic
#     if id_detected and shoe_detected:
#         detection_message = f"✅ ID Card and Shoe detected. Face → {student_id}"
#     elif id_detected:
#         detection_message = f"⚠ Only ID Card detected. Face → {student_id}" if student_id != "Unknown" else "⚠ Only ID Card detected. Unknown Face"
#     elif shoe_detected:
#         detection_message = f"⚠ Only Shoe detected. Face → {student_id}" if student_id != "Unknown" else "⚠ Only Shoe detected. Unknown Face"
#     else:
#         detection_message = f"❌ No ID Card or Shoe detected. Face → {student_id}"

#     return detection_message



# # -------------------- Upload image route --------------------
# @app.route('/', methods=['GET', 'POST'])
# def upload_image():
#     if request.method == 'POST':
#         file = request.files.get('image')
#         if file:
#             uploaded_path = os.path.join(UPLOAD_FOLDER, file.filename)
#             file.save(uploaded_path)
#             img = cv2.imread(uploaded_path)
#             if img is None:
#                 return "Failed to read uploaded image"

#             detection_message = process_image(img, file.filename)
#             return render_template('index.html',
#                                    detection_message=detection_message,
#                                    uploaded_image=file.filename,
#                                    result_image=file.filename)
#     return render_template('index.html')



# # Camera Capture Route (display results on another page)
# @app.route('/camera', methods=['GET', 'POST'])
# def camera_capture():
#     if request.method == 'POST':
#         file = request.files.get('image')
#         if not file:
#             return "No image uploaded"

#         # Convert uploaded file to OpenCV image
#         img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

#         # Process image
#         detection_message = process_image(img, "camera_capture.png")

#         # Redirect to result page
#         return redirect(url_for('camera_result',
#                                 uploaded="camera_capture.png",
#                                 result="camera_capture.png",
#                                 message=detection_message))

#     # GET request → show camera page
#     return render_template('camera.html')

# # Result page for camera capture
# @app.route('/camera_result')
# def camera_result():
#     uploaded = request.args.get('uploaded')
#     result = request.args.get('result')
#     message = request.args.get('message')
#     return render_template('camera_result.html',
#                            uploaded_image=uploaded,
#                            result_image=result,
#                            detection_message=message)

# # -------------------- Run app --------------------
# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, render_template, send_from_directory
import os
import cv2
import numpy as np
import faiss
import sqlite3
from insightface.app import FaceAnalysis
from inference_sdk import InferenceHTTPClient  # Roboflow client
from roboflow import Roboflow

app = Flask(__name__)

# -------------------- Paths --------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
DB_PATH = r"Facelive/database/student.db"
FAISS_INDEX_PATH = r"C:\Users\Sushmitha\OneDrive\Desktop\Face csec3\Face csec\Facelive\database\test1.faiss"
FAISS_MAPPING_PATH = r"C:\Users\Sushmitha\OneDrive\Desktop\Face csec3\Face csec\Facelive\database\embedding_id_to_student.npy"  # mapping array


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------- Initialize models --------------------
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 → first GPU

faiss_index = faiss.read_index(FAISS_INDEX_PATH)
embedding_to_student = np.load(FAISS_MAPPING_PATH)  # mapping FAISS index → student ID

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Roboflow client for object detection
CLIENT = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key="6PWvEvjPfoJxLbMy8Csh")

# -------------------- Helper functions --------------------
def get_face_embedding_from_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img_rgb)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    return face.normed_embedding.astype(np.float32)


# ✅ UPDATED FUNCTION
def find_student(embedding):
    if embedding is None:
        return "Unknown Person"

    embedding = embedding.reshape(1, -1)
    D, I = faiss_index.search(embedding, k=1)
    distance = D[0][0]  # smaller = more similar
    nearest_index = int(I[0][0])

    # ---- Threshold to decide known vs unknown ----
    threshold = 1.0  # Adjust if needed (try 0.9–1.2 range)
    print(f"[DEBUG] Match distance: {distance:.4f}")  # Optional: helps tune threshold

    if distance > threshold:
        return "Unknown Person"

    try:
        student_id = embedding_to_student[nearest_index]
        return student_id
    except IndexError:
        return "Unknown Person"


# -------------------- Serve uploaded/output images --------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# -------------------- Common detection logic --------------------
def process_image(img, filename):
    # Save temporary input
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(temp_path, img)

    # -------- Object detection (Roboflow) --------
    try:
        result = CLIENT.infer(temp_path, model_id="id-fqivr-kkbzf/1")
        rf_predictions = [p for p in result.get("predictions", []) if p.get("confidence", 0) > 0.5]
    except Exception as e:
        rf_predictions = []
        print("❌ Roboflow error:", e)

    # Determine detected objects
    detected_classes = [pred.get("class", "").lower() for pred in rf_predictions]
    id_detected = any("id" in cls for cls in detected_classes)
    shoe_detected = any("shoe" in cls for cls in detected_classes)

    # -------- Face recognition --------
    embedding = get_face_embedding_from_image(img)
    student_id = find_student(embedding)

    # -------- Draw object detection boxes --------
    for pred in rf_predictions:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        label = pred.get("class", "Object")
        conf = pred.get("confidence", 0) * 100
        color = (255, 255, 0) if "id" in label.lower() else (0, 255, 255)
        cv2.rectangle(img, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 3)
        cv2.putText(img, f"{label} ({conf:.1f}%)", (x - w//2, y - h//2 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    # Draw face label
    label_color = (0, 255, 0) if student_id != "Unknown Person" else (0, 0, 255)
    cv2.putText(img, f"Face: {student_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 3)

    # Save output image
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, img)

    # -------- Detection message --------
    if id_detected and shoe_detected:
        detection_message = f"✅ ID and Shoe detected. "
    elif id_detected:
        detection_message = f"⚠ Only ID detected. Face → {student_id}"
    elif shoe_detected:
        detection_message = f"⚠ Only Shoe detected. Face → {student_id}"
    else:
        detection_message = f"❌ No objects detected. Face → {student_id}"

    return detection_message


# -------------------- Upload image route --------------------
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            uploaded_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(uploaded_path)
            img = cv2.imread(uploaded_path)
            if img is None:
                return "Failed to read uploaded image"

            detection_message = process_image(img, file.filename)
            return render_template('index.html',
                                   detection_message=detection_message,
                                   uploaded_image=file.filename,
                                   result_image=file.filename)
    return render_template('index.html')


# -------------------- Camera capture route --------------------
@app.route('/camera')
def camera_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Cannot open camera"

    # Warm up the camera
    for _ in range(10):
        ret, frame = cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return "Failed to capture image from camera"

    # Save captured image
    filename = "captured_image.png"
    captured_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(captured_path, frame)

    # Call same detection logic
    detection_message = process_image(frame, filename)

    return render_template('index.html',
                           detection_message=detection_message,
                           uploaded_image=filename,
                           result_image=filename)


# -------------------- Run app --------------------
if __name__ == "__main__":
    app.run(debug=True)