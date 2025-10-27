# import os
# from flask import Flask, request, render_template, send_from_directory
# import cv2
# from inference_sdk import InferenceHTTPClient
# from face import FaceRecognition
# import numpy as np

# # -------------------- Flask Setup --------------------
# app = Flask(__name__)
# UPLOAD_FOLDER = "uploads"
# OUTPUT_FOLDER = "output"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # -------------------- Roboflow Setup --------------------
# CLIENT = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key="6PWvEvjPfoJxLbMy8Csh"
# )

# # -------------------- Face Recognition Setup --------------------
# face_recog = FaceRecognition(known_faces_dir="known_faces")

# # -------------------- Routes --------------------
# @app.route("/", methods=["GET", "POST"])
# def index():
#     uploaded_image = None
#     result_image = None

#     if request.method == "POST":
#         file = request.files.get("image")
#         if file:
#             # Save uploaded image
#             filepath = os.path.join(UPLOAD_FOLDER, file.filename)
#             file.save(filepath)
#             uploaded_image = file.filename

#             # Read image
#             frame = cv2.imread(filepath)
#             if frame is None:
#                 return "❌ Could not read uploaded image."

#             # -------- Convert to 8-bit RGB for face_recognition --------
#             if frame.dtype != 'uint8':
#                 frame = (frame / np.max(frame) * 255).astype('uint8')

#             # Convert image to RGB
#             if len(frame.shape) == 2:  # grayscale -> RGB
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#             elif frame.shape[2] == 4:  # RGBA -> RGB
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
#             else:  # BGR -> RGB
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # -------- Roboflow detection --------
#             try:
#                 result = CLIENT.infer(filepath, model_id="id-fqivr-kkbzf/1")
#             except Exception as e:
#                 result = None
#                 print("❌ Roboflow error:", e)

#             # -------- Face detection --------
#             try:
#                 faces = face_recog.recognize(frame_rgb)
#             except Exception as e:
#                 faces = []
#                 print("⚠️ Error during face recognition:", e)

#             # -------- Draw Roboflow detections with larger font --------
#             if result:
#                 for pred in result.get("predictions", []):
#                     x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
#                     label = pred.get("class", "Object")
#                     conf = pred.get("confidence", 0) * 100
#                     color = (255, 255, 0) if "id" in label.lower() else (0, 255, 255)
#                     cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 3)
#                     cv2.putText(frame, f"{label} ({conf:.1f}%)", (x - w//2, y - h//2 - 15),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

#             # -------- Draw face detections with larger font --------
#             for (top, right, bottom, left, name) in faces:
#                 color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#                 cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
#                 cv2.putText(frame, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

#             # Save result
#             output_path = os.path.join(OUTPUT_FOLDER, file.filename)
#             cv2.imwrite(output_path, frame)
#             result_image = file.filename

#     return render_template("index.html", uploaded_image=uploaded_image, result_image=result_image)

# @app.route("/uploads/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

# @app.route("/output/<filename>")
# def output_file(filename):
#     return send_from_directory(OUTPUT_FOLDER, filename)

# if __name__ == "__main__":
#     app.run(debug=True)


import os
from flask import Flask, request, render_template, send_from_directory
import cv2
from inference_sdk import InferenceHTTPClient
from face import FaceRecognition
import numpy as np

# -------------------- Flask Setup --------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------- Roboflow Setup --------------------
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="6PWvEvjPfoJxLbMy8Csh"
)

# -------------------- Face Recognition Setup --------------------
face_recog = FaceRecognition(known_faces_dir="known_faces")

# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_image = None
    result_image = None
    detection_message = None  # ✅ added variable

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            uploaded_image = file.filename

            frame = cv2.imread(filepath)
            if frame is None:
                return "❌ Could not read uploaded image."

            # -------- Convert to 8-bit RGB for face_recognition --------
            if frame.dtype != 'uint8':
                frame = (frame / np.max(frame) * 255).astype('uint8')

            if len(frame.shape) == 2:  # grayscale -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:  # RGBA -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            else:  # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # -------- Roboflow detection --------
            try:
                result = CLIENT.infer(filepath, model_id="id-fqivr-kkbzf/1")
                rf_predictions = result.get("predictions", [])
            except Exception as e:
                rf_predictions = []
                print("❌ Roboflow error:", e)

            # -------- Face detection --------
            try:
                faces = face_recog.recognize(frame_rgb)
            except Exception as e:
                faces = []
                print("⚠️ Error during face recognition:", e)

            # ✅ Check for detections
            # if rf_predictions or faces:
            #     detection_message = "✅ Object(s) detected."
            # else:
            #     detection_message = "⚠ No objects detected."
            # ✅ Check for detections more accurately
            detected_classes = [pred.get("class", "").lower() for pred in rf_predictions]
            face_detected = len(faces) > 0

            id_detected = any("id" in cls for cls in detected_classes)
            shoe_detected = any("shoe" in cls for cls in detected_classes)

            # Logic for message
            if id_detected and shoe_detected:
                detection_message = "✅ ID and Shoe detected."
            elif id_detected:
                detection_message = "⚠ Only ID detected. Shoe not found."
            elif shoe_detected:
                detection_message = "⚠ Only Shoe detected. ID not found."
            elif face_detected:
                detection_message = "⚠ Only Face detected. No ID or Shoe found."
            else:
                detection_message = "❌ No objects detected."


            # -------- Draw Roboflow detections --------
            for pred in rf_predictions:
                x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
                label = pred.get("class", "Object")
                conf = pred.get("confidence", 0) * 100
                color = (255, 255, 0) if "id" in label.lower() else (0, 255, 255)
                cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), color, 3)
                cv2.putText(frame, f"{label} ({conf:.1f}%)", (x - w//2, y - h//2 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            # -------- Draw face detections --------
            for (top, right, bottom, left, name) in faces:
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                cv2.putText(frame, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

            # Save result
            output_path = os.path.join(OUTPUT_FOLDER, file.filename)
            cv2.imwrite(output_path, frame)
            result_image = file.filename

    return render_template("index.html", uploaded_image=uploaded_image, result_image=result_image,
                           detection_message=detection_message)  # ✅ send message to HTML


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
