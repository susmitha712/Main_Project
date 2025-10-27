from flask import Flask, request, render_template, send_from_directory
import os
import cv2
import face_recognition
import numpy as np
import faiss
import sqlite3

# ---- Flask setup ----
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# ---- Load FAISS index ----
faiss_path = os.path.join("database", "test.faiss")
index = faiss.read_index(faiss_path)

# ---- Connect to student DB ----
conn = sqlite3.connect("student.db", check_same_thread=False)
cursor = conn.cursor()

# ---- Process uploaded image ----
def process_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not read uploaded image: {file_path}")

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        embedding = np.array([encoding], dtype=np.float32)
        distances, indices = index.search(embedding, k=1)
        student_id = int(indices[0][0])
        distance = float(distances[0][0])

        cursor.execute("SELECT name FROM students WHERE id=?", (student_id,))
        row = cursor.fetchone()
        name = row[0] if row else "Unknown"

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(img, f"{name} ({distance:.2f})", (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(file_path))
    cv2.imwrite(output_path, img)
    return output_path

# ---- Routes ----
@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_image = None
    output_image = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            uploaded_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(uploaded_path)
            uploaded_image = file.filename

            try:
                output_path = process_image(uploaded_path)
                output_image = os.path.basename(output_path)
            except Exception as e:
                error = f"⚠️ Error: {e}"

    return render_template("index.html",
                           uploaded_image=uploaded_image,
                           output_image=output_image,
                           error=error)

# ---- Serve uploaded/output images ----
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

# ---- Run Flask app ----
if __name__ == "__main__":
    app.run(debug=True)
