# import os
# import faiss
# import sqlite3
# import numpy as np
# import cv2
# from insightface.app import FaceAnalysis

# # ------------------------------
# # Paths Setup
# # ------------------------------
# DATA_DIR = "C:\\Users\\jamip\\OneDrive\\Desktop\\Facelive\\images1"
#              # Folder where student image folders are stored
# DB_PATH = 'database/student.db'
# FAISS_INDEX_PATH = 'database/test.faiss'

# # Choose which student folder to enroll
# # TARGET_STUDENT = "22341A05J5"        # <-- Change this folder name to the one you want to enroll
# # TARGET_STUDENT = "22341A05J5"   
# # ------------------------------
# # Initialize ArcFace (InsightFace)
# # ------------------------------
# app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU-only mode

# # ------------------------------
# # Extract Face Embedding
# # ------------------------------
# def get_face_embedding(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"❌ Failed to read image: {image_path}")
#         return None

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     faces = app.get(img)

#     # Retry detection with higher resolution if none found
#     if len(faces) == 0:
#         img_resized = cv2.resize(img, (800, 800))
#         faces = app.get(img_resized)
#         if len(faces) == 0:
#             print(f"⚠ No face detected in {image_path}")
#             return None

#     # Use the largest detected face
#     face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
#     embedding = face.normed_embedding.astype(np.float32)
#     return embedding

# # ------------------------------
# # Database Setup
# # ------------------------------
# def create_db():
#     os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS students (
#             embedding_id INTEGER PRIMARY KEY,
#             student_id TEXT UNIQUE NOT NULL,
#             name TEXT
#         )
#     ''')
#     conn.commit()
#     conn.close()

# # ------------------------------
# # Enrollment Function (Single Student)
# # ------------------------------
# def enroll_single_student():
#     create_db()
#     os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

#     d = 512  # ArcFace embedding dimension
#     index = faiss.IndexFlatIP(d)

#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.cursor()

#     # Locate the student folder
#     student_path = os.path.join(DATA_DIR, TARGET_STUDENT)

#     if not os.path.exists(student_path):
#         print(f"❌ Folder '{TARGET_STUDENT}' not found inside '{DATA_DIR}'.")
#         conn.close()
#         return

#     student_id = TARGET_STUDENT
#     name = student_id.replace('_', ' ').title()

#     print(f"\n📸 Processing student: {name}")
#     student_embeddings = []

#     for filename in os.listdir(student_path):
#         if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#             image_path = os.path.join(student_path, filename)
#             embedding = get_face_embedding(image_path)
#             if embedding is not None:
#                 student_embeddings.append(embedding)

#     if student_embeddings:
#         mean_embedding = np.mean(student_embeddings, axis=0)
#         mean_embedding /= np.linalg.norm(mean_embedding)

#         mean_embedding = mean_embedding.reshape(1, -1)
#         index.add(mean_embedding)

#         embedding_id = index.ntotal - 1
#         cursor.execute("INSERT OR REPLACE INTO students (embedding_id, student_id, name) VALUES (?, ?, ?)",
#                        (int(embedding_id), student_id, name))
#         conn.commit()
#         print(f"✅ Enrolled {name} (FAISS ID: {embedding_id}) with {len(student_embeddings)} valid images.")
#     else:
#         print(f"⚠ No valid embeddings found for {name}. Enrollment skipped.")

#     # ------------------------------
#     # Save FAISS and Close
#     # ------------------------------
#     try:
#         faiss.write_index(index, FAISS_INDEX_PATH)
#         print(f"\n🎯 FAISS index created successfully with {index.ntotal} student(s). Saved to '{FAISS_INDEX_PATH}'.")
#     except Exception as e:
#         print(f"⚠ Warning: Enrollment complete but FAISS save failed: {e}")

#     conn.close()

# # ------------------------------
# # Main Entry
# # ------------------------------
# if __name__ == "__main__":
#     enroll_single_student()




import os
import faiss
import sqlite3
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# ------------------------------
# Paths Setup
# ------------------------------
DATA_DIR = r"C:\Users\ANUSHA\Downloads\Face csec1\Face csec\Facelive\test_images"  # folder containing student subfolders
DB_PATH = 'database/student.db'
FAISS_INDEX_PATH = 'database/test.faiss'

# ------------------------------
# Initialize ArcFace (InsightFace)
# ------------------------------
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU-only mode

# ------------------------------
# Extract Face Embedding
# ------------------------------
def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to read image: {image_path}")
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img)

    if len(faces) == 0:
        img_resized = cv2.resize(img, (800, 800))
        faces = app.get(img_resized)
        if len(faces) == 0:
            print(f"⚠ No face detected in {image_path}")
            return None

    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
    embedding = face.normed_embedding.astype(np.float32)
    return embedding

# ------------------------------
# Database Setup
# ------------------------------
def create_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            embedding_id INTEGER PRIMARY KEY,
            student_id TEXT UNIQUE NOT NULL,
            name TEXT
        )
    ''')
    conn.commit()
    conn.close()

# ------------------------------
# Enrollment Function (Multiple Students)
# ------------------------------
def enroll_all_students():
    create_db()
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    d = 512  # ArcFace embedding dimension
    index = faiss.IndexFlatIP(d)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Iterate over all subfolders in DATA_DIR
    for student_folder in os.listdir(DATA_DIR):
        student_path = os.path.join(DATA_DIR, student_folder)
        if not os.path.isdir(student_path):
            continue

        student_id = student_folder
        name = student_id.replace('_', ' ').title()

        print(f"\n📸 Processing student: {name}")
        student_embeddings = []

        for filename in os.listdir(student_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(student_path, filename)
                embedding = get_face_embedding(image_path)
                if embedding is not None:
                    student_embeddings.append(embedding)

        if student_embeddings:
            mean_embedding = np.mean(student_embeddings, axis=0)
            mean_embedding /= np.linalg.norm(mean_embedding)
            mean_embedding = mean_embedding.reshape(1, -1)
            index.add(mean_embedding)

            embedding_id = index.ntotal - 1
            cursor.execute(
                "INSERT OR REPLACE INTO students (embedding_id, student_id, name) VALUES (?, ?, ?)",
                (int(embedding_id), student_id, name)
            )
            conn.commit()
            print(f"✅ Enrolled {name} (FAISS ID: {embedding_id}) with {len(student_embeddings)} valid images.")
        else:
            print(f"⚠ No valid embeddings found for {name}. Enrollment skipped.")

    # Save FAISS index
    try:
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"\n🎯 FAISS index created successfully with {index.ntotal} student(s). Saved to '{FAISS_INDEX_PATH}'.")
    except Exception as e:
        print(f"⚠ Warning: Enrollment complete but FAISS save failed: {e}")

    conn.close()

# ------------------------------
# Main Entry
# ------------------------------
if __name__ == "__main__":
    enroll_all_students()
