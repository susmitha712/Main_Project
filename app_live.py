# live_recognition.py

import faiss
import sqlite3
import cv2
import utils             # expects utils.py in same folder (provides get_all_live_faces)
import numpy as np
import time
import os
import sys

# --------- Paths (change if needed) ----------
FAISS_INDEX_PATH = "database/test.faiss"
DB_PATH = "database/student.db"

# ---------- Load FAISS index and DB ----------
try:
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DB_PATH):
        raise FileNotFoundError("FAISS index or DB file not found.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    print("✅ Database and FAISS index loaded successfully.")
except Exception as e:
    print(f"❌ Error loading database or FAISS index: {e}")
    print("Please run the enrollment script (e.g. augment.py / enroll script) first to create those files.")
    sys.exit(1)


def face_quality(frame, bbox):
    """
    Estimate face quality using blur detection and size check.
    Returns a quality score between 0.0 and 1.0.
    """
    x1, y1, x2, y2 = bbox
    # clamp bbox to frame
    h_frame, w_frame = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return 0.0

    # Blur detection (variance of Laplacian)
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Normalize blur score (0–1 scale; 100 is an approximate "good" threshold)
    blur_norm = min(1.0, blur_score / 100.0)

    # Size check (relative to frame size) — prefer faces that take decent area of frame
    face_area = (x2 - x1) * (y2 - y1)
    size_norm = min(1.0, face_area / (w_frame * h_frame * 0.2))  # expect at least 20% area for good score

    quality = 0.7 * blur_norm + 0.3 * size_norm
    return float(np.clip(quality, 0.0, 1.0))


def recognize_face(embedding, quality):
    """
    Searches the FAISS index for the closest match using inner-product (works if embeddings were normalized).
    Uses adaptive thresholding based on face quality.
    Returns (name, similarity, threshold)
    """
    if index.ntotal == 0:
        return "Unknown", 0.0, 0.0

    embedding = embedding.reshape(1, -1).astype(np.float32)

    try:
        similarities, indices = index.search(embedding, 1)
    except Exception as e:
        # If search fails for some reason, return Unknown
        return "Unknown", 0.0, 0.0

    # similarity and idx
    similarity = float(similarities[0][0])
    faiss_id = int(indices[0][0])

    # If invalid index returned
    if faiss_id < 0 or faiss_id >= index.ntotal:
        return "Unknown", similarity, 0.0

    # Adaptive threshold (tweak base_threshold to your operating point)
    base_threshold = 0.40  # with normalized embeddings and IndexFlatIP this is cosine-like
    if quality < 0.4:
        threshold = base_threshold + 0.05
    elif quality > 0.8:
        threshold = base_threshold - 0.05
    else:
        threshold = base_threshold

    if similarity >= threshold:
        cursor.execute("SELECT name FROM students WHERE embedding_id = ?", (faiss_id,))
        row = cursor.fetchone()
        if row:
            return row[0], similarity, threshold

    return "Unknown", similarity, threshold


def start_recognition(camera_idx=0):
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam. Check camera index or permissions.")
        return

    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("⚠ Warning: empty frame from camera. Exiting.")
                break

            # Detect all faces and embeddings
            detected_faces = utils.get_all_live_faces(frame)

            # draw results
            for face in detected_faces:
                embedding = face.get("embedding")
                bbox = face.get("bbox").tolist() if hasattr(face.get("bbox"), "tolist") else face.get("bbox")

                # ensure ints
                bbox = [int(v) for v in bbox]

                # quality and recognition
                quality = face_quality(frame, bbox)
                name, similarity, threshold = recognize_face(embedding, quality)

                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                text = f"{name} ({similarity:.2f}) Q={quality:.2f} T={threshold:.2f}"
                cv2.putText(frame, text, (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # FPS
            fps = 1.0 / max(1e-6, (time.time() - start_time))
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("Live Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        conn.close()


if __name__ == "__main__":
    # optionally accept camera index from CLI: python live_recognition.py 0
    cam_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_recognition(cam_idx)
