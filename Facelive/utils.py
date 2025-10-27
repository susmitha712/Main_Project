import numpy as np
import cv2
from insightface.app import FaceAnalysis

# ------------------------------
# Initialize ArcFace (InsightFace)
# ------------------------------
# Using the CPU-friendly "buffalo_l" model (same as enrollment)
# Works perfectly without GPU / CUDA
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 ensures CPU-only mode


def get_face_embedding(image_path):
    """
    Detects a face in an image, aligns it, and returns the L2-normalized ArcFace embedding.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return None

    # Convert to RGB (cv2 loads images in BGR format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run face detection
    faces = app.get(img)

    # Retry with larger resolution if no face is detected
    if len(faces) == 0:
        img_resized = cv2.resize(img, (800, 800))
        faces = app.get(img_resized)
        if len(faces) == 0:
            print(f"⚠ No face detected in {image_path}. Skipping.")
            return None

    # Select the largest face (for better accuracy)
    face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

    # Extract the L2-normalized embedding
    embedding = face.normed_embedding.astype(np.float32)
    return embedding


def bb_intersection_over_union(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Each box is [x1, y1, x2, y2].
    Returns a float between 0 and 1.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

    union = float(boxAArea + boxBArea - interArea)
    if union == 0:
        return 0.0

    return interArea / union


def get_all_live_faces(frame):
    """
    Detects all faces in a live camera frame and returns their embeddings and bounding boxes.
    """
    # The webcam frame is already BGR → no color conversion needed
    faces = app.get(frame)

    results = []
    for face in faces:
        embedding = face.normed_embedding.astype(np.float32)
        results.append({
            "bbox": face.bbox.astype(int),  # (x1, y1, x2, y2)
            "embedding": embedding
        })

    return results
