# import os
# import cv2
# import numpy as np
# import faiss
# from insightface.app import FaceAnalysis

# # Paths
# DATASET_FOLDER = r"C:\Users\GMRIT\Downloads\Face csec\Face csec\Face csec\Facelive\test_images"  # your enrollment folder
# FAISS_PATH = r"C:\Users\GMRIT\Downloads\Face csec\Face csec\Face csec\Facelive\database\test1.faiss"
# MAPPING_PATH = r"C:\Users\GMRIT\Downloads\Face csec\Face csec\Face csec\Facelive\database\embedding_id_to_student.npy"

# # Initialize FaceAnalysis
# # face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for GPU, -1 for CPU fallback
# # face_app.prepare(ctx_id=-1, det_size=(640, 640))

# # Lists to store embeddings and student IDs
# embeddings = []
# student_ids = []

# # Loop through students
# for student_folder in sorted(os.listdir(DATASET_FOLDER)):
#     folder_path = os.path.join(DATASET_FOLDER, student_folder)
#     if not os.path.isdir(folder_path):
#         continue
#     for img_file in sorted(os.listdir(folder_path)):
#         img_path = os.path.join(folder_path, img_file)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#         # Get embedding
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         faces = face_app.get(img_rgb)
#         if len(faces) == 0:
#             continue
#         face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
#         embedding = face.normed_embedding.astype(np.float32)
#         embeddings.append(embedding)
#         student_ids.append(student_folder)  # folder name = JNTU number

# # Convert to numpy arrays
# embeddings = np.array(embeddings)
# student_ids = np.array(student_ids)

# # Build FAISS index
# embedding_dim = embeddings.shape[1]
# index = faiss.IndexFlatL2(embedding_dim)
# index.add(embeddings)

# # Save FAISS and mapping
# faiss.write_index(index, FAISS_PATH)
# np.save(MAPPING_PATH, student_ids)

# print("✅ FAISS index saved to:", FAISS_PATH)
# print("✅ Mapping saved to:", MAPPING_PATH)
import os
import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

# Paths
DATASET_FOLDER = r"C:\Users\GMRIT\Downloads\Face csec\Face csec\Face csec\Facelive\test_images"
FAISS_PATH = r"C:\Users\GMRIT\Downloads\Face csec\Face csec\Face csec\Facelive\database\test1.faiss"
MAPPING_PATH = r"C:\Users\GMRIT\Downloads\Face csec\Face csec\Face csec\Facelive\database\embedding_id_to_student.npy"

# Initialize FaceAnalysis with GPU if available
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
face_app = FaceAnalysis(name='buffalo_l', providers=providers)
face_app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for GPU

# Lists to store embeddings and student IDs
embeddings = []
student_ids = []

# Count total images for progress
total_images = sum(
    len([f for f in os.listdir(os.path.join(DATASET_FOLDER, folder)) if os.path.isfile(os.path.join(DATASET_FOLDER, folder, f))])
    for folder in os.listdir(DATASET_FOLDER)
    if os.path.isdir(os.path.join(DATASET_FOLDER, folder))
)
count = 0

# Loop through student folders
for student_folder in sorted(os.listdir(DATASET_FOLDER)):
    folder_path = os.path.join(DATASET_FOLDER, student_folder)
    if not os.path.isdir(folder_path):
        continue

    for img_file in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Convert to RGB and get face embeddings
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(img_rgb)
        if len(faces) == 0:
            continue

        # Take the largest face
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        embedding = face.normed_embedding.astype(np.float32)
        embeddings.append(embedding)
        student_ids.append(student_folder)

        # Progress
        count += 1
        print(f"Processed {count}/{total_images} images")

# Convert lists to numpy arrays
embeddings = np.array(embeddings)
student_ids = np.array(student_ids)

# Build FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# Save FAISS index and mapping
faiss.write_index(index, FAISS_PATH)
np.save(MAPPING_PATH, student_ids)

