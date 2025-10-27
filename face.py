# face.py
import os
import cv2
import face_recognition
import numpy as np

class FaceRecognition:
    def __init__(self, known_faces_dir="known_faces"):
        """
        Load known faces and their encodings from the directory.
        Directory structure:
        known_faces/
            person1/
                img1.jpg
                img2.jpg
            person2/
                img1.jpg
        """
        self.known_encodings = []
        self.known_names = []

        if not os.path.exists(known_faces_dir):
            print(f"⚠️ Directory '{known_faces_dir}' not found")
            return

        for person_name in os.listdir(known_faces_dir):
            person_dir = os.path.join(known_faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            for filename in os.listdir(person_dir):
                path = os.path.join(person_dir, filename)
                image = cv2.imread(path)
                if image is None:
                    print(f"⚠️ Could not read image: {path}")
                    continue

                # Convert to 8-bit RGB
                image = self._convert_to_rgb(image)

                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_encodings.append(encodings[0])
                    self.known_names.append(person_name)
                else:
                    print(f"⚠️ No face found in {path}")

        print(f"✅ Loaded {len(self.known_encodings)} known faces")

    def _convert_to_rgb(self, image):
        """Ensure image is uint8 and RGB for face_recognition"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 2:  # grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:  # BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def recognize(self, frame):
        """
        Detect faces in the input frame and recognize them.
        Returns a list of tuples: (top, right, bottom, left, name)
        """
        results = []
        rgb_frame = self._convert_to_rgb(frame)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for location, encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_encodings, encoding)
            name = "Unknown"
            if True in matches:
                match_index = matches.index(True)
                name = self.known_names[match_index]

            top, right, bottom, left = location
            results.append((top, right, bottom, left, name))

        return results
