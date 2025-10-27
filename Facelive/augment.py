import os
import cv2
import numpy as np
np.bool = bool 
from imgaug import augmenters as iaa
from tqdm import tqdm

# Path to the parent folder containing all student subfolders
parent_folder_path = 'test_images'  # Replace with your parent folder path

# Name of the student subfolder to augment
student_subfolder = '23345A0518'  # Replace with the specific student folder

# Augmentation configurations
augmenters = iaa.SomeOf((2, 4), [  # Apply 2 to 4 augmentations from the list
    iaa.Fliplr(0.5),              # Horizontal flip with 50% probability
    iaa.Affine(rotate=(-45, 45)), # Rotate between -45 to 45 degrees
    iaa.Multiply((0.8, 1.2)),     # Random brightness adjustment
    iaa.GaussianBlur(sigma=(0, 1.0)),  # Add Gaussian blur
    iaa.AdditiveGaussianNoise(scale=(10, 30)),  # Add Gaussian noise
    iaa.Sharpen(alpha=(0.2, 0.5), lightness=(0.8, 1.2)),  # Sharpen the image
    iaa.Crop(percent=(0, 0.1)),   # Random cropping
    iaa.LinearContrast((0.75, 1.5)),  # Change contrast
    iaa.SomeOf((0, 1), [           # Optional grayscale conversion
        iaa.Grayscale(alpha=1.0)   # Convert to grayscale with 100% probability
    ])
])

def augment_single_student(parent_folder, student_folder):
    subfolder_path = os.path.join(parent_folder, student_folder)
    if not os.path.exists(subfolder_path):
        print(f"Error: Folder '{student_folder}' does not exist in '{parent_folder}'.")
        return

    images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"No valid images found in subfolder: {student_folder}")
        return

    print(f"Processing {len(images)} images in subfolder: {student_folder}")
    for img_name in tqdm(images, desc=f"Augmenting {student_folder}"):
        img_path = os.path.join(subfolder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read image {img_name}")
            continue

        # Generate 50 augmented versions of the image
        for i in range(1, 51):
            aug_img = augmenters(image=img)
            save_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            save_path = os.path.join(subfolder_path, save_name)
            cv2.imwrite(save_path, aug_img)

        # Optional: Add grayscale version
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayscale_path = os.path.join(subfolder_path, f"{os.path.splitext(img_name)[0]}_grayscale.jpg")
        cv2.imwrite(grayscale_path, grayscale_img)

    print("Augmentation complete!")

# Call the function for the specific student
augment_single_student(parent_folder_path, student_subfolder)
