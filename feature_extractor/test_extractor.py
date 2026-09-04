# feature_extractor/test_extractor.py

import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_extractor.reid_extractor import FeatureExtractor
from config.settings import DEVICE

extractor = FeatureExtractor(device=DEVICE)

print(f"Running on device: {DEVICE}")

# Find crop pictures saved from Module 3
crop_folder = "output/crops"
crop_files = [f for f in os.listdir(crop_folder) if f.endswith(".jpg")]

if len(crop_files) < 2:
    print("Need at least 2 saved crop pictures to test. Run Module 3 first.")
    sys.exit()

print(f"Found {len(crop_files)} crop pictures. Testing first 2...")

# Load two crop pictures
image1 = cv2.imread(os.path.join(crop_folder, crop_files[0]))
image2 = cv2.imread(os.path.join(crop_folder, crop_files[1]))

# Extract feature vectors
vector1 = extractor.extract(image1)
vector2 = extractor.extract(image2)

print(f"Picture 1: {crop_files[0]}")
print(f"Vector 1 shape: {vector1.shape}")
print(f"First 5 numbers: {vector1[:5]}")

print(f"\nPicture 2: {crop_files[1]}")
print(f"Vector 2 shape: {vector2.shape}")
print(f"First 5 numbers: {vector2[:5]}")


def simple_cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)


similarity = simple_cosine_similarity(vector1, vector2)
print(f"\n Similarity between picture 1 and 2: {similarity:.4f}")
print("(1.0 = identical, 0.0 = totally different)")