# decision/test_decision.py

import cv2
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_extractor.reid_extractor import FeatureExtractor
from memory.feature_memory import FeatureMemory
from decision.decision_logic import DecisionEngine
from config.settings import DEVICE

extractor = FeatureExtractor(device=DEVICE)
memory = FeatureMemory(max_age_seconds=3600)
decision_engine = DecisionEngine(
    memory=memory,
    similarity_threshold=0.85,
    time_window_seconds=300
)

print("Decision engine initialized.") # why print this? because we want to know if the decision engine is initialized successfully or not. If it is not initialized, we will not be able to test the decision logic.
crop_folder = "output/crops"
crop_files = sorted([f for f in os.listdir(crop_folder) if f.endswith(".jpg")])

if len(crop_files) == 0:
    print("No crop pictures found. Run Module 3 first.")
    sys.exit()

print(f"Found {len(crop_files)} crop pictures. Testing decision logic...\n")

for i, filename in enumerate(crop_files):
    image = cv2.imread(os.path.join(crop_folder, filename))
    feature_vector = extractor.extract(image)

    if feature_vector is None:
        print(f"{filename}: could not extract features, skipped.")
        continue

    decision, matched_index, matched_track_id, similarity, time_since_seen = decision_engine.evaluate(
        track_id=i, feature_vector=feature_vector
    )

    print(
        f"{filename}: decision = {decision}, matched_index = {matched_index}, "
        f"matched_track_id = {matched_track_id}, memory size = {memory.count()}"
    )