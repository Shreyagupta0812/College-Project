
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from siamese_dataset import SignatureDataset
from siamese_model import SiameseCNN


TEST_CSV = "data/clean_test.csv"
MODEL_PATH = "siamese_model_v3.pth"
THRESHOLD_PATH = "best_threshold_v3.txt"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(THRESHOLD_PATH, "r") as file:
    threshold = float(file.read().strip())

print("Using device:", device)
print("Selected validation threshold:", threshold)

dataset = SignatureDataset(TEST_CSV)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = SiameseCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

distances = []
labels = []

with torch.no_grad():
    for image1, image2, label in loader:
        image1 = image1.to(device)
        image2 = image2.to(device)

        feature1, feature2 = model(image1, image2)
        distance = torch.nn.functional.pairwise_distance(
            feature1, feature2
        )

        distances.extend(distance.cpu().numpy())
        labels.extend(label.numpy())

distances = np.array(distances)
labels = np.array(labels).astype(int)

# Label 1 = genuine, label 0 = forged
predictions = (distances < threshold).astype(int)

accuracy = accuracy_score(labels, predictions)
precision = precision_score(
    labels, predictions, pos_label=1, zero_division=0
)
recall = recall_score(
    labels, predictions, pos_label=1, zero_division=0
)
f1 = f1_score(
    labels, predictions, pos_label=1, zero_division=0
)

# Explicit ordering: forged (0), genuine (1)
cm = confusion_matrix(labels, predictions, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

far = fp / (fp + tn) if (fp + tn) else 0.0
frr = fn / (fn + tp) if (fn + tp) else 0.0

print("\n========== FINAL CLEAN TEST RESULTS ==========")
print("Test pairs :", len(labels))
print(f"Threshold  : {threshold:.2f}")
print(f"Accuracy   : {accuracy * 100:.2f}%")
print(f"Precision  : {precision * 100:.2f}%")
print(f"Recall     : {recall * 100:.2f}%")
print(f"F1-score   : {f1 * 100:.2f}%")
print(f"FAR        : {far * 100:.2f}%")
print(f"FRR        : {frr * 100:.2f}%")

print("\nConfusion matrix:")
print(cm)
print("Rows = Actual [Forged, Genuine]")
print("Columns = Predicted [Forged, Genuine]")
print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
