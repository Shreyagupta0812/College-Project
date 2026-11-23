
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from siamese_dataset import SignatureDataset
from siamese_model import SiameseCNN


VALIDATION_CSV = "data/clean_validation.csv"
MODEL_PATH = "siamese_model_v3.pth"
THRESHOLD_PATH = "best_threshold_v3.txt"

BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = SignatureDataset(VALIDATION_CSV)
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

print("Validation pairs:", len(labels))
print("\n========== THRESHOLD SEARCH ==========")

best_threshold = None
best_accuracy = -1.0

for threshold in np.arange(0.1, 2.01, 0.1):
    # Small distance means genuine (label 1).
    predictions = (distances < threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)

    print(f"Threshold {threshold:.2f}: Accuracy {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = float(threshold)

with open(THRESHOLD_PATH, "w") as file:
    file.write(f"{best_threshold:.2f}")

print("\nBest validation threshold:", f"{best_threshold:.2f}")
print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
print("Threshold saved to:", THRESHOLD_PATH)
