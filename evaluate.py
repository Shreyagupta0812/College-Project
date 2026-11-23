import os
import torch
from torch.utils.data import DataLoader

from siamese_dataset import SignatureDataset
from siamese_model import SiameseCNN

# -----------------------------
# Settings
# -----------------------------

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "siamese_model.pth"

print("Using device:", DEVICE)

# -----------------------------
# Load test dataset
# -----------------------------

test_dataset = SignatureDataset("data/test_data.csv")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test pairs:", len(test_dataset))

# -----------------------------
# Load trained model
# -----------------------------

model = SiameseCNN().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Model loaded successfully!")

# -----------------------------
# Test model
# -----------------------------

correct = 0
total = 0

with torch.no_grad():

    for image1, image2, label in test_loader:

        image1 = image1.to(DEVICE)
        image2 = image2.to(DEVICE)
        label = label.to(DEVICE)

        feature1, feature2 = model(image1, image2)

        # Calculate distance
        distance = torch.nn.functional.pairwise_distance(
            feature1,
            feature2
        )

        # Distance threshold
        # Smaller distance = more similar
        prediction = (distance < 1.0).float()

        correct += (prediction == label).sum().item()
        total += label.size(0)

# -----------------------------
# Accuracy
# -----------------------------

accuracy = (correct / total) * 100

print("\n-----------------------------")
print("Evaluation completed!")
print("-----------------------------")
print("Correct predictions:", correct)
print("Total test pairs:", total)
print(f"Test Accuracy: {accuracy:.2f}%")