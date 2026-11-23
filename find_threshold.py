
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from siamese_dataset import SignatureDataset
from siamese_model import SiameseCNN


# Settings
BATCH_SIZE = 16
MODEL_PATH = "siamese_model_v2.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)


# Load validation dataset
validation_dataset = SignatureDataset(
    "data/validation_data.csv"
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Validation pairs:", len(validation_dataset))


# Load trained model
model = SiameseCNN().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Model loaded successfully!")


# Calculate pair distances
all_distances = []
all_labels = []

with torch.no_grad():
    for image1, image2, label in validation_loader:

        image1 = image1.to(DEVICE)
        image2 = image2.to(DEVICE)

        feature1, feature2 = model(image1, image2)

        distance = torch.nn.functional.pairwise_distance(
            feature1, feature2
        )

        all_distances.extend(distance.cpu().tolist())
        all_labels.extend(label.cpu().tolist())


# Test thresholds on validation data only
thresholds = [
    round(x * 0.1, 1)
    for x in range(1, 16)
]

best_threshold = None
best_accuracy = -1

print("\n========== VALIDATION THRESHOLD TESTING ==========")

for threshold in thresholds:

    predictions = [
        1 if distance < threshold else 0
        for distance in all_distances
    ]

    accuracy = accuracy_score(
        all_labels, predictions
    ) * 100

    print(
        f"Threshold: {threshold:.1f} | "
        f"Validation Accuracy: {accuracy:.2f}%"
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold


print("\n========== SELECTED THRESHOLD ==========")
print("Best threshold:", best_threshold)
print(f"Best validation accuracy: {best_accuracy:.2f}%")

# Save threshold for final evaluation
with open("best_threshold.txt", "w") as file:
    file.write(str(best_threshold))

print("Threshold saved to best_threshold.txt")
