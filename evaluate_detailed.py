
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from siamese_dataset import SignatureDataset
from siamese_model import SiameseCNN


BATCH_SIZE = 16
MODEL_PATH = "siamese_model_v2.pth"
THRESHOLD_PATH = "best_threshold.txt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Read threshold selected using validation data
with open(THRESHOLD_PATH, "r") as file:
    THRESHOLD = float(file.read().strip())

print("Using device:", DEVICE)
print("Selected validation threshold:", THRESHOLD)


# Load untouched test dataset
test_dataset = SignatureDataset(
    "data/test_data.csv"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test pairs:", len(test_dataset))


# Load trained model
model = SiameseCNN().to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()
print("Model loaded successfully!")


# Predict test pairs
true_labels = []
predicted_labels = []

with torch.no_grad():
    for image1, image2, label in test_loader:

        image1 = image1.to(DEVICE)
        image2 = image2.to(DEVICE)

        feature1, feature2 = model(image1, image2)

        distances = torch.nn.functional.pairwise_distance(
            feature1, feature2
        )

        predictions = (
            distances < THRESHOLD
        ).long()

        true_labels.extend(label.cpu().numpy().tolist())
        predicted_labels.extend(predictions.cpu().numpy().tolist())


# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(
    true_labels, predicted_labels, zero_division=0
)
recall = recall_score(
    true_labels, predicted_labels, zero_division=0
)
f1 = f1_score(
    true_labels, predicted_labels, zero_division=0
)

# Explicit order: class 0 = forged, class 1 = genuine
cm = confusion_matrix(
    true_labels, predicted_labels, labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

# Genuine is the positive class:
# False Acceptance Rate = forged pairs predicted genuine
# False Rejection Rate = genuine pairs predicted forged
far = fp / (fp + tn) if (fp + tn) else 0
frr = fn / (fn + tp) if (fn + tp) else 0


# Display results
print("\n========== FINAL TEST EVALUATION ==========")
print(f"Threshold : {THRESHOLD:.2f}")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-score  : {f1 * 100:.2f}%")
print(f"FAR       : {far * 100:.2f}%")
print(f"FRR       : {frr * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nRows = Actual [Forged, Genuine]")
print("Columns = Predicted [Forged, Genuine]")
print("TN:", tn, "| FP:", fp, "| FN:", fn, "| TP:", tp)
