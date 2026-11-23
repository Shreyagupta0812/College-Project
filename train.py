
import torch
from torch.utils.data import DataLoader

from siamese_dataset import SignatureDataset
from siamese_model import SiameseCNN
from contrastive_loss import ContrastiveLoss


# -----------------------------
# Configuration
# -----------------------------
TRAIN_CSV = "data/clean_train.csv"
MODEL_PATH = "siamese_model_v3.pth"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# Load training data
# -----------------------------
dataset = SignatureDataset(TRAIN_CSV)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Training pairs:", len(dataset))


# -----------------------------
# Initialize model and loss
# -----------------------------
model = SiameseCNN().to(device)
criterion = ContrastiveLoss(margin=1.0)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training loop
# -----------------------------
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0

    for image1, image2, label in loader:

        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        feature1, feature2 = model(image1, image2)
        loss = criterion(feature1, feature2, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * image1.size(0)

    epoch_loss = running_loss / len(dataset)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f}"
    )


# -----------------------------
# Save new model
# -----------------------------
torch.save(model.state_dict(), MODEL_PATH)

print("\nTraining completed!")
print("Model saved as:", MODEL_PATH)
