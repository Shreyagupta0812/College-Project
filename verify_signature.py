import os
import torch
import numpy as np
from PIL import Image

from siamese_model import SiameseCNN


# -----------------------------
# Settings
# -----------------------------

IMAGE_SIZE = (105, 105)
THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "siamese_model.pth"


# -----------------------------
# Load model
# -----------------------------

model = SiameseCNN().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully!")
print("Using device:", DEVICE)


# -----------------------------
# Preprocess image
# -----------------------------

def preprocess_image(image_path):

    image = Image.open(image_path).convert("L")

    image = image.resize(IMAGE_SIZE)

    image = np.array(
        image,
        dtype=np.float32
    )

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=0
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return torch.tensor(
        image,
        dtype=torch.float32
    )


# -----------------------------
# Extract signature feature
# -----------------------------

def get_feature(image_path):

    image = preprocess_image(
        image_path
    ).to(DEVICE)

    with torch.no_grad():

        feature = model.forward_one(
            image
        )

    return feature


# -----------------------------
# Compare two signatures
# -----------------------------

def compare_signatures(
    reference_path,
    test_path
):

    reference_feature = get_feature(
        reference_path
    )

    test_feature = get_feature(
        test_path
    )

    distance = torch.nn.functional.pairwise_distance(
        reference_feature,
        test_feature
    ).item()

    if distance < THRESHOLD:
        result = "GENUINE"
    else:
        result = "FORGED"

    return distance, result


# -----------------------------
# Test with dataset images
# -----------------------------

reference_path = os.path.join(
    "data",
    "full",
    "100/100_22.png"
)

test_path = os.path.join(
    "data",
    "full",
    "100/100_22.png"
)

distance, result = compare_signatures(
    reference_path,
    test_path
)

print("\n==============================")
print("SIGNATURE VERIFICATION")
print("==============================")
print("Reference:", reference_path)
print("Test:", test_path)
print(f"\nDistance: {distance:.4f}")
print("Threshold:", THRESHOLD)
print("\nResult:", result)