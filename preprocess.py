import os
import pandas as pd
import numpy as np
from PIL import Image

# Image size required by the project
IMAGE_SIZE = (105, 105)

# Load training CSV
df = pd.read_csv("data/train_data.csv")


def preprocess_image(image_path):
    # Open image and convert to grayscale
    image = Image.open(image_path).convert("L")

    # Resize to 105 x 105
    image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array
    image = np.array(image, dtype=np.float32)

    # Normalize pixel values from 0-255 to 0-1
    image = image / 255.0

    return image


# Get first image path from CSV
relative_path = df.iloc[0]["0"]

# Complete path
image_path = os.path.join("data", "full", relative_path)

# Preprocess image
processed_image = preprocess_image(image_path)

print("Original image:", image_path)
print("Processed shape:", processed_image.shape)
print("Data type:", processed_image.dtype)
print("Minimum pixel value:", processed_image.min())
print("Maximum pixel value:", processed_image.max())