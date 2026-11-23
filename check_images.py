import os
import pandas as pd
from PIL import Image

# Read training CSV
df = pd.read_csv("data/train_data.csv")

# Take the first pair
image1 = df.iloc[0]["0"]
image2 = df.iloc[0]["1"]

# Build complete paths
path1 = os.path.join("data", "full", image1)
path2 = os.path.join("data", "full", image2)

print("Image 1:", path1)
print("Image 2:", path2)

# Check whether files exist
print("\nImage 1 exists:", os.path.exists(path1))
print("Image 2 exists:", os.path.exists(path2))

# Open images and display their information
if os.path.exists(path1):
    img1 = Image.open(path1)
    print("\nImage 1 size:", img1.size)
    print("Image 1 mode:", img1.mode)

if os.path.exists(path2):
    img2 = Image.open(path2)
    print("Image 2 size:", img2.size)
    print("Image 2 mode:", img2.mode)

print("\nLabel:", df.iloc[0]["2"])