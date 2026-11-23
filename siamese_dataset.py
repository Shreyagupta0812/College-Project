import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_SIZE = (105, 105)


class SignatureDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def preprocess_image(self, image_path):

        # Open image and convert to grayscale
        image = Image.open(image_path).convert("L")

        # Resize to 105 x 105
        image = image.resize(IMAGE_SIZE)

        # Convert to NumPy array
        image = np.array(image, dtype=np.float32)

        # Normalize pixel values
        image = image / 255.0

        # Add channel dimension
        # (105, 105) -> (1, 105, 105)
        image = np.expand_dims(image, axis=0)

        # Convert to PyTorch tensor
        image = torch.tensor(image, dtype=torch.float32)

        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Get image paths from CSV
        image1_path = os.path.join(
            "data", "full", self.data.iloc[index]["0"]
        )

        image2_path = os.path.join(
            "data", "full", self.data.iloc[index]["1"]
        )

        # Load and preprocess both images
        image1 = self.preprocess_image(image1_path)
        image2 = self.preprocess_image(image2_path)

        # Get label
        label = torch.tensor(
            float(self.data.iloc[index]["2"]),
            dtype=torch.float32
        )

        return image1, image2, label


# Test the dataset
if __name__ == "__main__":

    dataset = SignatureDataset("data/train_data.csv")

    print("Number of training pairs:", len(dataset))

    image1, image2, label = dataset[0]

    print("Image 1 shape:", image1.shape)
    print("Image 2 shape:", image2.shape)
    print("Label:", label.item())