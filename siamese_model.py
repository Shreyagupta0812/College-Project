import torch
import torch.nn as nn


class SiameseCNN(nn.Module):

    def __init__(self):
        super(SiameseCNN, self).__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(

            # First convolution block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Third convolution block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Calculate feature size after CNN
        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Linear(128 * 13 * 13, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward_one(self, x):

        # Extract features from one signature
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def forward(self, image1, image2):

        # Same CNN is used for both images
        feature1 = self.forward_one(image1)
        feature2 = self.forward_one(image2)

        return feature1, feature2


# Test the model
if __name__ == "__main__":

    model = SiameseCNN()

    # Create two dummy 105 x 105 grayscale images
    image1 = torch.randn(1, 1, 105, 105)
    image2 = torch.randn(1, 1, 105, 105)

    feature1, feature2 = model(image1, image2)

    print("Image 1 feature shape:", feature1.shape)
    print("Image 2 feature shape:", feature2.shape)