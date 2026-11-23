
import pandas as pd
from pathlib import PurePosixPath

FILES = {
    "train": "data/train_split.csv",
    "validation": "data/validation_data.csv",
    "test": "data/test_data.csv",
}


def signer_id(image_path):
    path = PurePosixPath(str(image_path).replace("\\", "/"))
    folder = path.parts[0]

    # Example: 100_forg -> signer 100
    return folder.split("_")[0]


def collect_data(csv_path):
    df = pd.read_csv(csv_path)

    images = set(df["0"].astype(str)) | set(df["1"].astype(str))

    signers = {
        signer_id(path)
        for path in images
    }

    return images, signers


data = {
    name: collect_data(path)
    for name, path in FILES.items()
}

print("\n========== IMAGE OVERLAP ==========")

names = list(FILES)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]

        overlap = data[a][0] & data[b][0]

        print(
            f"{a} vs {b}: "
            f"{len(overlap)} shared image paths"
        )

print("\n========== SIGNER OVERLAP ==========")

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]

        overlap = data[a][1] & data[b][1]

        print(
            f"{a} vs {b}: "
            f"{len(overlap)} shared signer IDs"
        )

print("\nCheck completed.")
