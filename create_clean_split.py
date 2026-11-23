
import random
from collections import defaultdict
from itertools import combinations
from pathlib import PurePosixPath

import pandas as pd

# Reproducible split
random.seed(42)

SOURCE_CSV = "data/data.csv"
OUTPUT_FILES = {
    "train": "data/clean_train.csv",
    "validation": "data/clean_validation.csv",
    "test": "data/clean_test.csv",
}

# Maximum pairs generated per signer and split
MAX_POSITIVE_PAIRS = 100
MAX_NEGATIVE_PAIRS = 100


def get_signer_and_type(image_path):
    path = PurePosixPath(str(image_path).replace("\\", "/"))
    folder = path.parts[0]

    is_forged = folder.endswith("_forg")
    signer = folder[:-5] if is_forged else folder

    return signer, is_forged


def split_images(images):
    """Split one image list into approximately 70%, 15%, 15%."""
    images = sorted(set(images))
    random.shuffle(images)

    n = len(images)
    if n < 3:
        return [], [], []

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # Ensure every split gets at least one image
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    return (
        images[:train_end],
        images[train_end:val_end],
        images[val_end:],
    )


def make_pairs(genuine_images, forged_images):
    pairs = []

    # Genuine pairs: two different genuine images
    positive_candidates = list(combinations(sorted(genuine_images), 2))
    random.shuffle(positive_candidates)
    positive_candidates = positive_candidates[:MAX_POSITIVE_PAIRS]

    for image1, image2 in positive_candidates:
        pairs.append((image1, image2, 1))

    # Forged pairs: one genuine image and one forged image
    negative_candidates = [
        (genuine, forged)
        for genuine in sorted(genuine_images)
        for forged in sorted(forged_images)
    ]
    random.shuffle(negative_candidates)
    negative_candidates = negative_candidates[:MAX_NEGATIVE_PAIRS]

    for image1, image2 in negative_candidates:
        pairs.append((image1, image2, 0))

    return pairs


# Read the original pair list and collect unique image paths
data = pd.read_csv(SOURCE_CSV)
images_by_signer = defaultdict(lambda: {"genuine": set(), "forged": set()})

for _, row in data.iterrows():
    for column in ["0", "1"]:
        image_path = str(row[column])
        signer, is_forged = get_signer_and_type(image_path)

        category = "forged" if is_forged else "genuine"
        images_by_signer[signer][category].add(image_path)

# Split genuine and forged images separately for each signer
split_images_by_name = {
    "train": defaultdict(lambda: {"genuine": [], "forged": []}),
    "validation": defaultdict(lambda: {"genuine": [], "forged": []}),
    "test": defaultdict(lambda: {"genuine": [], "forged": []}),
}

skipped_signers = []

for signer, categories in sorted(images_by_signer.items()):
    genuine_splits = split_images(categories["genuine"])
    forged_splits = split_images(categories["forged"])

    # Require at least one genuine and forged image in all three splits
    if any(len(part) == 0 for part in genuine_splits) or any(
        len(part) == 0 for part in forged_splits
    ):
        skipped_signers.append(signer)
        continue

    for split_name, genuine, forged in zip(
        ["train", "validation", "test"],
        genuine_splits,
        forged_splits,
    ):
        split_images_by_name[split_name][signer]["genuine"] = genuine
        split_images_by_name[split_name][signer]["forged"] = forged

# Construct pairs within each split only
all_pairs = {}

for split_name, signer_data in split_images_by_name.items():
    pairs = []

    for signer, categories in signer_data.items():
        pairs.extend(
            make_pairs(categories["genuine"], categories["forged"])
        )

    random.shuffle(pairs)
    all_pairs[split_name] = pairs

# Save new CSV files
for split_name, output_path in OUTPUT_FILES.items():
    frame = pd.DataFrame(all_pairs[split_name], columns=["0", "1", "2"])
    frame.to_csv(output_path, index=False)

    print(f"\n{split_name.upper()}")
    print("Pairs:", len(frame))
    if not frame.empty:
        print("Labels:")
        print(frame["2"].value_counts().sort_index().to_dict())
    print("Saved:", output_path)

# Verify that no image path crosses splits
image_sets = {}

for split_name, pairs in all_pairs.items():
    image_sets[split_name] = {
        image
        for image1, image2, _ in pairs
        for image in (image1, image2)
    }

print("\n========== IMAGE OVERLAP CHECK ==========")
names = list(image_sets)

for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        overlap = image_sets[a] & image_sets[b]
        print(f"{a} vs {b}: {len(overlap)} shared image paths")

print("\nSigners skipped due to insufficient images:", len(skipped_signers))
if skipped_signers:
    print("Examples:", skipped_signers[:10])

print("\nSplit creation completed.")
