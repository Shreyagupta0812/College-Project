
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the existing training pairs
data = pd.read_csv("data/train_data.csv")

print("Original training dataset:", data.shape)
print("Columns:", data.columns.tolist())

# Split into 90% training and 10% validation
train_data, validation_data = train_test_split(
    data,
    test_size=0.10,
    random_state=42,
    stratify=data["2"]
)

# Save separate files; do not overwrite the originals
train_data.to_csv("data/train_split.csv", index=False)
validation_data.to_csv("data/validation_data.csv", index=False)

print("\nSplit completed successfully!")
print("New training pairs:", len(train_data))
print("Validation pairs:", len(validation_data))
print("\nTraining label counts:")
print(train_data["2"].value_counts())

print("\nValidation label counts:")
print(validation_data["2"].value_counts())
