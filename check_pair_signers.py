
import pandas as pd

data = pd.read_csv("data/data.csv")

print("Columns:", data.columns.tolist())
print("\nLabel distribution:")
print(data["2"].value_counts().sort_index())

print("\nExamples with label 0:")
print(data[data["2"] == 0].head(5).to_string(index=False))

print("\nExamples with label 1:")
print(data[data["2"] == 1].head(5).to_string(index=False))
