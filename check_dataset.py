import pandas as pd

for file in ["data.csv", "test_data.csv", "train_data.csv"]:
    df = pd.read_csv("data/" + file)

    print("\n==============================")
    print(file)
    print("==============================")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())
    