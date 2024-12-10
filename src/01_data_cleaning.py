import numpy as np
import pandas as pd

df = pd.read_csv("../data/quikr_car.csv")

print("original shape:", df.shape)
print(df.head())

# keep only numeric year values
df = df[df["year"].str.isnumeric()]
df["year"] = df["year"].astype(int)

# remove rows where price is "Ask For Price"
df = df[df["Price"] != "Ask For Price"]
df["Price"] = df["Price"].str.replace(",", "").astype(int)

# clean kms_driven — remove "kms" text and commas, keep only numeric
df["kms_driven"] = df["kms_driven"].str.split(" ").str.get(0).str.replace(",", "")
df = df[df["kms_driven"].str.isnumeric()]
df["kms_driven"] = df["kms_driven"].astype(int)

# remove rows with missing fuel type
df = df[~df["fuel_type"].isna()]

# keep only first 3 words of car name
df["name"] = df["name"].str.split(" ").str.slice(0, 3).str.join(" ")

# reset index
df = df.reset_index(drop=True)

# remove price outliers above 6 million
df = df[df["Price"] <= 6000000].reset_index(drop=True)

print("\ncleaned shape:", df.shape)
print(df.info())
print(df.describe())

df.to_csv("../data/quikr_car_cleaned.csv", index=False)
print("\nsaved to data/quikr_car_cleaned.csv")
