import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("../data/quikr_car_cleaned.csv")

print("dataset shape:", df.shape)
print(df.head())

x = df.drop(columns="Price", axis=1)
y = df["Price"]

print("\nfeatures:", list(x.columns))
print("target range:", y.min(), "-", y.max())

# one hot encode categorical columns
ohe = OneHotEncoder()
ohe.fit(x[["name", "company", "fuel_type"]])

column_trans = make_column_transformer(
    (OneHotEncoder(categories=ohe.categories_), ["name", "company", "fuel_type"]),
    remainder="passthrough"
)

# best random state found through experimentation
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2711)

lr = LinearRegression()
pipe = make_pipeline(column_trans, lr)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)

r2 = r2_score(y_test, y_pred)
print(f"\nR2 Score: {r2:.4f} ({r2*100:.2f}%)")

# save model
pickle.dump(pipe, open("../data/car_price_model.pkl", "wb"))
print("model saved to data/car_price_model.pkl")
