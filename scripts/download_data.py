from sklearn.datasets import fetch_openml
import os

titanic = fetch_openml(name='titanic', version=1, as_frame=True)

os.makedirs("dataset", exist_ok=True)

titanic.data.to_csv("dataset/data.csv", index=False)
titanic.target.to_csv("dataset/target.csv", index=False)