import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(path):
    data = pd.read_csv(path)

    X = data.drop("mood", axis=1).values
    y = data["mood"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    joblib.dump(scaler, "scaler.pkl")

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    return X, y
