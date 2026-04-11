import torch
import numpy as np
from model import MoodModel
from utils import load_data

X, y = load_data("data.csv")

model = MoodModel()
model.load_state_dict(torch.load("mood_model.pth"))
model.eval()

def importance(model, X):
    base = model(X).detach().numpy()
    scores = []

    for i in range(X.shape[1]):
        X_temp = X.clone()
        X_temp[:, i] = X_temp[torch.randperm(X.shape[0]), i]

        new_pred = model(X_temp).detach().numpy()
        diff = np.mean(abs(base - new_pred))
        scores.append(diff)

    return scores

features = ["temp","humidity","rain","sleep","steps","hour","season"]

scores = importance(model, X)

for f, s in zip(features, scores):
    print(f"{f}: {s}")
