import torch
import joblib
from model import MoodModel

# Load scaler
scaler = joblib.load("scaler.pkl")

# Load model
model = MoodModel()
model.load_state_dict(torch.load("mood_model.pth"))
model.eval()

# Input
sample = [[30, 70, 1, 7, 5000, 12, 0]]

sample = scaler.transform(sample)
sample = torch.tensor(sample, dtype=torch.float32)

# Predict
with torch.no_grad():
    pred = model(sample)

print("Mood:", pred.item())
