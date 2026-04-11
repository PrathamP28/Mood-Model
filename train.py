import torch
import torch.nn as nn
import torch.optim as optim
from model import MoodModel
from utils import load_data

X_train, y_train = load_data("data.csv")

model = MoodModel()

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_loss = float('inf')

for epoch in range(200):
    model.train()

    output = model(X_train)
    loss = loss_fn(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save best model
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "mood_model.pth")
        print("✅ Saved best model")
