"""
This code shows how to fit a straight line to noisy data using PyTorch
It uses mini-batch SGD for optimization with a randomly selected data point.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# DDMLOT (Dataset -> DataLoader -> Model -> Loss Function -> Optimizer -> Training Loop)
class RegressionDataset(TensorDataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]

# x = [-1.5, -0.8, 0.1, 0.9, 1.7]
# y = [0.3, -0.3, 0.5, 1.8, 1.5]
x = np.linspace(-2, 2, 500)
# y = ax + b + noise. x.shape is (500,)
y = 0.8 * x + 0.7 + np.random.normal(0, 0.7, size=x.shape)  

x_numpy = np.array(x, dtype=np.float32)
y_numpy = np.array(y, dtype=np.float32)

x_tensor = torch.from_numpy(x_numpy).unsqueeze(1)
y_tensor = torch.from_numpy(y_numpy).unsqueeze(1)

# Dataset
# Converts data to PyTorch tensors
dataset = RegressionDataset(x_tensor, y_tensor)

# Dataset -> DataLoader
# batch_size = len(dataset)
batch_size = 32 
# Wraps the dataset in a DataLoader with batch_size=32 and shuffling,
# enabling mini-batch SGD.
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Datasset -> Dataloader -> Model
class LinearRegressionModel(nn.Module):
    def __init__(self, initial_weight=-10.0, initial_bias=10.0):
        super().__init__()

        # Trainable parameters (weights)
        # nn.Parameter(...) wraps this tensor, marking it as a parameter 
        # that should be tracked by PyTorch’s autograd system. 
        # This means that during training, the optimizer will update self.weight and self.bias
        # based on the computed gradients.
        self.weight = nn.Parameter(torch.tensor([initial_weight], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([initial_bias], dtype=torch.float32))

    def forward(self, x):
        out = self.weight * x + self.bias
        return out

model = LinearRegressionModel(initial_weight=-10, initial_bias=10)

# Dataset -> DataLoader -> Model -> Loss Function 
criterion = nn.MSELoss()  # Mean Squared Error Loss

# Dataset -> DataLoader -> Model -> Loss Function -> Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Stochastic Gradient Descent Optimizer

# Dataset -> DataLoader -> Model -> Loss Function -> Optimizer -> Training Loop
num_epochs = 100
losses = []
for epoch in range(num_epochs):
    for x_inputs, y_targets in dataloader:
        y_preds = model(x_inputs)  # Forward pass
        loss = criterion(y_preds, y_targets)  # Compute loss
        losses.append(loss.item())  # Store loss for plotting
        optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

    # print weighg, bias and loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Weight = {model.weight.item():.4f}, Bias = {model.bias.item():.4f}, Loss = {loss.item():.4f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')

# Plot scatter plot of x and y
plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Data Points', color='blue')
x_nums = np.linspace(min(x), max(x), 100, dtype=np.float32)
with torch.no_grad(): # w/o it, it tracks automatic differentiation(autograd)
    w = float(model.weight)
    b = float(model.bias)
    y_nums = w * x_nums + b 
plt.plot(x_nums, y_nums, label='Fitted Line', color='red')
plt.title("Linear Fit using Gradient Descent")
plt.show()
