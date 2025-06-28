"""
This code fits a linear regression model to predict miles per gallon (mpg)
based on horsepower using PyTorch with data from Kagglehub. 

Split data into three sets:
- Training set: Used to fit (train) the model's parameters.
- Validation set: Used to tune hyperparameters and monitor model performance during training
  for early stopping or model selection.  The model does not "see" this data during training.
- Test set: Used only once, after all training and tuning to evaluate the final model's performance.
  This gives an unbiased estimate of how the model will perform on new, unseen data.
"""
# brew install graphviz
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import altair as alt
import pandas as pd
import kagglehub

"""
DDMLOT (Dataset -> DataLoader -> Model -> Loss Function -> Optimizer -> Training Loop)
"""
class RegressionDataset(TensorDataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.x[index], self.y[index]

path = kagglehub.dataset_download("jayhingrajiya/auto-mpg-dataset-miles-per-gallon")
data = pd.read_csv(path + "/auto.csv")
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data = data.dropna(subset=['horsepower', 'mpg'])  # Add this line

x_numpy = data['horsepower'].values.astype(np.float32)
# When input values are large, the gradients and parameter updates become huge, causing instability and overflow.
x_numpy = (x_numpy - x_numpy.mean()) / x_numpy.std()  # Important! Normalize the feature
y_numpy = data['mpg'].values.astype(np.float32)

x_tensor = torch.from_numpy(x_numpy).unsqueeze(1)
y_tensor = torch.from_numpy(y_numpy).unsqueeze(1)

# Dataset
# Converts data to PyTorch tensors
dataset = RegressionDataset(x_tensor, y_tensor)

# --- Split dataset into train, validation, and test sets ---
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
# print train_dataset, val_dataset, test_dataset
print(f"Train dataset: {train_dataset}\nValidation dataset: {val_dataset}\nTest dataset: {test_dataset}")

# Dataset -> DataLoader
# batch_size = len(dataset)
batch_size = 32 
# dataset in a DataLoader with batch_size=32 enabling mini-batch SGD.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Datasset -> Dataloader -> Model
class LinearRegressionModel(nn.Module):
    def __init__(self, initial_weight=-10.0, initial_bias=50.0):
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
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    for x_inputs, y_targets in train_loader:
        y_preds = model(x_inputs)  # Forward pass
        loss = criterion(y_preds, y_targets)  # Compute loss
        train_losses.append(loss.item())  # Store loss for plotting
        optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
    
    # Validation loss
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for x_inputs, y_targets in val_loader:
            y_preds = model(x_inputs)
            val_loss = criterion(y_preds, y_targets)
            val_losses.append(val_loss.item())  # Store validation loss for plotting

    # print weighg, bias and loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Weight = {model.weight.item():.4f}, Bias = {model.bias.item():.4f}, "
            f"Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}")

print(f"Val Loss: {val_loss:.4f}, val_losses: {val_losses}")

# Evaluate on test set
model.eval()  # Set model to evaluation mode
test_losses = []
with torch.no_grad():
    for x_inputs, y_targets in test_loader:
        y_preds = model(x_inputs)
        test_loss = criterion(y_preds, y_targets).item()
        test_losses.append(test_loss)  # Store test loss for plotting
print(f"Test Loss: {test_loss:.4f}, test_losses: {test_losses}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.plot(test_losses, label='Test Loss', color='red')
plt.xlabel('Iteration/Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend()

# Plot scatter plot of x and y
# Get indices for each split
train_indices = train_dataset.indices
val_indices = val_dataset.indices
test_indices = test_dataset.indices 

# Plot scatter plot of x and y, colored by split
plt.subplot(1, 2, 2)
plt.scatter(x_numpy[train_indices], y_numpy[train_indices], label='Train Data', color='blue', s=10)
plt.scatter(x_numpy[val_indices], y_numpy[val_indices], label='Validation Data', color='orange', s=10)
plt.scatter(x_numpy[test_indices], y_numpy[test_indices], label='Test Data', color='red', s=10)

x_nums = np.linspace(np.min(x_numpy), np.max(x_numpy), 100, dtype=np.float32)
with torch.no_grad():
    w = float(model.weight)
    b = float(model.bias)
    y_nums = w * x_nums + b 
plt.plot(x_nums, y_nums, label='Fitted Line', color='red')
plt.title("Linear Fit using Gradient Descent")
plt.legend()
plt.show()