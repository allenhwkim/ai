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
    # Always pass a 2D NumPy array or a PyTorch tensor.
    #  using `torch.from_numpy(<numpy array>).unsqueeze(1)`
    # If pass a 1D NumPy array, IndexError: too many indices for array
    # If pass a list, TypeError: only integer scalar arrays can be converted to a scalar index
    # If pass a NumPy array, TypeError: expected Tensor as element 0 in argument 0, but got numpy.ndarray
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
y_numpy = (y_numpy - y_numpy.mean()) / y_numpy.std()  # Normalize the target

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

# Dataset -> DataLoader
# batch_size = len(dataset)
batch_size = 32 
# dataset in a DataLoader with batch_size=32 enabling mini-batch SGD.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Datasset -> Dataloader -> Model
# y = ax^3 + bx^2 + cx + d
class NonLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))  # Coefficient for x^3
        self.b = nn.Parameter(torch.randn(1))  # Coefficient for x^2
        self.c = nn.Parameter(torch.randn(1))  # Coefficient for x
        self.d = nn.Parameter(torch.randn(1))  # Constant term
    
    def forward(self, x):
        # Non-linear polynomial regression: y = ax^3 + bx^2 + cx + d
        return self.a * x**3 + self.b * x**2 + self.c * x + self.d

model = NonLinearRegressionModel()

# Dataset -> DataLoader -> Model -> Loss Function 
criterion = nn.MSELoss()  # Mean Squared Error Loss

# Dataset -> DataLoader -> Model -> Loss Function -> Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001) # Adam Optimizer

# Dataset -> DataLoader -> Model -> Loss Function -> Optimizer -> Training Loop
num_epochs = 10000
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    train_loss = 0.0
    for x_inputs, y_targets in train_loader:
        y_preds = model(x_inputs) # Forward pass
        loss = criterion(y_preds, y_targets)  # Compute loss
        train_loss += loss.item()  # Accumulate loss for averaging
        optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)  # Store loss for plotting
    
    # Validation
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        val_loss = 0.0
        for x_inputs, y_targets in val_loader:
            y_preds = model(x_inputs)
            loss = criterion(y_preds, y_targets)  # Compute validation loss
            val_loss += loss.item()  # Accumulate validation loss for averaging
    val_loss /= len(val_dataset)
    val_losses.append(val_loss)  # Store validation loss for plotting

model.eval()  # Set model to evaluation mode

with torch.no_grad():
    test_loss = 0.0
    for x_inputs, y_targets in test_loader:
        y_preds = model(x_inputs)
        loss += criterion(y_preds, y_targets)
        test_loss += loss.item() * len(x_inputs)
    test_loss /= len(test_dataset)  # Average test loss
print(f"Final Test Loss: {test_loss:.4f}")

train_losses = train_losses[20:]  # Remove initial losses for better visualization
val_losses = val_losses[20:]  # Remove initial losses for better visualization

# Plotting the loss curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Loss Curves')
plt.legend()

# Plotting the predictions
# Plot scatter plot of x and y, colored by split
plt.subplot(1, 2, 2)
train_indices = train_dataset.indices
val_indices = val_dataset.indices
test_indices = test_dataset.indices 
plt.scatter(x_numpy[train_indices], y_numpy[train_indices], label='Train Data', color='blue', s=10)
plt.scatter(x_numpy[val_indices], y_numpy[val_indices], label='Validation Data', color='orange', s=10)
plt.scatter(x_numpy[test_indices], y_numpy[test_indices], label='Test Data', color='red', s=10)

a = model.a.item()
b = model.b.item()
c = model.c.item()
d = model.d.item()
formula_str = f"y={a:.3f}x³+{b:.2f}x²+{c:.2f}x+{d:.2f}"
x_nums = torch.linspace(x_numpy.min(), x_numpy.max(), 100, dtype=torch.float32).reshape(-1, 1)
y_nums = model(x_nums).detach().numpy()
plt.plot(x_nums, y_nums, label=f"Fitted Line\n({formula_str})", color='red')
plt.title("Linear Fit using Gradient Descent")
plt.legend()
plt.show()