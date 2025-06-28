"""
This code shows how to fit a straight line to noisy data using PyTorch's
deep learning tools, following the standard pipeline for supervised learning.
It uses mini-batch SGD for optimization.

Split data into three sets:
- Training set: Used to fit (train) the model’s parameters.
- Validation set: Used to tune hyperparameters and monitor model performance
during training (e.g., for early stopping or model selection). 
  The model does not "see" this data during training.
- Test set: Used only once, after all training and tuning, 
  to evaluate the final model’s performance.
  This gives an unbiased estimate of how the model will perform on new, 
  unseen data.
"""
# brew install graphviz
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torchviz
from torch.utils.data import random_split

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

# --- Split dataset into train, validation, and test sets ---
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


# Dataset -> DataLoader
# batch_size = len(dataset)
batch_size = 32 
# dataset in a DataLoader with batch_size=32 enabling mini-batch SGD.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
train_losses = []
val_losses = []
test_losses = []

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
    val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            y_val_preds = model(x_val)
            val_loss += criterion(y_val_preds, y_val).item()
    val_loss /= len(val_loader)  # Average validation loss
    val_losses.append(val_loss)

    # print weighg, bias and loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Weight = {model.weight.item():.4f}, Bias = {model.bias.item():.4f}, "
            f"Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}")

# Evaluate on test set
model.eval()  # Set model to evaluation mode
test_loss = 0.0
with torch.no_grad():
    for x_test, y_test in test_loader:
        y_test_preds = model(x_test)
        test_loss += criterion(y_test_preds, y_test).item()
test_loss /= len(test_loader)  # Average test loss
test_losses.append(test_loss)
print(f"Test Loss: {test_loss:.4f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
plt.xlabel('Epoch' if len(train_losses) == len(val_losses) else 'Iteration/Epoch')
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
plt.scatter(x_numpy[test_indices], y_numpy[test_indices], label='Test Data', color='green', s=10)

x_nums = np.linspace(min(x), max(x), 100, dtype=np.float32)
with torch.no_grad():
    w = float(model.weight)
    b = float(model.bias)
    y_nums = w * x_nums + b 
plt.plot(x_nums, y_nums, label='Fitted Line', color='red')
plt.title("Linear Fit using Gradient Descent")
plt.legend()
plt.show()