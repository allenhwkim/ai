"""
This file demonstrates Stochastic Gradient Descent (SGD) for linear regression. 

 - Gradient Descent is an optimization algorithm that updates parameters 
   (w and b) to minimize a loss function (here, Mean Squared Error).
 - In this code, for each epoch, the gradients of the loss with respect to w
   and b are computed using one data point at a time.
 - The parameters w and b are updated by moving in the direction opposite to the gradient, 
   scaled by the learning rate.
 - The process is repeated for multiple epochs, and the loss is tracked and plotted.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Data ---
x_nums = np.array([-1.5, -0.8, 0.1, 0.9, 1.7])
y_nums = np.array([0.3, -0.3, 0.5, 1.8, 1.5])

# --- Initialization ---
w = -10.0  # Initial slope
b = 10.0   # Initial bias

learning_rate = 0.1

# --- Store Loss for Plotting ---
losses = []
w_s, b_s, dw_s, db_s = [], [], [], []

# --- SGD Loop (one random data point per epoch) ---
for epoch in range(100):
    # Randomly select one data point
    idx = np.random.randint(len(x_nums))
    x_i = x_nums[idx]
    y_i = y_nums[idx]

    # Compute prediction and error for this data point
    error = (w * x_i + b) - y_i
    loss = error ** 2
    losses.append(loss)

    # Compute gradients for this data point
    dw = 2 * error * x_i
    db = 2 * error

    print(f"w = {w:.4f}\t b = {b:.4f}\t dw = {dw:.4f}\t db = {db:.4f}")

    w_s.append(w)
    b_s.append(b)
    dw_s.append(dw)
    db_s.append(db)

    # Update weights and bias
    w = w - (learning_rate * dw)
    b = b - (learning_rate * db)
  
# --- Prepare data for plotting ---
x_data = np.array([-1.5, -0.8, 0.1, 0.9, 1.7])
y_data = np.array([0.3, -0.3, 0.5, 1.8, 1.5])
x_fit = np.linspace(min(x_data), max(x_data), 100, dtype=np.float32)
y_fit = w * x_fit + b

w_values = np.linspace(w - 5, w + 5, 200)
loss_w = []
b_fixed = b  # Use the final value of b

for w_test in w_values:
    y_pred_test = w_test * x_fit + b_fixed
    error_test = y_pred_test - (w_test * x_fit + b_fixed)
    loss_test = np.mean((w_test * x_fit + b_fixed - (w * x_fit + b_fixed)) ** 2)
    loss_w.append(loss_test)
loss_w = [np.mean((w_ * x_data + b_fixed - y_data) ** 2) for w_ in w_values]

epochs = np.arange(len(w_s))

# --- Plot all in a single window ---
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. Loss over Epochs
axs[0, 0].plot(losses)
axs[0, 0].set_title("Loss over Epochs")
axs[0, 0].set_xlabel("Epochs")
axs[0, 0].set_ylabel("Loss (MSE)")
axs[0, 0].grid()

# 2. Linear Fit using Gradient Descent
axs[0, 1].scatter(x_data, y_data, color='blue', label='Data Points')
axs[0, 1].plot(x_fit, y_fit, color='red', label='Fitted Line')
axs[0, 1].set_title("Linear Fit using Gradient Descent")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("y")
axs[0, 1].legend()
axs[0, 1].grid()

# 3. Loss Function with respect to w
axs[1, 0].plot(w_values, loss_w, label='Loss vs w')
axs[1, 0].scatter([w], [np.mean((w * x_data + b_fixed - y_data) ** 2)], color='red', label=f'Current w={w:.3f}')
axs[1, 0].set_title("Loss Function with respect to w")
axs[1, 0].set_xlabel("w")
axs[1, 0].set_ylabel("Loss (MSE)")
axs[1, 0].legend()
axs[1, 0].grid()

# 4. Parameter and Gradient History during Training
axs[1, 1].scatter(epochs, w_s, color='blue', label='w history', s=15)
axs[1, 1].scatter(epochs, b_s, color='green', label='b history', s=15)
axs[1, 1].scatter(epochs, dw_s, color='orange', label='dw history', s=15)
axs[1, 1].scatter(epochs, db_s, color='red', label='db history', s=15)
axs[1, 1].set_title("Parameter and Gradient History during Training")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Value")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout()
plt.show()