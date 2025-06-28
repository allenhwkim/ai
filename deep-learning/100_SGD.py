"""
This file demonstrates Stochastic Gradient Descent (SGD) for linear regression. 

 - Gradient Descent is an optimization algorithm that updates parameters 
   (w and b) to minimize a loss function (here, Mean Squared Error).
 - In this code, for each epoch, the gradients of the loss with respect to w
   and b are computed using all data points 
   (so technically, this is Batch Gradient Descent; if you used one data point at a time, it would be true SGD).
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

# --- Gradient Descent Loop ---
for epoch in range(100):
    # type of errors is np.ndarray
    errors = (w * x_nums + b) - y_nums # MSE error: (w*x + b) - y
    loss = np.mean(errors ** 2) 
    losses.append(loss)

    # Compute gradient
    # loss function L = (1/n) * Σ( (w*x + b) - y )^2
    # Similar to standard deviation s = √[ Σ (xi - x̄)² / (n - 1) ]
    # MSE measures the average squared difference between predicted values and actual values.
    # SD measures the spread of the data around its mean (how much the data varies).
    # Derivatives of L/w, dw = (1/n) * Σ(2 * (w*x + b - y) * x)
    # Devivatives of L/b, db = (1/n) * Σ(2 * (w*x + b - y))
    dw = np.mean(errors * x_nums) * 2.0
    db = np.mean(errors) * 2.0          
    print(f"w = {w:.4f}\t b = {b:.4f}\t dw = {dw:.4f}\t db = {db:.4f}")

    w_s.append(w)
    b_s.append(b)
    dw_s.append(dw)
    db_s.append(db)

    # Update weights and bias by substracting the gradients scaled by the learing rate
    # If dw (the gradient with respect to w) is large, you subtract a larger amount from w.
    # If dw is small, you subtract a smaller amount from w.
    # In other words, the bigger the gradient, the bigger the update.
    # This helps the parameter w move quickly when the error is large, and more slowly as 
    # it gets closer to the optimal value. The learning rate is a small constant that
    # controls how much we adjust the parameters.
    w = w - (learning_rate * dw)
    b = b - (learning_rate * db)
  
# --- Plotting the Loss ---
plt.plot(losses)
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.grid()
plt.show()

# --- Plotting the Results ---
plt.scatter(x_nums, y_nums, color='blue', label='Data Points')
x_nums = np.linspace(min(x_nums), max(x_nums), 100, dtype=np.float32)
y_nums = w * x_nums + b  # Calculate the predicted y values
plt.plot(x_nums, y_nums, color='red', label='Fitted Line')
plt.title("Linear Fit using Gradient Descent")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# --- Plot loss function for w ---
w_values = np.linspace(w - 5, w + 5, 200)
loss_w = []
b_fixed = b  # Use the final value of b

for w_test in w_values:
    y_pred_test = w_test * x_nums + b_fixed
    error_test = y_pred_test - y_nums
    loss_test = np.mean(error_test ** 2)
    loss_w.append(loss_test)


plt.figure()
plt.plot(w_values, loss_w, label='Loss vs w')
plt.scatter([w], [np.mean((w * x_nums + b_fixed - y_nums) ** 2)], color='red', label=f'Current w={w:.3f}')
plt.title("Loss Function with respect to w")
plt.xlabel("w")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid()
plt.show()

# --- Plotting the history of parameters and gradients ---
epochs = np.arange(len(w_s))
plt.figure(figsize=(10, 6))
plt.scatter(epochs, w_s, color='blue', label='w history', s=15)
plt.scatter(epochs, b_s, color='green', label='b history', s=15)
plt.scatter(epochs, dw_s, color='orange', label='dw history', s=15)
plt.scatter(epochs, db_s, color='red', label='db history', s=15)
plt.title("Parameter and Gradient History during Training")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()