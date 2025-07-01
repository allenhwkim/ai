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

def run():
  x_nums = np.array([-1.5, -0.8, 0.1, 0.9, 1.7])
  y_nums = np.array([0.3, -0.3, 0.5, 1.8, 1.5])

  losses = []
  w = -10.0  # Initial slope
  b = 10.0   # Initial bias
  learning_rate = 0.1

  w_s, b_s, dw_s, db_s = [], [], [], []

  # --- SGD Loop (one random data point per epoch) ---
  for epoch in range(100):
      # Randomly select one data point
      idx = np.random.randint(len(x_nums))
      x_i = x_nums[idx]
      y_i = y_nums[idx]

      # Compute prediction and error for this data point
      # loss function L = (1/n) * Σ( (w*x + b) - y )^2
      # Similar to standard deviation s = √[ Σ (xi - x̄)² / (n - 1) ]
      # MSE measures the average squared difference between predicted values and actual values.
      # SD measures the spread of the data around its mean (how much the data varies).
      # Derivatives of L/w, dw = (1/n) * Σ(2 * (w*x + b - y) * x)
      # Devivatives of L/b, db = (1/n) * Σ(2 * (w*x + b - y))
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

  return w, b, losses, w_s, b_s, dw_s, db_s

run()