import numpy as np
import matplotlib.pyplot as plt
import _1010_SGD as sgd

w, b, losses, w_s, b_s, dw_s, db_s = sgd.run()

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