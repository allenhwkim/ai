"""
This file demonstrates Batch Gradient Descent (BGD) for linear regression.

 - Batch Gradient Descent is an optimization algorithm that updates parameters 
   (w and b) to minimize a loss function (here, Mean Squared Error).
 - In this code, for each epoch, the gradients of the loss with respect to w
   and b are computed using all data points (the entire batch).
 - The parameters w and b are updated by moving in the direction opposite to the gradient, 
   scaled by the learning rate.
 - The process is repeated for multiple epochs, and the loss is tracked and plotted.
"""

import numpy as np
import matplotlib.pyplot as plt
import _1020_BGD as bgd

w, b, losses, w_s, b_s, dw_s, db_s = bgd.run()

# Prepare data for plotting
x_data = np.array([-1.5, -0.8, 0.1, 0.9, 1.7])
y_data = np.array([0.3, -0.3, 0.5, 1.8, 1.5])
x_fit = np.linspace(min(x_data), max(x_data), 100, dtype=np.float32)
y_fit = w * x_fit + b
w_values = np.linspace(-12, 4, 100)
loss_w = [np.mean((w_ * x_data + b - y_data) ** 2) for w_ in w_values]

from matplotlib.widgets import Button

# --- Prepare for interactive plot ---
current_epoch = [0]  # Use list for mutability in closure

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(bottom=0.2)  # Make space for button

def update_plots(epoch):
    # 1. Loss over Epochs
    axs[0, 0].cla()
    axs[0, 0].plot(losses[:epoch+1])
    axs[0, 0].set_title("Loss over Epochs")
    axs[0, 0].set_xlabel("Epochs")
    axs[0, 0].set_ylabel("Loss (MSE)")
    axs[0, 0].grid()

    # 2. Linear Fit using Gradient Descent
    axs[0, 1].cla()
    axs[0, 1].scatter(x_data, y_data, color='blue', label='Data Points')
    w_epoch = w_s[epoch]
    b_epoch = b_s[epoch]
    y_fit_epoch = w_epoch * x_fit + b_epoch
    axs[0, 1].plot(x_fit, y_fit_epoch, color='red', label='Fitted Line')
    axs[0, 1].set_title("Linear Fit using Gradient Descent")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")
    axs[0, 1].legend()
    axs[0, 1].grid()

    # 3. Loss Function with respect to w
    axs[1, 0].cla()
    loss_w_epoch = [np.mean((w_ * x_data + b_epoch - y_data) ** 2) for w_ in w_values]
    axs[1, 0].plot(w_values, loss_w_epoch, label='Loss vs w')
    axs[1, 0].scatter([w_epoch], [np.mean((w_epoch * x_data + b_epoch - y_data) ** 2)], color='red', label=f'Current w={w_epoch:.3f}')
    axs[1, 0].set_title("Loss Function with respect to w")
    axs[1, 0].set_xlabel("w")
    axs[1, 0].set_ylabel("Loss (MSE)")
    axs[1, 0].legend()
    axs[1, 0].grid()

    # 4. Parameter and Gradient History during Training
    axs[1, 1].cla()
    epochs = np.arange(epoch+1)
    axs[1, 1].scatter(epochs, w_s[:epoch+1], color='blue', label='w history', s=15)
    axs[1, 1].scatter(epochs, b_s[:epoch+1], color='green', label='b history', s=15)
    axs[1, 1].scatter(epochs, dw_s[:epoch+1], color='orange', label='dw history', s=15)
    axs[1, 1].scatter(epochs, db_s[:epoch+1], color='red', label='db history', s=15)
    axs[1, 1].set_title("Parameter and Gradient History during Training")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Value")
    axs[1, 1].legend()
    axs[1, 1].grid()

    fig.canvas.draw_idle()

# Initial plot
update_plots(0)

# --- Add Button ---
ax_button = plt.axes([0.45, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Next Epoch')

def next_epoch(event):
    if current_epoch[0] < len(w_s) - 1:
        current_epoch[0] += 1
        update_plots(current_epoch[0])

button.on_clicked(next_epoch)
plt.show()