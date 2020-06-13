from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt
plt.style.use("tableau-colorblind10")

def error_metrics(y_true, y_pred, show=False):
    """Calculate the error value of estimation task."""
    mse = MSE(y_true, y_pred)
    rmse = np.around(np.sqrt(mse), 4)
    
    mae = np.around(MAE(y_true, y_pred), 4)
    
    if show == True:
        print(f"RMSE: {rmse}")
        print(f"MAE:  {mae}")
    
    return rmse, mae

def plot_results(x, y_true, y_meas, y_preds, title, ekf=False, save_loc=False):
    """Plot the signal, the estimated signal, and the measured data."""
    plt.figure(figsize=(10, 7))
    plt.plot(x, y_meas, ".", label="measurements", markersize=12)
    plt.plot(x, y_true, "g-", label="signal", linewidth=2)
    if ekf==True:
        plt.plot(x, y_preds, "r-", label="estimation (EKF+NN)", linewidth=2)
    else:
        plt.plot(x, y_preds, "r-", label="estimation (NN)", linewidth=2)
    plt.title(title, fontsize=22)
    plt.legend(fontsize=20, loc="lower right")
    plt.grid()
    if save_loc:
        plt.savefig(save_loc, format="pdf", bbox_inches='tight')
    plt.show()

def get_nn_weights(nn):
    """Extracts the weights from a pytorch neural network."""
    weights = [param.data.flatten() 
               for name, param in nn.named_parameters()]

    return torch.cat(weights, dim=0).view(-1, 1)

def get_nn_weights_grad(nn):
    """Extracts the gradient of the weights of a pytorch neural network."""
    weight_grads = [param.grad.flatten() 
                    for name, param in nn.named_parameters()]
        
    return torch.cat(weight_grads, dim=0).view(-1, 1)

def set_nn_weights(nn, nn_weights):
    """
    Allows you to set custom weights to a pytorch neural network.
    """
    idx = 0
    for name, param in nn.named_parameters():
        selected_weights = nn_weights[idx : idx + 
                                torch.numel(param.data)]
        param.data = selected_weights.view(param.data.shape)
        idx = torch.numel(param.data)