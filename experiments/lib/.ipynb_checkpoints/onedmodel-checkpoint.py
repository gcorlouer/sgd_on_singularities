import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset

class TrivialModel(torch.nn.Module):
    """
    Trivial, single parameter model
    """
    def __init__(self) -> None:
        super(TrivialModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn([1]))

    def forward(self, input:Tensor):
        return input * 0.

# Defining the Linear Model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # Random weight initialization
        self.linear.weight.data.normal_()
    def forward(self, x):
        return self.linear(x)


class SingModel(torch.nn.Module):
    def __init__(self, w0:float, d1:int, d2:int, in_features:int, cst: float,
                 out_features:int, w_init: Optional[Tensor] = None) -> None:
         super(SingModel, self).__init__()
         self.w0 = w0
         self.d1 = d1
         self.d2 = d2
         self.cst = cst
         if w_init is not None:
            self.weight = torch.nn.Parameter(w_init)
         else:
            self.weight = torch.nn.Parameter(torch.randn((out_features, in_features)))

         
    def forward(self, input:Tensor):
        # Chose cst1 and cst2 such that K(1) = 0 and K'(1) = cst
        cst1 = - self.cst**(1/3)/2**(1/3)
        cst2 = (16*self.cst - 2**(2/3)*self.cst**(4/3))/64
        sing1 = (self.weight + self.w0)**self.d1
        sing2 = torch.sqrt((self.weight - self.w0 + cst1)**(2*self.d2) + cst2)
        return input * sing1 * sing2

# Defining the Linear Model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
        # Random weight initialization
        self.linear.weight.data.normal_()
    def forward(self, x):
        return self.linear(x)
    
# Training model

def train_model(model, data_loader, w_init: Optional[Tensor] = None, 
                linear=True, num_epochs = 1000, lr=0.01):
    # Loss tracking
    running_loss = []

    # Tracking weights
    if w_init is None:
         weights_over_epochs = []
    else:
         weights_over_epochs = [w_init.item()]

    # Loss and Optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training the Model
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(data_loader):
            # Forward pass
            y_pred = model(batch_x)
            loss = loss_function(y_pred, batch_y)
#             print("weight", model.weight.item())

            if torch.isnan(loss):
                print(f"iteration {i}\n================\n")

                print("batch_x", batch_x)
                print("batch_y", batch_y)
                print("y_pred", y_pred)
                print("weight", model.weight.item())

            # Backward pass and optimization
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Tracking the cumulative loss for the current epoch
            epoch_loss += loss.item()

        # Tracking loss and weight along epochs
        # Calculate average loss for the current epoch
        avg_epoch_loss = epoch_loss / len(data_loader)
        running_loss.append(avg_epoch_loss)
        if linear == True:
            current_weight = model.linear.weight.item()
        else:
            current_weight = model.weight.item() 
        weights_over_epochs.append(current_weight)

#         if epoch % 10 == 0:
#                         print(f'Epoch {epoch}, Loss: {loss.item()}, w: {current_weight}')
    return running_loss, weights_over_epochs

# Plot loss curve

def plot_loss_curve(loss_values):
    plt.plot(loss_values)
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

# Plot loss landscape

def plot_loss_landscape(ax, model, x_data, y_data,
                        weight_range, weights_over_epochs, linear=False):

    # Computing the loss for each weight value in the range
    loss_landscape = []
    loss_function = nn.MSELoss()

    for weight in weight_range:
        # Updating the model's weight and fixed bias
        if linear == True:
            model.linear.weight.data.fill_(weight)
        else:
            model.weight.data.fill_(weight) 
        
        # Forward pass with the updated weight
        y_pred = model(x_data)
        
        # Computing the loss
        loss = loss_function(y_pred, y_data)
        loss_landscape.append(loss.item())

    # Plotting the loss landscape
    
    ax.plot(weight_range, loss_landscape, label='Loss Landscape')

    # Plotting the trajectory of the weights during training
    ax.plot(weights_over_epochs, [loss_landscape[np.argmin(np.abs(weight_range - w))] for w in weights_over_epochs], 'ro-', label='Training Path', markersize=3)
    ax.set_title('Weight Dynamics on Loss Landscape')
#     ax.set_yscale("log")
    ax.set_xlabel('Weight')
    ax.set_ylabel('Loss')
