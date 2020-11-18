#!/usr/bin/env python

# -------------------------------------------------- #
#
# Author: Ryne Burden 
#
# Description:
#     - This script contains the base class for staley
#
#     - This is mostly used in the static_picks and fluid_picks scripts 
#
#     - This is also called from staley_retrain.py
#
# Tested on Python 3.8.1
#
# -------------------------------------------------- #

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer 1 linear regression
        self.hidden1 = nn.Linear(16,9)
        self.act1 = nn.ReLU()

        # Inputs to hidden layer 2
        self.hidden2 = nn.Linear(8,4)
        self.act2 = nn.ReLU()

        # Output layer, 2 nodes - one for probability of home / away win percentage
        self.output = nn.Linear(9,2)

        # Define sigmoid activation 
        #self.act3 = nn.Sigmoid()
        self.act3 = nn.Sigmoid()

    def forward(self, x):

        # Pass the input tensor through each of the layers
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.output(x)
        x = self.act3(x)

        return x

# Training data is a np array
def train_network(network, trainingData, numEpochs, optimizer):
    
    # Define optimizer and loss function
    loss_function = nn.BCELoss()

    # Variables to return
    running_loss = list()

    # Shuffle data before evaluation to ensure 
    np.random.default_rng().shuffle(trainingData)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, .65)

    # Enumerate epochs
    for epoch in range(numEpochs):
    
        # Enumerate data set
        for x in range(trainingData.shape[0]):

            # Take the data from the given row
            row = trainingData[x,:]

            # Store the expected value
            expectedValue = int(row[-1])

            # Data without classification value
            newRow = np.ndarray(shape=(1,trainingData.shape[1] - 1))
            
            # Assign data from row to newRow to leave out classification
            for x in range(row.size - 1):
                newRow[0,x] = row[x]

            # Make newRow into a tensor
            newRow = torch.Tensor(newRow)
            
            # Clear optimizer gradients
            optimizer.zero_grad()
            
            # Compute model output
            outputs = network.forward(newRow)

            # Make an NDarray with the tensor data
            outputND = outputs.detach().numpy()

            # Expected value from original data using a one-hot encoding
            expected = torch.zeros(1,2)
            expected[0][expectedValue] = 1

            # calculate loss
            loss = loss_function(outputs, expected)

            # Backwards propagation
            loss.backward()

            # Update model weights
            optimizer.step()

            scheduler.step()

        np.random.default_rng().shuffle(trainingData, axis=0)

        # Append the loss to the list to return at the end of each epoch
        running_loss.append(loss.detach().numpy())

    # Set state parameters for the network being trained
    state = {
        'epoch': numEpochs,
        'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    # Return variables
    return running_loss, state

def predict(row, model):
    
    # Convert row to data
    row = torch.Tensor([row])
    
    # make prediction
    yhat = model.forward(row)
    
    # Retrieve numpy array
    yhat = yhat.detach().numpy()

    # Variable to hold the result - 0 for away win 1 for home win
    result = np.argmax(yhat)

    # Return result of prediction and probability of that result
    return result
