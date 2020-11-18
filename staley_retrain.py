#!/usr/bin/env python

# -------------------------------------------------- #
#
# Author: Ryne Burden
#
# Description:
#     - This file is used for retraining fluid models each week on games from the previous week
#
# Tested on Python 3.8.1
#
# -------------------------------------------------- #

import torch
import staley_base
import openpyxl
import numpy as np
import os
import sys 
from sklearn import preprocessing

# Take info from the command line
weekNum = sys.argv[1]
currentSeason = sys.argv[2]

print("** Retraining the fluid models on week " + str(int(weekNum) - 1) + " games **\n")

# Load the games from the previous week to train with
trainingData = np.loadtxt('Data/Data Sets/' + str(currentSeason) + '/Training Data/' + str(currentSeason) + '_Week_' + str(int(weekNum) - 1) + '.txt')

# Define the number of training epochs for retraining
# The original model was trained 168 epochs on a data set of 512 games
# This comes to a ratio of ~32%, so that's where the .32 comes from
# This makes the retraining period similar to the original training period
trainingEpochs = round(trainingData.shape[0] * .32)

# Load current fluid model states
state1 = torch.load('Data/Network States/Fluid Models/fluid_model_1.staley')
state2 = torch.load('Data/Network States/Fluid Models/fluid_model_2.staley')
state3 = torch.load('Data/Network States/Fluid Models/fluid_model_3.staley')
state4 = torch.load('Data/Network States/Fluid Models/fluid_model_4.staley')
state5 = torch.load('Data/Network States/Fluid Models/fluid_model_5.staley')

# Create Network objects to load states into
model1 = staley_base.Network()
model2 = staley_base.Network()
model3 = staley_base.Network()
model4 = staley_base.Network()
model5 = staley_base.Network()

# Load network and optimizer state_dicts for all 5 models
model1.load_state_dict(state1['state_dict'])
optimizer1 = torch.optim.Adam(model1.parameters(), lr = .0005)
optimizer1.load_state_dict(state1['optimizer'])

model2.load_state_dict(state2['state_dict'])
optimizer2 = torch.optim.Adam(model1.parameters(), lr = .0005)
optimizer2.load_state_dict(state2['optimizer'])

model3.load_state_dict(state3['state_dict'])
optimizer3 = torch.optim.Adam(model3.parameters(), lr = .0005)
optimizer3.load_state_dict(state3['optimizer'])

model4.load_state_dict(state4['state_dict'])
optimizer4 = torch.optim.Adam(model4.parameters(), lr = .0005)
optimizer4.load_state_dict(state4['optimizer'])

model5.load_state_dict(state5['state_dict'])
optimizer5 = torch.optim.Adam(model5.parameters(), lr = .0005)
optimizer5.load_state_dict(state5['optimizer'])

# Train and record the state after training for each model
loss1, state6 = staley_base.train_network(model1, trainingData, trainingEpochs, optimizer1)
loss2, state7 = staley_base.train_network(model2, trainingData, trainingEpochs, optimizer2)
loss3, state8 = staley_base.train_network(model3, trainingData, trainingEpochs, optimizer3)
loss4, state9 = staley_base.train_network(model4, trainingData, trainingEpochs, optimizer4)
loss5, state10 = staley_base.train_network(model5, trainingData, trainingEpochs, optimizer5)

# Save all states for future use
torch.save(state6, 'Data/Network States/Fluid Models/fluid_model_1.staley')
torch.save(state7, 'Data/Network States/Fluid Models/fluid_model_2.staley')
torch.save(state8, 'Data/Network States/Fluid Models/fluid_model_3.staley')
torch.save(state9, 'Data/Network States/Fluid Models/fluid_model_4.staley')
torch.save(state10, 'Data/Network States/Fluid Models/fluid_model_5.staley')

print("** Models retrained on week " + str(int(weekNum) - 1) + " games and ready to predict games for week " + str(weekNum) + " **\n")
