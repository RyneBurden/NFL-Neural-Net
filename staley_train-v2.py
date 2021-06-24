#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn import preprocessing
from scipy.signal import savgol_filter

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        #layers = [29, 22, 16, 12, 6, 2]
        #layers = [29, 20, 13, 7, 2]
        layers = [29, 26, 20, 15, 10, 5, 2]


        # Inputs layer 
        self.input_layer = nn.Linear(layers[0], layers[1])
        self.input_activation = nn.LeakyReLU()
        self.input_layer.bias = torch.nn.Parameter(torch.Tensor([.01 for i in range(layers[1])]))
        nn.init.kaiming_uniform_(self.input_layer.weight)

        # Hidden layer 1 to hidden layer 2
        self.hidden_1 = nn.Linear(layers[1], layers[2])
        self.hidden_1_activation = nn.LeakyReLU()
        self.hidden_1.bias = torch.nn.Parameter(torch.Tensor([.01 for i in range(layers[2])]))
        nn.init.kaiming_uniform_(self.hidden_1.weight)

        # Hidden layer 2 to hidden layer 3
        self.hidden_2 = nn.Linear(layers[2], layers[3])
        self.hidden_2_activation = nn.LeakyReLU()
        self.hidden_2.bias = torch.nn.Parameter(torch.Tensor([.01 for i in range(layers[3])]))
        nn.init.kaiming_uniform_(self.hidden_2.weight)

        self.hidden_3 = nn.Linear(layers[3], layers[4])
        self.hidden_3_activation = nn.LeakyReLU()
        self.hidden_3.bias = torch.nn.Parameter(torch.Tensor([.01 for i in range(layers[4])]))
        nn.init.kaiming_uniform_(self.hidden_3.weight)

        # hidden layer 4 to output
        self.hidden_4 = nn.Linear(layers[4], layers[5])
        self.hidden_4_activation = nn.LeakyReLU()
        self.hidden_4.bias = torch.nn.Parameter(torch.Tensor([.01 for i in range(layers[5])]))
        nn.init.kaiming_uniform_(self.hidden_4.weight)

        self.hidden_5 = nn.Linear(layers[5], layers[6])
        self.hidden_5_activation = nn.Softmax(dim=0)
        self.hidden_5.bias = torch.nn.Parameter(torch.Tensor([0 for i in range(layers[6])]))

    def forward(self, x):

        # Pass the input tensor through each of the layers
        x = self.input_layer(x)
        x = self.input_activation(x)
        x = self.hidden_1(x)
        x = self.hidden_1_activation(x)
        x = self.hidden_2(x)
        x = self.hidden_2_activation(x)
        x = self.hidden_3(x)
        x = self.hidden_3_activation(x)
        x = self.hidden_4(x)
        x = self.hidden_4_activation(x)
        x = self.hidden_5(x)
        x = self.hidden_5_activation(x)

        return x

def train_network(network, training_data, valid_data, num_epochs, learning_rate, scaler):

    if torch.cuda.is_available():
        my_device = "cuda"
    else:
        my_device = "cpu"

    my_device = "cpu"

    # min-max scaler for scaling batch data
    #scaler = preprocessing.StandardScaler()
    #training_data = pd.DataFrame(scaler.fit_transform(training_data.to_numpy()))
    #valid_data = pd.DataFrame(scaler.fit_transform(valid_data.to_numpy()))

    # Batch size variable
    batch_size = 16

    # Define optimizer and loss function
    loss_function = nn.BCELoss()

    # Variables to hold our losses
    running_train_loss = list()  # for loss during each batch
    total_train_loss = list()
    total_valid_loss = list()
    running_valid_loss = list()

    # List to hold accuracy for each epoch
    train_accuracy = list()
    valid_accuracy = list()

    # Learning Rate Scheduler
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(network.parameters(), lr = learning_rate, momentum = 0.8, nesterov = True)
    #optimizer = torch.optim.Adagrad(network.parameters(), lr = learning_rate)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", verbose = True, factor = 0.1, cooldown = 10)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epochs / 3), int(num_epochs / 1.5)], gamma=.01)

    # Enumerate epochs
    for epoch in tqdm(range(int(num_epochs))):

        batch_index = 0

        running_train_loss.clear()
        running_valid_loss.clear()

        # Variable to hold correct predictions
        correct_training = 0
        correct_valid = 0

        # Shuffle data before evaluation to ensure the data doesn't develop biases
        training_data = shuffle(training_data)

        for x in range(int(training_data.shape[0] / batch_size)):

            train_batch_outputs = torch.zeros(16, 2, requires_grad=False)
            train_batch_expected = torch.zeros(16, 2, requires_grad=False)
            batch_counter = 0

            # Scale current batch data except for the last value
            current_batch = training_data.iloc[batch_index:(batch_index + 16)]
            current_batch_pts_diff = current_batch.values[:, -1]
            current_batch = current_batch.values[:, :-1]
            current_batch_scaled = scaler.fit_transform(current_batch)
            # print(current_batch.shape)

            # Enumerate data set
            for current_row in range(batch_size):

                # Take the data from the given row
                current_data = current_batch_scaled[current_row]
                current_pts_diff = current_batch_pts_diff[current_row]

                # Set the expected win/loss value
                if current_pts_diff <= 0:
                    expected_value = 0
                elif current_pts_diff > 0:
                    expected_value = 1

                # Make newRow into a tensor
                current_data = torch.Tensor(current_data)

                # Clear optimizer gradients
                optimizer.zero_grad()

                # Compute model output
                outputs = network.forward(current_data)
                train_batch_outputs[batch_counter] = outputs

                # Expected value from original data using a one-hot encoding
                expected = torch.zeros(2)
                expected[expected_value] = 1
                train_batch_expected[batch_counter] = expected

                if expected[np.argmax(outputs.detach().numpy())] == 1:
                    correct_training += 1

                batch_counter = batch_counter + 1

            #print(train_batch_expected.detach().numpy())
            #print(train_batch_outputs.detach().numpy())

            # calculate loss
            current_loss = loss_function(train_batch_outputs, train_batch_expected)

            # Backwards propagation
            current_loss.backward()

            # Update model weights
            # scheduler.step()
            optimizer.step()
            #scheduler.step()

            # Append the loss to the list to return at the end of each epoch
            running_train_loss.append(current_loss.detach().numpy())

            # Increment our batch
            batch_index = batch_index + 16
            if batch_index > training_data.shape[0]:
                break

        batch_index = 0

        valid_data = shuffle(valid_data)

        # Validation data
        for x in range(int(valid_data.shape[0] / batch_size)):

            valid_batch_outputs = torch.zeros(16, 2, requires_grad=False)
            valid_batch_expected = torch.zeros(16, 2, requires_grad=False)
            batch_counter = 0

            # Scale current batch data except for the last value
            current_batch = valid_data.iloc[batch_index:(batch_index + 16)]
            current_batch = current_batch.values[:, :-1]
            current_batch_scaled = scaler.fit_transform(current_batch)
            # print(current_batch.shape)

            # Enumerate data set
            for current_row in range(batch_size):

                # Take the data from the given row
                current_data = current_batch_scaled[current_row]
                current_pts_diff = valid_data.iloc[batch_index + current_row].values[-1]

                # Set the expected win/loss value
                if current_pts_diff < 0:
                    expected_value = 0
                elif current_pts_diff > 0:
                    expected_value = 1

                # Make newRow into a tensor
                current_data = torch.Tensor(current_data)

                # Compute model output
                outputs = network.forward(current_data)
                valid_batch_outputs[batch_counter] = outputs

                # Expected value from original data using a one-hot encoding
                expected = torch.zeros(2)
                expected[expected_value] = 1
                valid_batch_expected[batch_counter] = expected

                if expected[np.argmax(outputs.detach().numpy())] == 1:
                    correct_valid += 1

                batch_counter = batch_counter + 1

            # calculate loss
            current_loss = loss_function(valid_batch_outputs, valid_batch_expected)

            # Append the loss to the list to return at the end of each epoch
            running_valid_loss.append(current_loss.detach().numpy())

            # Increment our batch
            batch_index = batch_index + 16
            if batch_index > valid_data.shape[0]:
                break

        #scheduler.step(sum(running_valid_loss) / len(running_valid_loss))

        total_train_loss.append(sum(running_train_loss) / len(running_train_loss))
        total_valid_loss.append(sum(running_valid_loss) / len(running_valid_loss))

        train_accuracy.append(correct_training / training_data.shape[0])
        valid_accuracy.append(correct_valid / valid_data.shape[0])

    # Set state parameters for the network being trained
    state = {
        'epoch': num_epochs,
        'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    # Return variables
    return total_train_loss, train_accuracy, total_valid_loss, valid_accuracy, state

def predict(row, model):

    # make prediction
    yhat = model.forward(torch.Tensor(row))

    # Retrieve numpy array
    yhat = yhat.detach().numpy()

    # Variable to hold the result - 0 for away win 1 for home win
    result = np.argmax(yhat)

    # Return result of prediction and probability of that result
    return result

def net_main(data, learning_rate, number_epochs):

    data = shuffle(data)

    training_index = data.shape[0] - 512
    valid_index = training_index + 256
    testing_index = valid_index + 256

    training_data = data.iloc[0:training_index]
    valid_data = data.iloc[training_index:valid_index]
    testing_data = data.iloc[valid_index:testing_index]

    #scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MaxAbsScaler()
    #testing_data = pd.DataFrame(scaler.fit_transform(testing_data.to_numpy()))

    staley = Network()

    epoch_list = list()

    training_loss, training_accuracy, valid_loss, valid_accuracy, state = train_network(staley, training_data, valid_data, number_epochs, learning_rate, scaler)

    for x in range(len(training_loss)):
        epoch_list.append(x)

    staley.load_state_dict(state['state_dict'])
    staley.eval()


    for x in range(10):

        testing_data = pd.DataFrame(shuffle(testing_data.to_numpy()))

        correct_testing = 0
        batch_index = 0
        # measure validation data
        for current_index in range(int(testing_data.shape[0] / 16)):

            current_batch_data = testing_data.iloc[batch_index:(batch_index+16)].values
            current_batch_data = scaler.fit_transform(current_batch_data)

            for x in range(current_batch_data.shape[0]):

                current_row = current_batch_data[x]

                if current_row[29] > 0:
                    expected = 1
                else:
                    expected = 0

                current_row = current_row[:-1]

                current_pass = predict(current_row, staley)

                if current_pass == expected:
                    correct_testing += 1

            batch_index = batch_index + 16

        print(str(round((correct_testing / testing_data.shape[0]) * 100, 2)) + "% of testing data correctly predicted")

    training_loss_smoothed = savgol_filter(training_loss, 15, 2).tolist()
    valid_loss_smoothed = savgol_filter(valid_loss, 15, 2).tolist()
    training_accuracy_smoothed = savgol_filter(training_accuracy, 15, 2).tolist()
    valid_accuracy_smoothed = savgol_filter(valid_accuracy, 15, 2).tolist()

    plt.plot(epoch_list, training_loss, linewidth = 0.25, linestyle='dashed', color='r', alpha=0.95)
    plt.plot(epoch_list, training_loss_smoothed, linewidth=1.0, color='r', label = "Training Loss")
    plt.plot(epoch_list, valid_loss, linewidth=0.25, linestyle='dashed', color='g', alpha=0.95)
    plt.plot(epoch_list, valid_loss_smoothed, linewidth=1.0, color='g', label="Validation Loss")
    plt.plot(epoch_list, training_accuracy, linewidth=0.25, linestyle='dashed', color='b', alpha=0.95)
    plt.plot(epoch_list, training_accuracy_smoothed, linewidth=1.0, color='b', label = "Training Accuracy")
    plt.plot(epoch_list, valid_accuracy, linewidth = 0.25, linestyle='dashed', color='c', alpha=0.95)
    plt.plot(epoch_list, valid_accuracy_smoothed, linewidth=1.0, color='c', label = "Validation Accuracy")
    plt.xlim(left = 0)
    plt.xlim(right = int(number_epochs))
    plt.ylim(bottom = 0)
    plt.ylim(top = 1)
    plt.xlabel('Number Of Epochs')
    plt.ylabel('Binary Cross Entropy Avg/Classification Accuracy %')
    plt.suptitle(str(number_epochs) + ' training epochs using data from the 1999-2020 NFL seasons')
    plt.title(str(learning_rate) + " learning rate")
    plt.legend()
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    to_save = input("Would you like to save the network state? ")
    #to_save = "no"

    if to_save == 'yes':
        save_name = input("What would you like to name the network state? ")
        save_path =  str(save_name) + '.staley'
        torch.save(state, save_path)

def lr_eval(data):

    num_epochs = 1
    lr_max = .1
    lr_mod = 10
    lr = .000000001

    model = Network()

    scaler = preprocessing.QuantileTransformer(output_distribution="normal")

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.BCELoss()

    loss_list = list()
    lr_list = list()
    learning_rates = list()
    current_lr = .000000001

    while True:
        if current_lr >= lr_max:
            break
        learning_rates.append(current_lr)

        current_lr = current_lr * lr_mod

    for x in range(10):
        while (lr < lr_max):

            data = shuffle(data)

            current_batch = data.iloc[0:16].values
            current_batch = scaler.fit_transform(current_batch)

            running_batch_loss = list()
            running_batch_expected = torch.zeros(16, 2, requires_grad=False)
            running_batch_outputs = torch.zeros(16, 2, requires_grad=False)

            for batch_index in range(16):

                current_row = current_batch[batch_index]

                if current_row[-1] > 0:
                    current_expected = 1
                else:
                    current_expected = 0

                current_expected_tensor = torch.zeros(2)
                current_expected_tensor[current_expected] = 1
                running_batch_expected[batch_index] = current_expected_tensor


                current_row = (current_row[:-1])

                #optimizer.zero_grad()

                current_output = model.forward(torch.Tensor(current_row))
                running_batch_outputs[batch_index] = current_output

            #print(running_batch_expected)
            current_loss = loss_function(running_batch_outputs, running_batch_expected)

            current_loss.backward()

            #optimizer.step()

            loss_list.append(current_loss.detach().numpy())

            lr_list.append(lr)
            lr = lr * lr_mod


    print(learning_rates)
    plt.scatter(lr_list, loss_list)
    plt.xlabel("Learning Rate")
    plt.ylabel("BCE Loss")
    #plt.ticklabel_format(axis='x', style='sci', scilimits=(.000000001, .1))
    plt.xlim(left=.00001)
    plt.xlim(right=.01)
    plt.xticks(learning_rates)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

def test_staley(testing_data):

    testing_data = testing_data.iloc[0:4224]
    print(testing_data.shape)

    staley_test = torch.load('test.staley')

    model = Network()

    model.load_state_dict(staley_test['state_dict'])
    model.eval()

    scaler = preprocessing.MaxAbsScaler()

    correct_testing = 0
    batch_index = 0

    for epoch in range(10):

        testing_data = pd.DataFrame(shuffle(testing_data.to_numpy()))
        batch_index = 0
        correct_testing = 0

        # measure testing data
        for current_index in range(int(testing_data.shape[0] / 16)):

            current_batch_data = testing_data.iloc[batch_index:(batch_index + 16)].values
            current_batch_data = scaler.fit_transform(current_batch_data)

            for x in range(current_batch_data.shape[0]):

                current_row = current_batch_data[x]

                if current_row[29] > 0:
                    expected = 1
                else:
                    expected = 0

                current_row = current_row[:-1]

                current_pass = predict(current_row, model)

                if current_pass == expected:
                    correct_testing += 1

            batch_index = batch_index + 16

        print(str(round((correct_testing / testing_data.shape[0]) * 100, 2)) + "% of testing data correctly predicted")