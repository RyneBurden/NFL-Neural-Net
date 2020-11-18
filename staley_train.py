#!/usr/bin/env python

# This script was used to train the 5 networks used for staley
# staleyBase.py is a derived and updated version of this file

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys
from tqdm import tqdm

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
        self.act3 = nn.Sigmoid()
        # Define sigmoid activation 
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        # Pass the input tensor through each of the layers
        x = self.hidden1(x)
        x = self.act1(x)
#        x = self.hidden2(x)
#        x = self.act2(x)
        x = self.output(x)
        x = self.act3(x)
        
        return x

# Training data is a np array
def train_network(network, trainingData, numEpochs):
    
    # Define optimizer and loss function
    loss_function = nn.BCELoss()
    #optimizer = torch.optim.SGD(network.parameters(), lr = .05, momentum = 0.9)
    optimizer = torch.optim.Adam(network.parameters(), lr = .00001)
    # Variables to return
    running_loss = list()

    np.random.default_rng().shuffle(trainingData)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, numEpochs) 

    # Enumerate epochs
    for epoch in tqdm(range(numEpochs)):

        scheduler.step()
        
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

            # Expected value from original data
            expected = torch.zeros(1,2)
            expected[0][expectedValue] = 1

            # calculate loss
            loss = loss_function(outputs, expected)

            # Backwards propagation
            loss.backward()

            # Update model weights
            optimizer.step()

        np.random.default_rng().shuffle(trainingData, axis=0)


        # Append the loss to the list to return at the end of each epoch
        running_loss.append(loss.detach().numpy())
        
    state = {
        'epoch': numEpochs,
        'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict()
        'lr_scheduler': scheduler.state_dict()
    }

    return running_loss, state

def predict(row, model):
    # Convert row to data
    row = torch.Tensor([row])
    # make prediction
    yhat = model.forward(row)
    # Retrieve numpy array
    yhat = yhat.detach().numpy()
    return np.argmax(yhat)

def main():

    # Variables taken from script call
    numEpochs = sys.argv[1]
    trainYear1 = sys.argv[2]
    trainYear2 = sys.argv[3]
    testYear = sys.argv[4]
    validYear = int(testYear) - 1
    
    # Set filepaths based on variables above
    trainingData1Path = 'Data/Data Sets/' + str(trainYear1) + '/' + str(trainYear1) + 'data.txt'
    trainingData2Path = 'Data/Data Sets/' + str(trainYear2) + '/' + str(trainYear2) + 'data.txt'
    testingDataPath = 'Data/Data Sets/' + str(testYear) + '/' + str(testYear) + 'data.txt'

    # Filepath for saving pyplot later
    lossPlotPath = 'Data/Plots/' + str(testYear) + 'test_' + str(numEpochs) + 'epochsLoss.png'    

    # Scale trainingData1 and trainingData2 in batches of 16 (16 games per week)
    
    # Load data into numpy arrays
    trainingData1 = np.loadtxt(trainingData1Path)
    trainingData2 = np.loadtxt(trainingData2Path)
    testingData = np.loadtxt(testingDataPath)
    trainingData = np.concatenate((trainingData1,trainingData2), axis=0)

    np.random.default_rng().shuffle(testingData)
    np.random.default_rng().shuffle(testingData)
    np.random.default_rng().shuffle(testingData)
    np.random.default_rng().shuffle(testingData)
    np.random.default_rng().shuffle(testingData)

    #Testing data sets with less data, this list will be used with np.delete on the training and testing data
    # toDelete = [3,4,11,12]
    # for index in toDelete:
    #     trainingData = np.delete(trainingData, index, axis=1)
    #     testingData = np.delete(testingData, index, axis=1)
        
    # Scale the data values between 0 and 1 using the MinMaxScaler() from scikitlearn.preprocessing
    #scaler = preprocessing.MinMaxScaler()
    #trainingData = scaler.fit_transform(trainingData)
    #testingData = scaler.fit_transform(testingData)
    #validationData = scaler.fit_transform(validationData)
    
    # Make the model
    model = Network()
    
    # train_network returns a list containing the loss found at the end of each epoch
    running_loss, networkState = train_network(model, trainingData, int(numEpochs))
    # make running_loss into a numpy array and perform argmin to find the index (and epoch) that had the lowest loss
    running_loss = np.array(running_loss)
    lowestLossIndex = np.argmin(running_loss)
    
    # List of sequential numbers the size of the number of epochs give
    # Pyplot requires a list be passed to represent axis data
    numberOfEpochs = list()
    for x in range(int(numEpochs)):
        numberOfEpochs.append(int(x))

    # Variable to count how many games the network correctly classifies
    correctPredictions = 0
    validationPredictionsCorrect = 0

    # Loop through the testing data testing one row at a time
    for x in range(testingData.shape[0]):
        # Numpy array that will be used to store the testing data without the result
        # This won't be needed when predicting future games
        newData1 = np.zeros(shape=(1,testingData.shape[1] - 1))
        # Assign data from testing set to newData
        for y in range(testingData.shape[1] - 1):
            newData1[0,y] = testingData[x,y]
        # Pass the given row (newData) to predict and store the result in prediction
        testPrediction = predict(newData1[0,:], model)
        #print('Expected=%d, Got=%d' % (testingData[x,-1], prediction))
        # Add 1 to correctPredictions when a match is made
        if testPrediction == testingData[x,-1]:
            correctPredictions += 1
            
    # Loop through validation data and test the network against it
    for week in range(1,11):
        # Validation data holds the game data for teams of a given week
        validationData = np.loadtxt('Data/Data Sets/' + str(testYear) + '/Validation Data/' + str(testYear) + '_Week_1' + 'v.txt')
        validationResults = np.loadtxt('Data/Data Sets/' + str(testYear) + '/' + str(testYear) + '_Week_1' + '.txt')

        np.random.default_rng().shuffle(validationData)

        # for index in toDelete:
        #     validationData = np.delete(validationData, index, axis=1)

        # Loop through validation data
        for x in range(validationResults.shape[0]):
            # Numpy array that will hold the current game being predicted
            newData2 = np.zeros(shape=(1,validationData.shape[1]))
            # Assign data from validation set to newData2
            for y in range(validationData.shape[1]):
                newData2[0,y] = validationData[x, y]
            # Pass the validation row to predict and store the result
            validationPrediction = predict(newData2[0,:], model)
            # print statement
            #print('Expected=%d\t Got=%d' % (validationResults[x,-1], validationPrediction))
            # Add one to correct if found
            if validationPrediction == validationResults[x,-1]:
                validationPredictionsCorrect += 1

        print(" " + str(validationPredictionsCorrect) + " (" + str((validationPredictionsCorrect / testingData.shape[0]) * 100) + "% ) of validation data games correclty predicted")
        
        validationPredictionsCorrect = 0

        
        
    #Print the prediction rate
    print("\n The network predicted " + str(correctPredictions) + " (" +  str(((correctPredictions / testingData.shape[0]) * 100)) + "%) of " + str(testYear) +  " Training Data")
    print('\n')

    # print("Lowest loss: " + str(running_loss[lowestLossIndex]) + " found in epoch " + str(lowestLossIndex))

    test = np.loadtxt('Data/Data Sets/2020/Validation Data/2020_Week_1v.txt')

    # for index in toDelete:
    #     test = np.delete(test, index, axis=1)
    
    testp = predict(test[0,:], model)
    
    # ----- BELOW WILL NEED TO BE CHANGED TO SHOW NUMBER CORRECT AND NUMBER INCORRECT ETC. AFTER TESTING EPOCHS AND NETWORK SIZE ----- #

    # Pass numberOfEpochs list and running_loss list to plt to plot
    plt.plot(numberOfEpochs, running_loss)
    # Format axes
    #  X axis
    plt.xlim(left = 0)
    plt.xlim(right = int(numEpochs))
    #  Y axis
    #plt.ylim(bottom = 0)
    #plt.ylim(top = 1.5)
    # Label the graph and axes
    plt.xlabel('Number Of Epochs')
    plt.ylabel('Binary Cross Entropy')
    plt.suptitle('Trained On ' + str(trainYear1) + '/' + str(trainYear2) + ' - Tested On ' + str(testYear) + ' - ' + str(correctPredictions) + ' Testing & ' + str(validationPredictionsCorrect) + ' Validation')
    # Annotate the point of lowest loss
    plt.annotate('Lowest loss: ' + str(running_loss[lowestLossIndex]) + ' @ epoch ' + str(lowestLossIndex), xy=(lowestLossIndex, running_loss[lowestLossIndex]), xytext=(int(numEpochs) / 2.5, .8), arrowprops=dict(arrowstyle="->", color="r"))
    # Save the plots based on the information given at 
    #plt.savefig(lossPlotPath)
    plt.show()
    
    toSave = input("Would you like to save the network state? ")

    if toSave == 'yes':
        saveName = input("What would you like to name the network state? ")
        savePath = 'Data/Network States/' + str(saveName) + '.staley'
        torch.save(networkState, savePath)

    

main()

