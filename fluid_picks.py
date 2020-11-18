#!/usr/bin/env python

# ------------------------------------------------------------------- #
#
# Author: Ryne Burden
#
# Description:
#     - This file writes picks made by the fluid models - those that are retrained each week 
#     
#     - It is called from staleySays.py based on user input, but can be called alone
#     
#     - This script also calls staleyRetrain.py if specified
#
# Tested on Python 3.8.1
#
# ------------------------------------------------------------------- #

import torch
import staley_base
import openpyxl
import numpy as np
import os
import sys 
from sklearn import preprocessing

# Input data to use when calling staleyRetrain and saving the correct files
weekNum = sys.argv[1]
dataYear = sys.argv[2]
retrain = sys.argv[3]

# Run staleyRetrain.py using an os call before loading the states
if retrain == 'y' and dataYear == '2020':
    os.system('./staleyRetrain.py ' + str(weekNum) + ' ' + str(dataYear))

# Load the fluid models
fluidState1 = torch.load('Data/Network States/Fluid Models/fluid_model_1.staley')
fluidState2 = torch.load('Data/Network States/Fluid Models/fluid_model_2.staley')
fluidState3 = torch.load('Data/Network States/Fluid Models/fluid_model_3.staley')
fluidState4 = torch.load('Data/Network States/Fluid Models/fluid_model_4.staley')
fluidState5 = torch.load('Data/Network States/Fluid Models/fluid_model_5.staley')

# Define Network() objects for fluid models
fluidModel1 = staley_base.Network()
fluidModel2 = staley_base.Network()
fluidModel3 = staley_base.Network()
fluidModel4 = staley_base.Network()
fluidModel5 = staley_base.Network()

# Load state_dict for each fluid model
fluidModel1.load_state_dict(fluidState1['state_dict'])
fluidModel2.load_state_dict(fluidState2['state_dict'])
fluidModel3.load_state_dict(fluidState3['state_dict'])
fluidModel4.load_state_dict(fluidState4['state_dict'])
fluidModel5.load_state_dict(fluidState5['state_dict'])

# Put all models in eval mode
fluidModel1.eval()
fluidModel2.eval()
fluidModel3.eval()
fluidModel4.eval()
fluidModel5.eval()

# Open the schedule workbook for the current season and set the current sheet
scheduleWB = openpyxl.load_workbook('Data/Schedule/' + dataYear + '/' + dataYear + '.xlsx')
currentSheet = scheduleWB.active

# Variables for use later
totalGames = currentSheet.max_row
numGames = 0

# This loop goes through the whole excel sheet and finds the total number of games for the given week
for x in range(1,totalGames + 1):

    # Variables to hold current week and away/home team
    weekCell = 'A' + str(x)
    awayTeam = 'B' + str(x)
    homeTeam = 'C' + str(x)

    # This conditional adds one to numGames if the week number on the schedule and the user-given year match
    if int(currentSheet[weekCell].value) == int(weekNum):
        numGames += 1

# Load testing data and open the output file based on the year given
# This will be changed after the 2020 season
if dataYear == "2020":

    testingData = np.loadtxt('Data/Data Sets/2020/Validation Data/2020_Week_' + str(weekNum) + 'v.txt')
    outputFile = open('Data/Staley Picks/2020/2020 data/Fluid Model/' + str(dataYear) + ' Week ' + str(weekNum) + 'f.txt', 'w')

elif dataYear == "2019":

    testingData = np.loadtxt('Data/Data Sets/2020/Validation Data/2019 Averages/2020_Week_' + str(weekNum) + 'v.txt')
    outputFile = open('Data/Staley Picks/2020/2019 data/Fluid Model/' + str(int(dataYear) + 1) + ' Week ' + str(weekNum) + 'f.txt', 'w')

# Dummy arrays to hold results for each model
model1results = np.zeros(numGames)
model2results = np.zeros(numGames)
model3results = np.zeros(numGames)
model4results = np.zeros(numGames)
model5results = np.zeros(numGames)

# Dummy arrays to hold probabilities for the results in the above arrays
model1probabilities = np.zeros(numGames)
model2probabilities = np.zeros(numGames)
model3probabilities = np.zeros(numGames)
model4probabilities = np.zeros(numGames)
model5probabilities = np.zeros(numGames)

# Run each game in testingData through all 5 fluid models
for game in range(testingData.shape[0]):

    currentResult = staley_base.predict(testingData[game,:], fluidModel1)
    model1results[game] = currentResult
    
    currentResult = staley_base.predict(testingData[game,:], fluidModel2)
    model2results[game] = currentResult
    
    currentResult = staley_base.predict(testingData[game,:], fluidModel3)
    model3results[game] = currentResult
    
    currentResult = staley_base.predict(testingData[game,:], fluidModel4)
    model4results[game] = currentResult
    
    currentResult = staley_base.predict(testingData[game,:], fluidModel5)
    model5results[game] = currentResult
    
# Set a counter variable for counting wins in each list
homeWinCount = 0
awayWinCount = 0

# List to hold win counts (3 - 5) for winners of given games
winCountList = np.zeros(numGames)

# This list will hold the probabilities for the majority result
homeWinProbabilities = list()
awayWinProbabilities = list()

# Probabilities for each game
weekProbabilities = np.zeros(numGames)

# Numpy array to hold the results for each model
resultTotal = np.zeros(numGames)

# Loop through prediction results for each game and each model
for x in range(model1results.shape[0]):

    # Add one to homeWinCount if the prediciton is a home team win
    # Add the probability for the given game to the correct list as well
    if model1results[x] == 1:
        homeWinCount += 1
        homeWinProbabilities.append(model1probabilities[x])

    # Add one to awayWinCount if the prediction is an away team win
    # Add the probability for the given game to the correct list as well
    elif model1results[x] == 0:
        awayWinCount += 1
        awayWinProbabilities.append(model1probabilities[x])

    #
    if model2results[x] == 1:
        homeWinCount += 1
        homeWinProbabilities.append(model2probabilities[x])

    elif model2results[x] == 0:
        awayWinCount += 1
        awayWinProbabilities.append(model2probabilities[x])

    #
    if model3results[x] == 1:
        homeWinCount += 1
        homeWinProbabilities.append(model3probabilities[x])
        
    elif model3results[x] == 0:
        awayWinCount += 1
        awayWinProbabilities.append(model3probabilities[x])

    #
    if model4results[x] == 1:
        homeWinCount += 1
        homeWinProbabilities.append(model4probabilities[x])

    elif model4results[x] == 0:
        awayWinCount += 1
        awayWinProbabilities.append(model4probabilities[x])

    #
    if model5results[x] == 1:
        homeWinCount += 1
        homeWinProbabilities.append(model5probabilities[x])

    elif model5results[x] == 0:
        awayWinCount += 1
        awayWinProbabilities.append(model5probabilities[x])

    # Add home team win if the majority of models predicts a win for the home teeam (1)
    # Record confidence level in winCountList
    if homeWinCount >= 3:
        
        # Put homeWinProbabilities into a numpy array later
        tempArr = np.array(homeWinProbabilities)

        # Put the result in the resultTotal list to be written to file and determine winner
        resultTotal[x] = 1
        
        # Put how many models predict a home win in the winCountList 
        winCountList[x] = int(homeWinCount)
        
        # Put the average win % in the weekProbabilities list
        weekProbabilities[x] = round(np.average(tempArr), 2)

    # Add away team win if the majority of models predict a win for the away team (0)
    # Record confidence level in winCountList
    elif awayWinCount >= 3:

        # Put awayWinProbabilities into a numpy array for averaging later
        tempArr = np.array(awayWinProbabilities)
        
        # put the result in the resultTotal list to be written to file and determine winner
        resultTotal[x] = 0
        
        # Put how many models preidct an away win in the winCountList
        winCountList[x] = int(awayWinCount)
        
        # Put the average win % in the weekProbabilities list
        weekProbabilities[x] = round(np.average(tempArr), 2)

    # Set winCount back to 0 before the next iteration
    awayWinCount = 0
    homeWinCount = 0

    # Clear probability lists
    awayWinProbabilities.clear()
    homeWinProbabilities.clear()

# Header for the output file
outputFile.write("------------------------------------ Fluid Week " + str(weekNum) + " -----------------------------------\n\n")

# This variable will be incremented each time a game is found for the given week
# It will be used to correlate the current game result with the resultTotal list
currentWeekGames = 0

# Loop through the whole schedule and only consider games for the given week
for x in range(1,totalGames + 1):
    
    # Variables that hold cell indentifiers for use in probing the excel sheet
    weekCell = 'A' + str(x)
    awayTeam = 'B' + str(x)
    homeTeam = 'C' + str(x)
    
    if int(currentSheet[weekCell].value) == int(weekNum):
        
        if resultTotal[currentWeekGames] == 1:

            outputFile.write("\t\t(" + str(int(winCountList[currentWeekGames])) + "/5) " + currentSheet[homeTeam].value + " over the " + currentSheet[awayTeam].value + " at home\n\n")
        
        elif resultTotal[currentWeekGames] == 0:

            outputFile.write("\t\t(" + str(int(winCountList[currentWeekGames])) + "/5) " + currentSheet[awayTeam].value + " over the " + currentSheet[homeTeam].value + " on the road\n\n")
        
        currentWeekGames += 1


outputFile.write("-------------------------------------------------------------------------------------\n\n")

# Close the output file
outputFile.close()

# Print success message
print(" ----- Week " + str(weekNum) + " fluid picks written using " + str(dataYear) + " data  ----- \n")
