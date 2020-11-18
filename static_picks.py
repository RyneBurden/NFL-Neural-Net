#!/usr/bin/env python

# ------------------------------------------------------------------- #
#
# Author: Ryne Burden
#
# Description:
#     - This file makes static model predictions using the current season data and previous season data
# 
#     - This script was built to be called by staleySays.py
#
# Tested o Python 3.8.1
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

# Load the fluid models
staticState1 = torch.load('Data/Network States/Static Models/static_model_1.staley')
staticState2 = torch.load('Data/Network States/Static Models/static_model_2.staley')
staticState3 = torch.load('Data/Network States/Static Models/static_model_3.staley')
staticState4 = torch.load('Data/Network States/Static Models/static_model_4.staley')
staticState5 = torch.load('Data/Network States/Static Models/static_model_5.staley')

# Define Network() objects for fluid models
staticModel1 = staley_base.Network()
staticModel2 = staley_base.Network()
staticModel3 = staley_base.Network()
staticModel4 = staley_base.Network()
staticModel5 = staley_base.Network()

# Load state_dict for each fluid model
staticModel1.load_state_dict(staticState1['state_dict'])
staticModel2.load_state_dict(staticState2['state_dict'])
staticModel3.load_state_dict(staticState3['state_dict'])
staticModel4.load_state_dict(staticState4['state_dict'])
staticModel5.load_state_dict(staticState5['state_dict'])

# Put all models in eval mode
staticModel1.eval()
staticModel2.eval()
staticModel3.eval()
staticModel4.eval()
staticModel5.eval()

# Open the schedule workbook for the current season and set the current sheet
wb = openpyxl.load_workbook('Data/Schedule/' + dataYear + '/' + dataYear + '.xlsx')
currentSheet = wb.active

# Variables for use later
totalGames = currentSheet.max_row
numGames = 0

# This loop goes through the whole excel sheet and finds the total number of games for the given week
for x in range(1,totalGames + 1):
    
    weekCell = 'A' + str(x)
    awayTeam = 'B' + str(x)
    homeTeam = 'C' + str(x)

    if int(currentSheet[weekCell].value) == int(weekNum):
        numGames += 1

# Numpy array to hold the results for each model
resultTotal = np.zeros(numGames)

# Load testing data and open the output file based on the year given
if dataYear == "2020":

    testingData = np.loadtxt('Data/Data Sets/2020/Validation Data/2020_Week_' + str(weekNum) + 'v.txt')
    outputFile = open('Data/Staley Picks/2020/2020 data/Static Model/' + str(dataYear) + ' Week ' + str(weekNum) + 's.txt', 'w')

elif dataYear == "2019":

    testingData = np.loadtxt('Data/Data Sets/2020/Validation Data/2019 Averages/2020_Week_' + str(weekNum) + 'v.txt')
    outputFile = open('Data/Staley Picks/2020/2019 data/Static Model/' + str(int(dataYear) + 1) + ' Week ' + str(weekNum) + 's.txt', 'w')

# Dummy arrays to hold results for each model
model1results = np.zeros(numGames)
model2results = np.zeros(numGames)
model3results = np.zeros(numGames)
model4results = np.zeros(numGames)
model5results = np.zeros(numGames)

# Run each game in testingData through all 5 fluid models
for game in range(testingData.shape[0]):

    currentResults = staley_base.predict(testingData[game,:], staticModel1)
    model1results[game] = currentResults
        
    currentResults = staley_base.predict(testingData[game,:], staticModel2)
    model2results[game] = currentResults

    currentResults = staley_base.predict(testingData[game,:], staticModel3)
    model3results[game] = currentResults

    currentResults = staley_base.predict(testingData[game,:], staticModel4)
    model4results[game] = currentResults

    currentResults = staley_base.predict(testingData[game,:], staticModel5)
    model5results[game] = currentResults

# Set a counter variable for counting wins in each list
winCount = 0
# List to hold win counts (3 - 5) for winners of given games
winCountList = np.zeros(numGames)

# Loop through prediction results for each game and each model
for x in range(model1results.shape[0]):

    # Add one to win count if the prediction is a home team win
    if model1results[x] == 1:
        winCount += 1
    if model2results[x] == 1:
        winCount += 1
    if model3results[x] == 1:
        winCount += 1
    if model4results[x] == 1:
        winCount += 1
    if model5results[x] == 1:
        winCount += 1

    # Add home team win if the majority of models predicts a win for the home teeam (1)
    # Record confidence level in winCountList
    if winCount >= 3:

        resultTotal[x] = 1
        winCountList[x] = int(winCount)
    # Add away team win if the majority of models predict a win for the away team (0)
    # Record confidence level in winCountList
    else:

        resultTotal[x] = 0
        winCountList[x] = 5 - winCount

    # Set winCount back to 0 before the next iteration
    winCount = 0

outputFile.write("------------------------------------ Static Week " + str(weekNum) + " -----------------------------------\n\n")

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
           
            outputFile.write("\t\t(" + str(int(winCountList[currentWeekGames]))  + "/5) " + currentSheet[homeTeam].value + " over the " + currentSheet[awayTeam].value + " at home\n\n")

        elif resultTotal[currentWeekGames] == 0:
           
            outputFile.write("\t\t(" + str(int(winCountList[currentWeekGames])) + "/5) " +  currentSheet[awayTeam].value + " over the " + currentSheet[homeTeam].value + " on the road\n\n")

        currentWeekGames += 1

outputFile.write("------------------------------------------------------------------------------------\n\n")

# Close the output file
outputFile.close()

# Print success message
print(" ----- Week " + str(weekNum) + " static picks written using " + str(dataYear) + " data ----- \n")
