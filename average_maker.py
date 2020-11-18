#!/usr/bin/env python

# -------------------------------------------------- #
#
# Author: Ryne Burden
#
# Description:
#     - This script is used to calculate averages for each input stat to the prediction network
#
#     - It's usually called from the data_maker script before the new data sets are created
#
# Tested on Python 3.8.1
#
# -------------------------------------------------- #

import numpy as np
import teams
import os
from sklearn import preprocessing
import sys

# Take the year from the user as a command line argument
year = sys.argv[1]
# Cast the year variable as an integer since sys.argv reads in as a string
year = int(year)

# This variables is used in a loop to help find the bye week for a given team
zeroCount = 0

# These np arrays are used to hold the averages for each team
# The size allocated here (8) is arbitary, as the first 8 values are deleted later
# The order of the values in this list is determined by the order in teams.py
NFC = np.zeros(8)
AFC = np.zeros(8)

# Loop through all teams in the NFC list in teams.py
for x in range(4):
    for y in range(4):
        
        # Assign the current indexed team to a variable
        team = teams.NFC[x][y]
        # Load the data associated with the current team into a numpy array
        currentData = np.loadtxt('Data/Stats/' + str(year) + '/' + str(team.replace(" ","_")) + "_" + str(year) + ".txt")
        
        # Set byeWeek to None, it will be changed when the bye week found
        byeWeek = None

        # Iterate through currentData to find the bye week
        for i in range(currentData.shape[0]):
            for j in range(currentData.shape[1]):
                if (currentData[i][j] == 0.000):
                    zeroCount += 1    
                if zeroCount == 9:
                    byeWeek = i
        
            # Reset zero count for each week
            zeroCount = 0
        
        # This protection was added for when the script is called and the team in question hasn't had a bye week yet
        if byeWeek != None:
            # Delete the bye week stats from the list if the current team has had one
            currentData = np.delete(currentData, byeWeek, 0)
        
        # Delete the game outcome from the stat list
        currentData = np.delete(currentData, 8, 1)
        # Average the data for each column
        currentData = np.average(currentData,axis=0)
        # Append the current team data to the NFC list
        NFC = np.append(NFC, currentData, axis=0)

for x in range(4):
    for y in range(4):
        
        # Assign the current indexed team to a variable
        team = teams.AFC[x][y]
        # Load the data associated with the current team into a numpy array
        currentData = np.loadtxt('Data/Stats/' + str(year) + '/' + str(team.replace(" ","_")) + "_" + str(year) + ".txt")
        
        # Set byeWeek to None, it will be changed when the bye week is found
        byeWeek = None

        # Iterate through currentData to find the bye week
        for i in range(currentData.shape[0]):
            for j in range(currentData.shape[1]):
                if (currentData[i][j] == 0.000):
                    zeroCount += 1    
                if zeroCount == 9:
                        byeWeek = i
            
            # Reset zero count for each week
            zeroCount = 0
    
        # This protection was added for when the script is called and the team in question hasn't had a bye week yet
        if byeWeek != None:
            currentData = np.delete(currentData, byeWeek, 0)

        # Delete the game outcome from the stat list
        currentData = np.delete(currentData, 8, 1)
        # Average the data for each column
        currentData = np.average(currentData,axis=0)
        # Append the current team data to the AFC list
        AFC = np.append(AFC, currentData, axis=0)


# Delete the first 8 values from NFC, this is because NFC is initialized as a np array filled with eight 0's
NFC = np.delete(NFC, [0,1,2,3,4,5,6,7], axis=0)
# Reshape the array to have a row for each team, 8 features per team
NFC = np.reshape(NFC, (16,8))

# Delete the first 8 values from AFC, this is because AFC is initialized as a np array filled with eight 0's
AFC = np.delete(AFC, [0,1,2,3,4,5,6,7], axis=0)
# Reshape the array to have a row for each team, 8 features per team
AFC = np.reshape(AFC, (16,8))

# Save AFC
np.savetxt('Data/Data Sets/' + str(year) + '/' + 'Validation Data/' +  str(year) + 'AFCavg.txt', AFC, delimiter=' ', newline='\n')

# Save NFC
np.savetxt('Data/Data Sets/' + str(year) + '/' + 'Validation Data/' + str(year) + 'NFCavg.txt', NFC, delimiter=' ', newline='\n')

print(" *-- League Averages Re-Calculated --* ")
