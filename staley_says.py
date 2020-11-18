#!/usr/bin/env python  

# ------------------------------------------------------------------- #
#
# Author: Ryne Burden
#
# Description:
#     - This is the main driver for the Staley prediction model
# 
#     - It calls fluidPicks.py and/or staticPicks.py based on user input
#
# Tested on Python 3.8.1
#
# ------------------------------------------------------------------- #

import sys
import os

# Take input from the user to determine which scripts to run
choice = input("(f)luid, (s)tatic, or (b)oth? ")
weekNum = input("Week number: ")
currentSeason = input('Current season: ')

# Determine if the fluid model should be retrained
if choice == 'f' or choice == 'b':
    retrain = input("retrain the fluid model? ")

# Clear the terminal screen
os.system('clear')

# ------------------------------------------------------------------- #
# Action tree based on user input
if choice == 'f':
    
    # Action tree for retrain
    if retrain == 'y':
        
        # Print status
        print("Writing fluid model picks for week " + str(weekNum) + " and retraining the models\n")
        
        # Make fluid picks for the week with current season data and data from the last season
        os.system('./fluid_picks.py ' + str(weekNum) + ' ' + str(currentSeason) + ' y')
        # Don't retrain for the second run 
        os.system('./fluid_picks.py ' + str(weekNum) + ' ' + str(int(currentSeason) - 1) + ' n')
    
    elif retrain == 'n':
        
        #Print status
        print("Writing fluid model picks for week " + str(weekNum) + " without retraining\n")
        
        # Make fluid picks for the week with current season data and data from the last season
        os.system('./fluid_picks.py ' + str(weekNum) + ' ' + str(currentSeason) + ' n')
        # Don't retrain for the second run
        os.system('./fluid_picks.py ' + str(weekNum) + ' ' + str(int(currentSeason) - 1) + ' n')

    else:
        print('Invalid retraining input')

# ------------------------------------------------------------------- #
elif choice == 's':
    
    # Print status
    print("Writing static model picks for week " + str(weekNum) + "\n")
    
    # Make static picks for current season and previous season
    os.system('./static_picks.py ' + str(weekNum) + ' ' + str(currentSeason))
    os.system('./static_picks.py ' + str(weekNum) + ' ' + str(int(currentSeason) - 1))

# ------------------------------------------------------------------- #
elif choice == 'b':
    # Print status
    print("Writing static and fluid model picks for week " + str(weekNum) + "\n")
    
    # Make static picks for current season and previous season
    os.system('./static_picks.py ' + str(weekNum) + ' ' + str(currentSeason))
    os.system('./static_picks.py ' + str(weekNum) + ' ' + str(int(currentSeason) - 1))
    
    # Make fluid picks based on retrain choice
    if retrain == 'y':
        
        # Print status
        print("Writing fluid model picks for week " + str(weekNum) + " and retraining the model\n")
        
        # Make fluid picks for the week with current season data and data from the last season
        os.system('./fluid_picks.py ' + str(weekNum) + ' ' + str(currentSeason) + ' y')
        # Don't retrain for the second run
        os.system('./fluid_picks.py ' + str(weekNum) + ' ' + str(int(currentSeason) - 1) + ' n')

    elif retrain == 'n':
        
        # Print status
        print("Writing fluid model picks for week " + str(weekNum) + " without retraining the model\n")
        
        # Make fluid picks for the week with current sseason data and data from the last season
        os.system('./fluid_picks.py ' + str(weekNum) + ' ' + str(currentSeason) + ' n')
        os.system('./fluid_picks.py ' + str(weekNum) + ' ' + str(int(currentSeason) - 1) + ' n')

    else:
        print('Invalid retraining input')

# ------------------------------------------------------------------- #
else:
    print('Invalid choice')
        
