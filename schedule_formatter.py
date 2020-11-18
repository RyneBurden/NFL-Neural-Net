#!/usr/bin/env python

# -------------------------------------------------- #
#
# Author: Ryne Burden
#
# Description:
#     - This script formats the schedule spreadsheet to always list the away team in the left column
#     
#     - The schedule spreadsheet was made from a csv file downloaded from pro football reference
#
#     - Past season schedules lists winning teams first, so this script was needed to correct that
#
# Tested on Python 3.8.1
#
# -------------------------------------------------- #

import openpyxl
import sys

def main():
    # Take the year from the command line
    year = sys.argv[1]
    # Open the spreadsheet and load it into a openpyxl object
    wb = openpyxl.load_workbook('Data/Schedule/' + str(year) + '/' + str(year) + '.xlsx')
    # Set the active sheet as the current sheet
    # This is needed to get values from cells, etc
    currentSheet = wb.active

    # Get the number of games in the spreadsheet
    numGames = currentSheet.max_row

    # Loop through every row in the open workbook
    # Range is 1 through numGames+1 so the loops uses indices 1 through the max instead of 0 through max-1
    for game in range(1, numGames + 1):
        # Variables for the current B, C and D columns
        currentB = 'B' + str(game)
        currentC = 'C' + str(game)
        currentD = 'D' + str(game)
        # Swap the values of columns B and D if the away team is not in column B
        # An @ in columnC means the away team is listed first
        if currentSheet[currentC].value != '@':
            tempVar = currentSheet[currentB].value
            currentSheet[currentB] = currentSheet[currentD].value
            currentSheet[currentD] = tempVar
            currentSheet[currentC] = '@'

    # Delete the column of @'s since we are changing all away teams to be the first team
    currentSheet.delete_cols(3)
            
    # Save the workbook
    wb.save('Data/Schedule/' + str(year) + '/' + str(year) + '.xlsx')

main()
