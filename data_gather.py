#!/usr/bin/env python3

# -------------------------------------------------- #
#
# Author: Ryne Burden
#
# Description:
#     - This script calls team_scraper.py for each team in teams.py
#
# Tested on Python 3.8.1
# 
# -------------------------------------------------- #

import os
import teams
import random
import time
import sys

# Take the year, past week, and random seed from the command line
year = sys.argv[1]
pastWeek = sys.argv[2]
seed = sys.argv[3]

# Cycle through all teams to run teamScrape for each given team and each given year
for x in range(0, 4):
    for y in range(0, 4):
        # Random wait time for sleep timer 
        waitTimeInSeconds = random.randrange(30, 60)
            
        # Change team to the next NFC team
        team = teams.NFC[x][y]
        print("Scraping the " + str(year) + " " + team)
        os.system("./team_scraper.py " + team.replace(" ", "\ ") + " " + str(year) + " " + pastWeek)
            
        # Built in sleep to avoid timeouts from PFR server
        print("Scrape Successful, waiting for " + str(waitTimeInSeconds) + " seconds")
        time.sleep(waitTimeInSeconds)
            
        # Random wait time for sleep timer 2
        waitTimeInSeconds = random.randrange(30, 60)
            
        # Change team to the next AFC team
        team = teams.AFC[x][y]
        print("Scraping the " + str(year)+ " " + team)
        os.system("./team_scraper.py " + team.replace(" ", "\ ") + " " + str(year) + " " + pastWeek)

        # Built in sleep to avoid timeouts from PFR server
        print("Scrape Successful, waiting for " + str(waitTimeInSeconds) + " seconds")
        time.sleep(waitTimeInSeconds)
