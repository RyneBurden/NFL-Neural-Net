#!/usr/bin/env python3

# -------------------------------------------------- #
#
# Author: Ryne Burden
#
# Description: 
#     - This script can be called on it's own, however it's usually called from data_gather.py
#     
# Standalone usage example: 
#     - python3 team_scraper.py Chicago\ Bears 2020 8 OR ./team_scraper.py Chicago\ Bears 2020 8
# 
#     - The script call above will scrape data for the 2020 Chicago Bears through week 8. Use 17 to scrape a whole season 
#
# Tested on Python 3.8.1
#
# -------------------------------------------------- #

from bs4 import BeautifulSoup as bs
import requests
import sys
import os
import teams

# Global list of dictionaries with PFR data-stat tags to search with Beautiful Soup
statList = [
    {'data-stat':"opp"}, # [0] - opposing team
    {'data-stat':"first_down_off"}, # [1] - offensive first downs gained
    {'data-stat':"pass_yds_off"}, # [2] - offensive passing yards gained per game
    {'data-stat':"rush_yds_off"}, # [3] - offensive rushing yards gained per game
    {'data-stat':"to_off"}, # [4] - offensive giveaways 
    {'data-stat':"first_down_def"}, # [5] - defensive first downs allowed per game
    {'data-stat':"pass_yds_def"}, # [6] - defensive passing yards allowed per game
    {'data-stat':"rush_yds_def"}, # [7] - defensive rushing yards allowed per game
    {'data-stat':"to_def"}, # [8] defensive takeaways
    {'data-stat':"game_outcome"} # [9] 1 for win - 0 for loss
]

def main():
    # Take arguments from the script call
    # This method makes it easier to automate the process later
    team = sys.argv[1]
    year = sys.argv[2]
    week = sys.argv[3]
    
    # Formulate URL based on given team and season
    # This will pull up the basic stat boxscores for each game
    # teams.py is a file I made that correlates a team's whole name with their PFR abbreviation
    seasonUrl = "https://www.pro-football-reference.com/teams/" + teams.URL[team] + "/" + str(year) + ".htm"
    
    # request the HTML from URL
    pageRequest = requests.get(seasonUrl)
    
    # Turn the HTML document into a Beautiful Soup object
    seasonToScrape = bs(pageRequest.text, "lxml")
    
    # Find the game stat table
    currentSeason = seasonToScrape.find(id="games")

    # Get rid of everything but the game stats that are located in the table body
    currentSeason = currentSeason.tbody

    # Type case team as a string to make sure it is processed correctly
    # This may be unecessary but I did it anyway - it worked
    team = str(team)

    # Opens a new txt file for the team's stats
    seasonToOutput = open("Data/Stats/" + str(year) + "/" + team.replace(" ", "_") + "_" + str(year) + ".txt", "w")

    # priming variable
    # Pro Football Reference uses HTML tables, so this script looks for table rows (tr) of data
    currentWeek = currentSeason.find("tr")

    # This loop goes through each game in the user given range
    for week in range (0, int(week)):
        for stat in statList:
            
            # Find the next stat in the statList dictionary list
            currentStat = currentWeek.find(attrs = stat)
            # Get the text from the HTML tag
            currentStat = currentStat.get_text()

            # This list will hold stats to reference while scraping
            stats = []

            #Conditionals for uniform data formatting
            if currentStat != "":
                # Write a 1 to file if the given week's game was won
                if currentStat == "W":
                    seasonToOutput.write("1.000 ")
                    stats.append("1")
                # Write a 0 to file for a loss or tie
                elif currentStat == "L" or currentStat == "T":
                    if currentStat == "L":
                        seasonToOutput.write("0.000 ")
                        stats.append("0")
                    #elif currentStat== "T":
                        #seasonToOutput.write("0.500 ")
                        #stats.append("0.5")
                # Each elif statement below ensure the given stat is output with 4 figures
                # This makes the .txt files much easier to read
                elif len(currentStat) == 1:
                    seasonToOutput.write(currentStat + ".000 ")
                    stats.append(currentStat)
                
                elif len(currentStat) == 2:
                    seasonToOutput.write(currentStat + ".00 ")
                    stats.append(currentStat)
                
                elif len(currentStat) == 3:
                    seasonToOutput.write(currentStat + ".0 ")
                    stats.append(currentStat)

            # Write a 0.000 if currentStat is blank - used for bye weeks
            else: 
                seasonToOutput.write("0.000 ")

            # Reset current Stat after find
            currentStat = None

        # Clear stat list before moving on to the next week
        stats.clear()

        # Write a newline to the file to separate weeks
        if currentStat != "Bye Week":
            seasonToOutput.write("\n")

        # Cycle currentWeek to the next table row
        currentWeek = currentWeek.find_next_sibling("tr")

    # Close the output file to end
    seasonToOutput.close()

main()
