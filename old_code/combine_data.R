num_weeks <- 17
# get year range from data so if make_data changes, it won't affec this
years <- staley_data %>% group_by(SEASON) %>%  filter(HOA!="BYE") %>% select(SEASON) %>% unique()
begin_year <- years[1,]$SEASON
end_year <- years[nrow(years),]$SEASON

# Data frames to hold our training data
training_data <- data.frame()
current_season_data = data.frame()
current_week_data = data.frame()

# Loop through each year
for (current_year in begin_year:end_year) {
    
    # Set the schedule for the year being made
    current_season_schedule <- nfl_schedule %>% filter(season==current_year)
    current_season_data <- data.frame()
    
    # Loop through each week
    for (current_week in 1:num_weeks) {
        
        # filter current_season_schedule to only consider games for the given week
        current_week_schedule <- current_season_schedule %>% filter(week==current_week)
        current_week_data <- data.frame()
        
        # Loop through every game for the given week
        for (current_game_index in 1:nrow(current_week_schedule)) {

            # Set current game based on the current index
            current_game = current_week_schedule[current_game_index,]
                        
            # Account for team relocation and name changes in the schedule file
            # OAK to LV
            if(current_game$home_team=="OAK") {current_game$home_team="LV"}
            if(current_game$away_team=="OAK") {current_game$away_team="LV"}
            # STL to LA
            if(current_game$home_team=="STL") {current_game$home_team="LA"}
            if(current_game$away_team=="STL") {current_game$away_team="LA"}
            # SD to LAC
            if(current_game$home_team=="SD") {current_game$home_team="LAC"}
            if(current_game$away_team=="SD") {current_game$away_team="LAC"}
            
            # Put together dataset for each team
            current_home_data <- staley_data %>% filter(WEEK==current_week, SEASON==current_year, TEAM==current_game$home_team)
            current_away_data <- staley_data %>% filter(WEEK==current_week, SEASON==current_year, TEAM==current_game$away_team)
            
            # Delete unnecessary data from current_home and current_away data
            current_home_data$TEAM <- NULL
            current_home_data$SEASON <- NULL
            current_home_data$WEEK <- NULL
            current_home_data$HOA <- NULL
            
            current_away_data$TEAM <- NULL
            current_away_data$SEASON <- NULL
            current_away_data$WEEK <- NULL
            current_away_data$HOA <- NULL
            # Delete the point differential from the away dataset only 
            # This was a design choice so a positive diff means the home team won
            # and therefore the expected result is [0][1]
            current_away_data$PTS_DIFF <- NULL
            # Get rid of the div game classifier in the away data so it isn't repeated
            current_away_data$DIV <- NULL
            # Get rid of away team turnovers since that will be reflected in the home team turnovers too
            #current_away_data$OFF_TO <- NULL
            #current_away_data$DEF_TO <- NULL
            
            ### LOOK OVER THE LAST 3 DATA REMOVALS AND THINK ABOUT HOW THEY AFFECT TESTING DATA
            
            # combine datasets into one parent set
            current_game_data <- append(current_away_data, current_home_data)

            # Add current_game_data to training_data
            current_week_data <- rbind(current_week_data, current_game_data)
            
        }
        
        # Add current_week_training_data to current_season_training_data after each week
        current_season_data <- rbind(current_season_data, current_week_data)

    }
    
    # Add current_season_training_data to training_data after each season
    training_data <- rbind(training_data, current_season_data)
}

remove(years)
remove(begin_year)
remove(end_year)
remove(current_year)
remove(current_week)
remove(current_week_schedule)
remove(current_season_schedule)
remove(current_game_index)
remove(current_game)
remove(current_home_data)
remove(current_away_data)
remove(current_game_data)
remove(num_weeks)
remove(current_week_data)
remove(current_season_data)