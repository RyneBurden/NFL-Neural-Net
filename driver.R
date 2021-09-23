# Load Python files
reticulate::use_virtualenv("C:/Users/Ryne/Documents/R Projects and Files/Staley_2.0/virtualenvs/staley")
reticulate::py_run_file("staley_train-v2.py")

# Read in current week
current_season <- readline("What season are we looking at right now? ")
current_week <- readline(glue("What week of the ", current_season ," season should I analyze? "))
to_write <- readline("Should I write to the excel file when I finish? ")
current_week = as.integer(current_week)
current_season <- as.integer(current_season)

current_pbp <- nflfastR::load_pbp(current_season)

if (as.integer(current_week) < 8) {
    last_season_pbp <- nflfastR::load_pbp(as.integer(current_season) - 1)
    last_season_pbp <- last_season_pbp %>% filter(week > (17 - (8 - as.integer(current_week))) & week < 18)
    current_pbp <- current_pbp %>% rbind(last_season_pbp)
}


games = nfl_schedule %>% filter(week==current_week, season==current_season)
current_week_data <- data.frame()
predictions <- data.frame()
current_week_predictions <- data.frame()
current_week_picks <- data.frame(matrix(ncol = 7))

writeLines("Bada bing I'm on it boss")

#current_pbp <- current_pbp %>% filter(week != 1)

for(current_game in 1:nrow(games)) {
    
    # Get the current home and away teams
    current_away = games[current_game,] %>% select(away_team)
    current_home = games[current_game,] %>% select(home_team)
    div_game = games[current_game,] %>% select(div_game)
    
    # make variables for each data point needed away and home
    # calculate each variable using a for loop and 2021_pbp
    # NOTE: first 2 weeks will need to use 2019 stats
    
    if (current_week == 1) {
        
        current_pbp <- nflfastR::load_pbp(as.integer(current_season) - 1)
        
        # Rolling week so the remove statements below work
        rolling_week = 0
        
        # I'm using last season here since it's the first week of the season
        last_season = current_season - 1

        ################### AWAY STATS ###################
        # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
        away_general_data <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_away$away_team) %>% select(posteam_type, div_game) %>% unique()
        total_away_off_plays <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_away$away_team, play_type=="pass" | play_type=="run") %>% count()
        total_away_def_plays <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_away$away_team, play_type=="pass" | play_type=="run") %>% count()
        
        # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
        current_away_off_rush_epa <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_away$away_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(OFF_RUSH_EPA=mean(epa))
        current_away_off_pass_epa <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_away$away_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(OFF_PASS_EPA=mean(epa))
        current_away_first_downs_for <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_away$away_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
        current_away_off_fdr <- current_away_first_downs_for / total_away_off_plays
        
        # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
        current_away_def_rush_epa <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_away$away_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(DEF_RUSH_EPA=mean(epa))
        current_away_def_pass_epa <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_away$away_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(DEF_PASS_EPA=mean(epa))
        current_away_first_downs_allowed <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_away$away_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
        current_away_def_fdr <- current_away_first_downs_allowed / total_away_def_plays
        
        # ----- TURNOVER DATA ----- #
        current_away_giveaways <- (current_pbp %>% filter(season==last_season, week <= 17, posteam==current_away$away_team, fumble_lost==1 | interception==1) %>% count()) / 16
        current_away_takeaways <- (current_pbp %>% filter(season==last_season, week <= 17, defteam==current_away$away_team, fumble_lost==1 | interception==1) %>% count()) / 16
        
        # ----- EXPLOSIVE PLAY RATE ----- #
        current_away_off_exp_plays <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_away$away_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
        current_away_off_exp_play_rate <- current_away_off_exp_plays / total_away_off_plays
        current_away_def_exp_plays <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_away$away_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
        current_away_def_exp_play_rate <- current_away_def_exp_plays / total_away_def_plays
        
        # ----- PENALTIES & PENALTY YARDS ----- #
        current_away_off_penalties <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_away$away_team, penalty==1 & penalty_team==current_away$away_team) %>% count()
        current_away_off_penalty_yds <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_away$away_team, penalty==1 & penalty_team==current_away$away_team) %>% summarise(Off_Pen_Yds=sum(penalty_yards))
        current_away_def_penalties <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_away$away_team, penalty==1 & penalty_team==current_away$away_team) %>% count()
        current_away_def_penalty_yds <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_away$away_team, penalty==1 & penalty_team==current_away$away_team) %>% summarise(Def_Pen_Yds=sum(penalty_yards))
        current_away_total_penalties <- (current_away_off_penalties + current_away_def_penalties) / 16
        current_away_total_penalty_yds <- (current_away_off_penalty_yds + current_away_def_penalty_yds) / 16
        
        # ----- QB HIT + SACK DIFFERENTIAL ----- #
        # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
        # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
        current_away_off_line_metric <- (current_pbp %>% filter(season==last_season, week <= 17, posteam==current_away$away_team, qb_hit==1 | sack==1) %>% count()) / 16
        current_away_def_line_metric <- (current_pbp %>% filter(season==last_season, week <= 17, defteam==current_away$away_team, qb_hit==1 | sack==1) %>% count()) / 16
        
        # Make away data frame
        current_away_data <- data.frame(
            OFF_RUSH_EPA <- current_away_off_rush_epa$OFF_RUSH_EPA,
            OFF_PASS_EPA <- current_away_off_pass_epa$OFF_PASS_EPA,
            OFF_FDR <- current_away_off_fdr$n,
            DEF_RUSH_EPA <- current_away_def_rush_epa$DEF_RUSH_EPA,
            DEF_PASS_EPA <- current_away_def_pass_epa$DEF_PASS_EPA,
            DEF_FDR <- current_away_def_fdr$n,
            OFF_TO <- current_away_giveaways$n,
            DEF_TO <- current_away_takeaways$n,
            OFF_EXP_RATE <- current_away_off_exp_play_rate$n,
            DEF_EXP_RATE <- current_away_def_exp_play_rate$n,
            PENS <- current_away_total_penalties$n,
            PEN_YDS <- current_away_total_penalty_yds$Off_Pen_Yds,
            OL_METRIC <- current_away_off_line_metric$n,
            DL_METRIC <- current_away_def_line_metric$n
        )
        
        
        ################### HOME STATS ###################
        # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
        home_general_data <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_home$home_team) %>% select(posteam_type, div_game) %>% unique()
        total_home_off_plays <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_home$home_team, play_type=="pass" | play_type=="run") %>% count()
        total_home_def_plays <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_home$home_team, play_type=="pass" | play_type=="run") %>% count()
        
        # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
        current_home_off_rush_epa <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(OFF_RUSH_EPA=mean(epa))
        current_home_off_pass_epa <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(OFF_PASS_EPA=mean(epa))
        current_home_first_downs_for <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_home$home_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
        current_home_off_fdr <- current_home_first_downs_for / total_home_off_plays
        
        # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
        current_home_def_rush_epa <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(DEF_RUSH_EPA=mean(epa))
        current_home_def_pass_epa <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(DEF_PASS_EPA=mean(epa))
        current_home_first_downs_allowed <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_home$home_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
        current_home_def_fdr <- current_home_first_downs_allowed / total_home_def_plays
        
        # ----- TURNOVER DATA ----- #
        current_home_giveaways <- (current_pbp %>% filter(season==last_season, week <= 17, posteam==current_home$home_team, fumble_lost==1 | interception==1) %>% count()) / 16
        current_home_takeaways <- (current_pbp %>% filter(season==last_season, week <= 17, defteam==current_home$home_team, fumble_lost==1 | interception==1) %>% count()) / 16
        
        # ----- EXPLOSIVE PLAY RATE ----- #
        current_home_off_exp_plays <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_home$home_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
        current_home_off_exp_play_rate <- current_home_off_exp_plays / total_home_off_plays
        current_home_def_exp_plays <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_home$home_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
        current_home_def_exp_play_rate <- current_home_def_exp_plays / total_home_def_plays
        
        # ----- PENALTIES & PENALTY YARDS ----- #
        current_home_off_penalties <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% count()
        current_home_off_penalty_yds <- current_pbp %>% filter(season==last_season, week <= 17, posteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% summarise(Off_Pen_Yds=sum(penalty_yards))
        current_home_def_penalties <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% count()
        current_home_def_penalty_yds <- current_pbp %>% filter(season==last_season, week <= 17, defteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% summarise(Def_Pen_Yds=sum(penalty_yards))
        current_home_total_penalties <- (current_home_off_penalties + current_home_def_penalties) / 16
        current_home_total_penalty_yds <- (current_home_off_penalty_yds + current_home_def_penalty_yds) / 16
        
        # ----- QB HIT + SACK DIFFERENTIAL ----- #
        # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
        # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
        current_home_off_line_metric <- (current_pbp %>% filter(season==last_season, week <= 17, posteam==current_home$home_team, qb_hit==1 | sack==1) %>% count()) / 16
        current_home_def_line_metric <- (current_pbp %>% filter(season==last_season, week <= 17, defteam==current_home$home_team, qb_hit==1 | sack==1) %>% count()) / 16
        
        # Make home data frame
        current_home_data <- data.frame(
            OFF_RUSH_EPA <- current_home_off_rush_epa$OFF_RUSH_EPA,
            OFF_PASS_EPA <- current_home_def_pass_epa$DEF_PASS_EPA,
            OFF_FDR <- current_home_off_fdr$n,
            DEF_RUSH_EPA <- current_home_def_rush_epa$DEF_RUSH_EPA,
            DEF_PASS_EPA <- current_home_def_pass_epa$DEF_PASS_EPA,
            DEF_FDR <- current_home_def_fdr$n,
            OFF_TO <- current_home_giveaways$n,
            DEF_TO <- current_home_takeaways$n,
            OFF_EXP_RATE <- current_home_off_exp_play_rate$n,
            DEF_EXP_RATE <- current_home_def_exp_play_rate$n,
            PENS <- current_home_total_penalties$n,
            PEN_YDS <- current_home_total_penalty_yds$Off_Pen_Yds,
            OL_METRIC <- current_home_off_line_metric$n,
            DL_METRIC <- current_home_def_line_metric$n,
            DIV_GAME <- div_game
        )
        
    }
    else {
        
        ################### AWAY STATS ###################
        # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
        away_general_data <- current_pbp %>% filter(week < current_week, posteam==current_away$away_team) %>% select(posteam_type, div_game) %>% unique()
        total_away_off_plays <- current_pbp %>% filter(week < current_week, posteam==current_away$away_team, play_type=="pass" | play_type=="run") %>% count()
        total_away_def_plays <- current_pbp %>% filter(week < current_week, defteam==current_away$away_team, play_type=="pass" | play_type=="run") %>% count()
        
        # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
        current_away_off_rush_epa <- current_pbp %>% filter(week < current_week, posteam==current_away$away_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(OFF_RUSH_EPA=mean(epa))
        current_away_off_pass_epa <- current_pbp %>% filter(week < current_week, posteam==current_away$away_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(OFF_PASS_EPA=mean(epa))
        current_away_first_downs_for <- current_pbp %>% filter(week < current_week, posteam==current_away$away_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
        current_away_off_fdr <- current_away_first_downs_for / total_away_off_plays
        
        # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
        current_away_def_rush_epa <- current_pbp %>% filter(week < current_week, defteam==current_away$away_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(DEF_RUSH_EPA=mean(epa))
        current_away_def_pass_epa <- current_pbp %>% filter(week < current_week, defteam==current_away$away_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(DEF_PASS_EPA=mean(epa))
        current_away_first_downs_allowed <- current_pbp %>% filter(week < current_week, defteam==current_away$away_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
        current_away_def_fdr <- current_away_first_downs_allowed / total_away_def_plays
        
        # ----- TURNOVER DATA ----- #
        current_away_giveaways <- (current_pbp %>% filter(week < current_week, posteam==current_away$away_team, fumble_lost==1 | interception==1) %>% count()) / (current_week - 1) 
        current_away_takeaways <- (current_pbp %>% filter(week < current_week, defteam==current_away$away_team, fumble_lost==1 | interception==1) %>% count()) / (current_week - 1)
        
        # ----- EXPLOSIVE PLAY RATE ----- #
        current_away_off_exp_plays <- current_pbp %>% filter(week < current_week, posteam==current_away$away_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
        current_away_off_exp_play_rate <- current_away_off_exp_plays / total_away_off_plays
        current_away_def_exp_plays <- current_pbp %>% filter(week < current_week, defteam==current_away$away_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
        current_away_def_exp_play_rate <- current_away_def_exp_plays / total_away_def_plays
        
        # ----- PENALTIES & PENALTY YARDS ----- #
        current_away_off_penalties <- current_pbp %>% filter(week < current_week, posteam==current_away$away_team, penalty==1 & penalty_team==current_away$away_team) %>% count()
        current_away_off_penalty_yds <- current_pbp %>% filter(week < current_week, posteam==current_away$away_team, penalty==1 & penalty_team==current_away$away_team) %>% summarise(Off_Pen_Yds=sum(penalty_yards))
        current_away_def_penalties <- current_pbp %>% filter(week < current_week, defteam==current_away$away_team, penalty==1 & penalty_team==current_away$away_team) %>% count()
        current_away_def_penalty_yds <- current_pbp %>% filter(week < current_week, defteam==current_away$away_team, penalty==1 & penalty_team==current_away$away_team) %>% summarise(Def_Pen_Yds=sum(penalty_yards))
        current_away_total_penalties <- (current_away_off_penalties + current_away_def_penalties) / (current_week - 1)
        current_away_total_penalty_yds <- (current_away_off_penalty_yds + current_away_def_penalty_yds) / (current_week - 1)
        
        # ----- QB HIT + SACK DIFFERENTIAL ----- #
        # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
        # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
        current_away_off_line_metric <- (current_pbp %>% filter(week < current_week, posteam==current_away$away_team, qb_hit==1 | sack==1) %>% count()) / (current_week - 1)
        current_away_def_line_metric <- (current_pbp %>% filter(week < current_week, defteam==current_away$away_team, qb_hit==1 | sack==1) %>% count()) / (current_week - 1)
        
        # Make away data frame
        current_away_data <- data.frame(
            OFF_RUSH_EPA <- current_away_off_rush_epa$OFF_RUSH_EPA / (current_week - 1),
            OFF_PASS_EPA <- current_away_off_pass_epa$OFF_PASS_EPA / (current_week - 1),
            OFF_FDR <- current_away_off_fdr$n / (current_week - 1),
            DEF_RUSH_EPA <- current_away_def_rush_epa$DEF_RUSH_EPA / (current_week - 1),
            DEF_PASS_EPA <- current_away_def_pass_epa$DEF_PASS_EPA / (current_week - 1),
            DEF_FDR <- current_away_def_fdr$n / (current_week - 1),
            OFF_TO <- current_away_giveaways$n / (current_week - 1),
            DEF_TO <- current_away_takeaways$n / (current_week - 1),
            OFF_EXP_RATE <- current_away_off_exp_play_rate$n / (current_week - 1),
            DEF_EXP_RATE <- current_away_def_exp_play_rate$n / (current_week - 1),
            PENS <- current_away_total_penalties$n / (current_week - 1),
            PEN_YDS <- current_away_total_penalty_yds$Off_Pen_Yds / (current_week - 1),
            OL_METRIC <- current_away_off_line_metric$n / (current_week - 1),
            DL_METRIC <- current_away_def_line_metric$n / (current_week - 1)
        )
        
        ################### HOME STATS ###################
        # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
        home_general_data <- current_pbp %>% filter(posteam==current_home$home_team) %>% select(posteam_type, div_game) %>% unique()
        total_home_off_plays <- current_pbp %>% filter(posteam==current_home$home_team, play_type=="pass" | play_type=="run") %>% count()
        total_home_def_plays <- current_pbp %>% filter(defteam==current_home$home_team, play_type=="pass" | play_type=="run") %>% count()
        
        # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
        current_home_off_rush_epa <- current_pbp %>% filter(posteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(OFF_RUSH_EPA=mean(epa))
        current_home_off_pass_epa <- current_pbp %>% filter(posteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(OFF_PASS_EPA=mean(epa))
        current_home_first_downs_for <- current_pbp %>% filter(posteam==current_home$home_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
        current_home_off_fdr <- current_home_first_downs_for / total_home_off_plays
        
        # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
        current_home_def_rush_epa <- current_pbp %>% filter(defteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(DEF_RUSH_EPA=mean(epa))
        current_home_def_pass_epa <- current_pbp %>% filter(defteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(DEF_PASS_EPA=mean(epa))
        current_home_first_downs_allowed <- current_pbp %>% filter(defteam==current_home$home_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
        current_home_def_fdr <- current_home_first_downs_allowed / total_home_def_plays
        
        # ----- TURNOVER DATA ----- #
        current_home_giveaways <- (current_pbp %>% filter(posteam==current_home$home_team, fumble_lost==1 | interception==1) %>% count()) / (current_week - 1)
        current_home_takeaways <- (current_pbp %>% filter(defteam==current_home$home_team, fumble_lost==1 | interception==1) %>% count()) / (current_week - 1)
        
        # ----- EXPLOSIVE PLAY RATE ----- #
        current_home_off_exp_plays <- current_pbp %>% filter(posteam==current_home$home_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
        current_home_off_exp_play_rate <- current_home_off_exp_plays / total_home_off_plays
        current_home_def_exp_plays <- current_pbp %>% filter(defteam==current_home$home_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
        current_home_def_exp_play_rate <- current_home_def_exp_plays / total_home_def_plays
        
        # ----- PENALTIES & PENALTY YARDS ----- #
        current_home_off_penalties <- current_pbp %>% filter(posteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% count()
        current_home_off_penalty_yds <- current_pbp %>% filter(posteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% summarise(Off_Pen_Yds=sum(penalty_yards))
        current_home_def_penalties <- current_pbp %>% filter(defteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% count()
        current_home_def_penalty_yds <- current_pbp %>% filter(defteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% summarise(Def_Pen_Yds=sum(penalty_yards))
        current_home_total_penalties <- (current_home_off_penalties + current_home_def_penalties) / (current_week - 1)
        current_home_total_penalty_yds <- (current_home_off_penalty_yds + current_home_def_penalty_yds) / (current_week - 1)
        
        # ----- QB HIT + SACK DIFFERENTIAL ----- #
        # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
        # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
        current_home_off_line_metric <- (current_pbp %>% filter(posteam==current_home$home_team, qb_hit==1 | sack==1) %>% count()) / (current_week - 1)
        current_home_def_line_metric <- (current_pbp %>% filter(defteam==current_home$home_team, qb_hit==1 | sack==1) %>% count()) / (current_week - 1)
        
        # Make home data frame
        current_home_data <- data.frame(
            OFF_RUSH_EPA <- current_home_off_rush_epa$OFF_RUSH_EPA / (current_week - 1),
            OFF_PASS_EPA <- current_home_def_pass_epa$DEF_PASS_EPA / (current_week - 1),
            OFF_FDR <- current_home_off_fdr$n / (current_week - 1),
            DEF_RUSH_EPA <- current_home_def_rush_epa$DEF_RUSH_EPA / (current_week - 1),
            DEF_PASS_EPA <- current_home_def_pass_epa$DEF_PASS_EPA / (current_week - 1),
            DEF_FDR <- current_home_def_fdr$n / (current_week - 1),
            OFF_TO <- current_home_giveaways$n / (current_week - 1),
            DEF_TO <- current_home_takeaways$n / (current_week - 1),
            OFF_EXP_RATE <- current_home_off_exp_play_rate$n / (current_week - 1),
            DEF_EXP_RATE <- current_home_def_exp_play_rate$n / (current_week - 1),
            PENS <- current_home_total_penalties$n / (current_week - 1),
            PEN_YDS <- current_home_total_penalty_yds$Off_Pen_Yds / (current_week - 1),
            OL_METRIC <- current_home_off_line_metric$n / (current_week - 1),
            DL_METRIC <- current_home_def_line_metric$n / (current_week - 1),
            DIV_GAME <- div_game
        )
        
    }
    # Not used, I got better results without the 4-week rolling average
    # else if (current_week >= 6) {
    #     
    #     # Set initial value for stats calculated
    #     current_away_off_rush_epa <- 0
    #     current_away_off_pass_epa <- 0
    #     current_away_first_downs_for <- 0
    #     current_away_off_fdr <- 0
    #     current_away_def_rush_epa <- 0
    #     current_away_def_pass_epa <- 0
    #     current_away_first_downs_allowed <- 0
    #     current_away_def_fdr <- 0
    #     current_away_giveaways <- 0
    #     current_away_takeaways <- 0
    #     current_away_off_exp_play_rate <- 0
    #     current_away_def_exp_play_rate <- 0
    #     current_away_total_penalties <- 0
    #     current_away_total_penalty_yds <- 0
    #     current_away_off_line_metric <- 0
    #     current_away_def_line_metric <- 0
    #     current_home_off_rush_epa <- 0
    #     current_home_off_pass_epa <- 0
    #     current_home_off_fdr <- 0
    #     current_home_def_rush_epa <- 0
    #     current_home_def_pass_epa <- 0
    #     current_home_def_fdr <- 0
    #     current_home_giveaways <- 0
    #     current_home_takeaways <- 0
    #     current_home_off_exp_play_rate <- 0
    #     current_home_def_exp_play_rate <- 0
    #     current_home_total_penalties <- 0
    #     current_home_total_penalty_yds <- 0
    #     current_home_off_line_metric <- 0
    #     current_home_def_line_metric <- 0
    #     total_away_off_plays <- 0
    #     total_away_def_plays <- 0
    #     total_home_off_plays <- 0
    #     total_home_def_plays <- 0
    #     
    #     # Rolling week starts at 5 since we're using 3 week rolling averages for games
    #     rolling_week_home <- as.integer(current_week - 4)
    #     rolling_week_away <- as.integer(current_week - 4)
    #     
    #     # Get team stats using a rolling 4-week stat average
    #     for (x in 1:4) {
    #         
    #         if (rolling_week_away == as.integer(bye_weeks_2020 %>% filter(team_abbr==current_away$away_team) %>% select(bye_week))) {
    # 
    #             rolling_week_away = current_week - 5
    #             change_away_rolling_week = TRUE
    #         
    #         }
    #         else {
    #             change_away_rolling_week = FALSE
    #         }
    #         
    #         if (rolling_week_home == as.integer(bye_weeks_2020 %>% filter(team_abbr==current_home$home_team) %>% select(bye_week))) {
    # 
    #                 rolling_week_home = current_week - 5
    #                 change_home_rolling_week = TRUE
    # 
    #         }
    #         else {
    #             change_home_rolling_week = FALSE
    #         }
    #         ################### AWAY STATS ###################
    #         # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
    #         away_general_data <- current_pbp %>% filter(season==current_season, week==rolling_week_away, posteam==current_away$away_team) %>% select(posteam_type, div_game) %>% unique()
    #         total_away_off_plays <- current_pbp %>% filter(season==current_season, week==rolling_week_away, posteam==current_away$away_team, play_type=="pass" | play_type=="run") %>% count()
    #         total_away_def_plays <- current_pbp %>% filter(season==current_season, week==rolling_week_away, defteam==current_away$away_team, play_type=="pass" | play_type=="run") %>% count()
    #         
    #         # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
    #         current_away_off_rush_epa <- current_away_off_rush_epa + (current_pbp %>% filter(season==current_season, week==rolling_week_away, posteam==current_away$away_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(OFF_RUSH_EPA=mean(epa)))
    #         current_away_off_pass_epa <- current_away_off_pass_epa + (current_pbp %>% filter(season==current_season, week==rolling_week_away, posteam==current_away$away_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(OFF_PASS_EPA=mean(epa)))
    #         current_away_first_downs_for <- current_pbp %>% filter(season==current_season, week==rolling_week_away, posteam==current_away$away_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
    #         current_away_off_fdr <- current_away_off_fdr + (current_away_first_downs_for / total_away_off_plays)
    #         
    #         # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
    #         current_away_def_rush_epa <- current_away_def_rush_epa + (current_pbp %>% filter(season==current_season, week==rolling_week_away, defteam==current_away$away_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(DEF_RUSH_EPA=mean(epa)))
    #         current_away_def_pass_epa <- current_away_def_pass_epa + (current_pbp %>% filter(season==current_season, week==rolling_week_away, defteam==current_away$away_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(DEF_PASS_EPA=mean(epa)))
    #         current_away_first_downs_allowed <- current_pbp %>% filter(season==current_season, week==rolling_week_away, defteam==current_away$away_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
    #         current_away_def_fdr <- current_away_def_fdr + (current_away_first_downs_allowed / total_away_def_plays)
    #         
    #         # ----- TURNOVER DATA ----- #
    #         current_away_giveaways <- current_away_giveaways + (current_pbp %>% filter(season==current_season, week==rolling_week_away, posteam==current_away$away_team, fumble_lost==1 | interception==1) %>% count())
    #         current_away_takeaways <- current_away_takeaways + (current_pbp %>% filter(season==current_season, week==rolling_week_away, defteam==current_away$away_team, fumble_lost==1 | interception==1) %>% count())
    #         
    #         # ----- EXPLOSIVE PLAY RATE ----- #
    #         current_away_off_exp_plays <- current_pbp %>% filter(season==current_season, week==rolling_week_away, posteam==current_away$away_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
    #         current_away_off_exp_play_rate <- current_away_off_exp_play_rate + (current_away_off_exp_plays / total_away_off_plays)
    #         current_away_def_exp_plays <- current_pbp %>% filter(season==current_season, week==rolling_week_away, defteam==current_away$away_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
    #         current_away_def_exp_play_rate <- current_away_def_exp_play_rate + (current_away_def_exp_plays / total_away_def_plays)
    #         
    #         # ----- PENALTIES & PENALTY YARDS ----- #
    #         current_away_off_penalties <- current_pbp %>% filter(season==current_season, week==rolling_week_away, posteam==current_away$away_team, penalty==1 & penalty_team==current_away$away_team) %>% count()
    #         current_away_off_penalty_yds <- current_pbp %>% filter(season==current_season, week==rolling_week_away, posteam==current_away$away_team, penalty==1 & penalty_team==current_away$away_team) %>% summarise(Off_Pen_Yds=sum(penalty_yards))
    #         current_away_def_penalties <- current_pbp %>% filter(season==current_season, week==rolling_week_away, defteam==current_away$away_team, penalty==1 & penalty_team==current_away$away_team) %>% count()
    #         current_away_def_penalty_yds <- current_pbp %>% filter(season==current_season, week==rolling_week_away, defteam==current_away$away_team, penalty==1 & penalty_team==current_away$away_team) %>% summarise(Def_Pen_Yds=sum(penalty_yards))
    #         current_away_total_penalties <- current_away_total_penalties + (current_away_off_penalties + current_away_def_penalties)
    #         current_away_total_penalty_yds <- current_away_total_penalty_yds + (current_away_off_penalty_yds + current_away_def_penalty_yds)
    #         
    #         # ----- QB HIT + SACK DIFFERENTIAL ----- #
    #         # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
    #         # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
    #         current_away_off_line_metric <- current_away_off_line_metric + (current_pbp %>% filter(season==current_season, week==rolling_week_away, posteam==current_away$away_team, qb_hit==1 | sack==1) %>% count())
    #         current_away_def_line_metric <- current_away_def_line_metric + (current_pbp %>% filter(season==current_season, week==rolling_week_away, defteam==current_away$away_team, qb_hit==1 | sack==1) %>% count())
    #         
    #         ################### HOME STATS ###################
    #         # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
    #         home_general_data <- current_pbp %>% filter(season==current_season, week==rolling_week_home, posteam==current_home$home_team) %>% select(posteam_type, div_game) %>% unique()
    #         total_home_off_plays <- current_pbp %>% filter(season==current_season, week==rolling_week_home, posteam==current_home$home_team, play_type=="pass" | play_type=="run") %>% count()
    #         total_home_def_plays <- current_pbp %>% filter(season==current_season, week==rolling_week_home, defteam==current_home$home_team, play_type=="pass" | play_type=="run") %>% count()
    #         
    #         # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
    #         current_home_off_rush_epa <- current_home_off_rush_epa + (current_pbp %>% filter(season==current_season, week==rolling_week_home, posteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(OFF_RUSH_EPA=mean(epa)))
    #         current_home_off_pass_epa <- current_home_def_rush_epa + (current_pbp %>% filter(season==current_season, week==rolling_week_home, posteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(OFF_PASS_EPA=mean(epa)))
    #         current_home_first_downs_for <- current_pbp %>% filter(season==current_season, week==rolling_week_home, posteam==current_home$home_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
    #         current_home_off_fdr <- current_home_off_fdr + (current_home_first_downs_for / total_home_off_plays)
    #         
    #         # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
    #         current_home_def_rush_epa <- current_home_def_rush_epa + (current_pbp %>% filter(season==current_season, week==rolling_week_home, defteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(DEF_RUSH_EPA=mean(epa)))
    #         current_home_def_pass_epa <- current_home_def_pass_epa + (current_pbp %>% filter(season==current_season, week==rolling_week_home, defteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(DEF_PASS_EPA=mean(epa)))
    #         current_home_first_downs_allowed <- current_pbp %>% filter(season==current_season, week==rolling_week_home, defteam==current_home$home_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
    #         current_home_def_fdr <- current_home_def_fdr + (current_home_first_downs_allowed / total_home_def_plays)
    #         
    #         # ----- TURNOVER DATA ----- #
    #         current_home_giveaways <- current_home_giveaways + (current_pbp %>% filter(season==current_season, week==rolling_week_home, posteam==current_home$home_team, fumble_lost==1 | interception==1) %>% count())
    #         current_home_takeaways <- current_home_takeaways + (current_pbp %>% filter(season==current_season, week==rolling_week_home, defteam==current_home$home_team, fumble_lost==1 | interception==1) %>% count())
    #         
    #         # ----- EXPLOSIVE PLAY RATE ----- #
    #         current_home_off_exp_plays <- current_pbp %>% filter(season==current_season, week==rolling_week_home, posteam==current_home$home_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
    #         current_home_off_exp_play_rate <- current_home_off_exp_play_rate + (current_home_off_exp_plays / total_home_off_plays)
    #         current_home_def_exp_plays <- current_pbp %>% filter(season==current_season, week==rolling_week_home, defteam==current_home$home_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
    #         current_home_def_exp_play_rate <- current_home_def_exp_play_rate + (current_home_def_exp_plays / total_home_def_plays)
    #         
    #         # ----- PENALTIES & PENALTY YARDS ----- #
    #         current_home_off_penalties <- current_pbp %>% filter(season==current_season, week==rolling_week_home, posteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% count()
    #         current_home_off_penalty_yds <- current_pbp %>% filter(season==current_season, week==rolling_week_home, posteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% summarise(Off_Pen_Yds=sum(penalty_yards))
    #         current_home_def_penalties <- current_pbp %>% filter(season==current_season, week==rolling_week_home, defteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% count()
    #         current_home_def_penalty_yds <- current_pbp %>% filter(season==current_season, week==rolling_week_home, defteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% summarise(Def_Pen_Yds=sum(penalty_yards))
    #         current_home_total_penalties <- current_home_total_penalties + (current_home_off_penalties + current_home_def_penalties)
    #         current_home_total_penalty_yds <- current_home_total_penalty_yds + (current_home_off_penalty_yds + current_home_def_penalty_yds)
    #         
    #         # ----- QB HIT + SACK DIFFERENTIAL ----- #
    #         # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
    #         # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
    #         current_home_off_line_metric <- current_home_off_line_metric + (current_pbp %>% filter(season==current_season, week==rolling_week_home, posteam==current_home$home_team, qb_hit==1 | sack==1) %>% count())
    #         current_home_def_line_metric <- current_home_def_line_metric + (current_pbp %>% filter(season==current_season, week==rolling_week_home, defteam==current_home$home_team, qb_hit==1 | sack==1) %>% count())
    #         
    #         # Edit rolling weeks if needed and reset the boolean condition
    #         if (change_away_rolling_week == TRUE) {
    #             # subtracting the x index time -1 and adding one takes it back to the same index as the other 
    #             rolling_week_away = (rolling_week_away - (x * -1)) + 1
    #             change_away_rolling_week = FALSE
    #         } else {
    #             rolling_week_away = rolling_week_away + 1
    #         }
    #         
    #         if (change_home_rolling_week == TRUE) {
    #             # subtracting the x index time -1 and adding one takes it back to the same index as the other
    #             rolling_week_home = (rolling_week_home - (x * -1)) + 1
    #             change_home_rolling_week = FALSE
    #         } else {
    #             rolling_week_home = rolling_week_home + 1
    #         }
    #         
    #         # Increment rolling week
    #         #rolling_week_away = rolling_week_away + 1
    #         #rolling_week_home = rolling_week_home + 1
    #         
    #     }
    #     
    #     # Divide data by 4 here to account for rolling average
    #     # These stats will work out to averages for the last 3 games each team has played
    #     # Make away data frame
    #     current_away_data <- data.frame(
    #         OFF_RUSH_EPA <- current_away_off_rush_epa$OFF_RUSH_EPA / 4,
    #         OFF_PASS_EPA <- current_away_off_pass_epa$OFF_PASS_EPA / 4,
    #         OFF_FDR <- current_away_off_fdr$n / 4,
    #         DEF_RUSH_EPA <- current_away_def_rush_epa$DEF_RUSH_EPA / 4,
    #         DEF_PASS_EPA <- current_away_def_pass_epa$DEF_PASS_EPA / 4,
    #         DEF_FDR <- current_away_def_fdr$n / 4,
    #         OFF_TO <- current_away_giveaways$n / 4,
    #         DEF_TO <- current_away_takeaways$n / 4,
    #         OFF_EXP_RATE <- current_away_off_exp_play_rate$n / 4,
    #         DEF_EXP_RATE <- current_away_def_exp_play_rate$n / 4,
    #         PENS <- current_away_total_penalties$n / 4,
    #         PEN_YDS <- current_away_total_penalty_yds$Off_Pen_Yds / 4,
    #         OL_METRIC <- current_away_off_line_metric$n / 4,
    #         DL_METRIC <- current_away_def_line_metric$n / 4
    #     )
    #     
    #     # Make home data frame
    #     current_home_data <- data.frame(
    #         OFF_RUSH_EPA <- current_home_off_rush_epa$OFF_RUSH_EPA / 4,
    #         OFF_PASS_EPA <- current_home_def_pass_epa$DEF_PASS_EPA / 4,
    #         OFF_FDR <- current_home_off_fdr$n / 4,
    #         DEF_RUSH_EPA <- current_home_def_rush_epa$DEF_RUSH_EPA / 4,
    #         DEF_PASS_EPA <- current_home_def_pass_epa$DEF_PASS_EPA / 4,
    #         DEF_FDR <- current_home_def_fdr$n / 4,
    #         OFF_TO <- current_home_giveaways$n / 4,
    #         DEF_TO <- current_home_takeaways$n / 4,
    #         OFF_EXP_RATE <- current_home_off_exp_play_rate$n / 4,
    #         DEF_EXP_RATE <- current_home_def_exp_play_rate$n / 4,
    #         PENS <- current_home_total_penalties$n / 4,
    #         PEN_YDS <- current_home_total_penalty_yds$Off_Pen_Yds / 4,
    #         OL_METRIC <- current_home_off_line_metric$n / 4,
    #         DL_METRIC <- current_home_def_line_metric$n / 4,
    #         DIV_GAME <- div_game
    #     )
    #     
    # }
    
    current_game_data <- append(current_away_data, current_home_data)
    current_week_data <- rbind(current_week_data, current_game_data)
    
}

# Cycle through 
for (y in 1:10) {
    #print(glue("---------- Model ", y, " predictions ----------", "\n\n"))
    predictions <- reticulate::py$make_predictions(reticulate::r_to_py(current_week_data), y)
    
    for (x in 1:nrow(predictions)) {
        
        current_away = games[x,] %>% select(away_team)
        current_home = games[x,] %>% select(home_team)
        
        current_prediction = predictions[x,]
        # if (current_prediction[1] < current_prediction[2]) {
        #     print(glue(" ", current_away$away_team, " @ ", current_home$home_team, " -- ", current_home$home_team, " wins at home with ", round(as.double(current_prediction[2]) * 100, 2), "% confidence"))
        # } else if (current_prediction[1] > current_prediction[2]) {
        #     print(glue(" ", current_away$away_team, " @ ", current_home$home_team, " -- ", current_away$away_team, " wins on the road with ", round(as.double(current_prediction[1]) * 100, 2), "% confidence"))
        # }
    }
    writeLines("\n")
    
    current_week_predictions <- rbind(current_week_predictions, predictions)
}

print(glue("---------- Majority winners and average confidence ratings for week ", current_week, " ", current_season , " ----------", "\n\n"))
for (x in 1:nrow(games)) {

    # Variables to hold current away and home teams
    current_away = games[x,] %>% select(away_team)
    current_home = games[x,] %>% select(home_team)

    # Variables to hold average confidence ratings
    current_away_confidence = 0
    current_away_wins = 0
    current_home_confidence = 0
    current_home_wins = 0
    
    current_pick <- data.frame(matrix(ncol = 7))
    
    # This loop will help index the current_week_predictions data frame
    for (y in 1:10) {
    
        if(y == 1){
            if (current_week_predictions[x, 1] > current_week_predictions[x, 2]) {
                current_away_wins = current_away_wins + 1
                current_away_confidence = current_away_confidence + (current_week_predictions[x, 1])
                current_home_confidence = current_home_confidence + (current_week_predictions[x, 2])
            } else {
                current_home_wins = current_home_wins + 1
                current_away_confidence = current_away_confidence + (current_week_predictions[x, 1])
                current_home_confidence = current_home_confidence + (current_week_predictions[x, 2])
            }
        }
        else {
            if (current_week_predictions[((y-1)*nrow(games))+x, 1] > current_week_predictions[((y-1)*nrow(games))+x, 2]) {
                current_away_wins = current_away_wins + 1
                current_away_confidence = current_away_confidence + (current_week_predictions[((y-1)*nrow(games))+x, 1])
                current_home_confidence = current_home_confidence + (current_week_predictions[((y-1)*nrow(games))+x, 2])
            } else {
                current_home_wins = current_home_wins + 1
                current_away_confidence = current_away_confidence + (current_week_predictions[((y-1)*nrow(games))+x, 1])
                current_home_confidence = current_home_confidence + (current_week_predictions[((y-1)*nrow(games))+x, 2])
            }
        }
        
    }
    
    if (current_away_confidence > current_home_confidence) {
        print(glue(" ", current_away$away_team, " @ ", current_home$home_team, " -- ", current_away$away_team, " has (", current_away_wins, "/10) predicted wins on the road with ", round((current_away_confidence * 100) / 10, 2), "% average confidence"))
        current_pick <- data.frame(
            AWAY_TEAM <- current_away$away_team,
            AWAY_WIN_COUNT <- current_away_wins,
            AWAY_WIN <- round((current_away_confidence * 100) / 10, 2),
            HOME_TEAM <- current_home$home_team,
            HOME_WIN_COUNT <- current_home_wins,
            HOME_WIN <- round((current_home_confidence * 100) / 10, 2),
            WINNER <- current_away$away_team
        )
    } else if (current_away_confidence < current_home_confidence) {
        print(glue(" ", current_away$away_team, " @ ", current_home$home_team, " -- ", current_home$home_team, " has (", current_home_wins, "/10) predicted wins at home with ", round((current_home_confidence * 100) / 10, 2), "% average confidence"))
        current_pick <- data.frame(
            AWAY_TEAM <- current_away$away_team,
            AWAY_TEAM_WINS <- current_away_wins,
            AWAY_TEAM_CONFIDENCE <- round((current_away_confidence * 100) / 10, 2),
            HOME_TEAM <- current_home$home_team,
            HOME_TEAM_WINS <- current_home_wins,
            HOME_TEAM_CONFIDENCE <- round((current_home_confidence * 100) / 10, 2),
            WINNER <- current_home$home_team
        )
    } else if (current_away_wins == current_home_wins) {
        if (current_away_confidence / 10 > current_home_confidence / 10) {
            print(glue(" ", current_away$away_team, " @ ", current_home$home_team, " -- ", current_away$away_team, " has (", current_away_wins, "/10) predicted wins on the road with ", round((current_away_confidence * 100) / 10, 2), "% average confidence"))
            current_pick <- data.frame(
                AWAY_TEAM <- current_away$away_team,
                AWAY_WIN_COUNT <- current_away_wins,
                AWAY_WIN <- round((current_away_confidence * 100) / 10, 2),
                HOME_TEAM <- current_home$home_team,
                HOME_WIN_COUNT <- current_home_wins,
                HOME_WIN <- round((current_home_confidence * 100) / 10, 2),
                WINNER <- current_away$away_team
            )
        } else {
            print(glue(" ", current_away$away_team, " @ ", current_home$home_team, " -- ", current_home$home_team, " has (", current_home_wins, "/10) predicted wins at home with ", round((current_home_confidence * 100) / 10, 2), "% average confidence"))
            current_pick <- data.frame(
                AWAY_TEAM <- current_away$away_team,
                AWAY_TEAM_WINS <- current_away_wins,
                AWAY_TEAM_CONFIDENCE <- round((current_away_confidence * 100) / 10, 2),
                HOME_TEAM <- current_home$home_team,
                HOME_TEAM_WINS <- current_home_wins,
                HOME_TEAM_CONFIDENCE <- round((current_home_confidence * 100) / 10, 2),
                WINNER <- current_home$home_team
            )
        }
    }
    
    colnames(current_pick) <- c("AWAY_TEAM","AWAY_WIN_COUNT","AWAY_WIN_CONFIDENCE","HOME_TEAM", "HOME_WIN_COUNT", "HOME_WIN_CONFIDENCE", "WINNER")
    colnames(current_week_picks) <- c("AWAY_TEAM","AWAY_WIN_COUNT","AWAY_WIN_CONFIDENCE","HOME_TEAM", "HOME_WIN_COUNT", "HOME_WIN_CONFIDENCE", "WINNER")
    current_week_picks <- rbind(current_week_picks, current_pick)
    
}

# Remove the empty first row from current_week_picks
current_week_picks <- current_week_picks[-c(1),]

# Write current_week_picks to the excel sheet
if (to_write == "yes" | to_write == "y"){
    write.xlsx(current_week_picks, glue("C:/Users/Ryne/Documents/R Projects and Files/Staley_2.0/Weekly Picks/", current_season ,"_Weekly_Data.xlsx"), sheetName = glue("Week ", current_week), col.names = TRUE, row.names = FALSE, append = TRUE)
    print(glue("\n\nExcel sheet created and written to file for week ", current_week, " of the ", current_season, " season"))
}

remove(current_pick)
remove(current_away_confidence)
remove(current_away_wins)
remove(current_home_confidence)
remove(current_home_wins)
remove(games)
remove(predictions)
remove(div_game)
remove(current_away)
remove(current_home)
if(current_week == 1) {
    remove(last_season)
}
# if (current_week >= 6){
#     remove(rolling_week_away)
#     remove(rolling_week_home)
# }
remove(current_week)
remove(away_general_data)
remove(total_away_off_plays)
remove(total_away_def_plays)
remove(current_away_off_rush_epa)
remove(current_away_off_pass_epa)
remove(current_away_first_downs_for)
remove(current_away_off_fdr)
remove(current_away_def_rush_epa)
remove(current_away_def_pass_epa)
remove(current_away_first_downs_allowed)
remove(current_away_def_fdr)
remove(current_away_giveaways)
remove(current_away_takeaways)
remove(current_away_off_exp_plays)
remove(current_away_off_exp_play_rate)
remove(current_away_def_exp_plays)
remove(current_away_def_exp_play_rate)
remove(current_away_off_penalties)
remove(current_away_off_penalty_yds)
remove(current_away_def_penalties)
remove(current_away_def_penalty_yds)
remove(current_away_total_penalties)
remove(current_away_total_penalty_yds)
remove(current_away_off_line_metric)
remove(current_away_def_line_metric)
remove(home_general_data)
remove(total_home_off_plays)
remove(total_home_def_plays)
remove(current_home_off_rush_epa)
remove(current_home_off_pass_epa)
remove(current_home_first_downs_for)
remove(current_home_off_fdr)
remove(current_home_def_rush_epa)
remove(current_home_def_pass_epa)
remove(current_home_first_downs_allowed)
remove(current_home_def_fdr)
remove(current_home_giveaways)
remove(current_home_takeaways)
remove(current_home_off_exp_plays)
remove(current_home_off_exp_play_rate)
remove(current_home_def_exp_plays)
remove(current_home_def_exp_play_rate)
remove(current_home_off_penalties)
remove(current_home_off_penalty_yds)
remove(current_home_def_penalties)
remove(current_home_def_penalty_yds)
remove(current_home_total_penalties)
remove(current_home_total_penalty_yds)
remove(current_home_off_line_metric)
remove(current_home_def_line_metric)
remove(current_away_data)
remove(current_game)
remove(current_game_data)
remove(current_home_data)
remove(DIV_GAME)
remove(current_season)
remove(DEF_EXP_RATE)
remove(DEF_FDR)
remove(DEF_PASS_EPA)
remove(DEF_RUSH_EPA)
remove(DEF_TO)
remove(DL_METRIC)
remove(OFF_EXP_RATE)
remove(OFF_FDR)
remove(OFF_PASS_EPA)
remove(OFF_RUSH_EPA)
remove(OFF_TO)
remove(OL_METRIC)
remove(PEN_YDS)
remove(PENS)
remove(x)
remove(y)
remove(current_prediction)
remove(AWAY_TEAM)
remove(AWAY_TEAM_CONFIDENCE)
remove(AWAY_TEAM_WINS)
remove(AWAY_WIN)
remove(AWAY_WIN_COUNT)
remove(HOME_TEAM)
remove(HOME_TEAM_CONFIDENCE)
remove(HOME_TEAM_WINS)
remove(HOME_WIN)
remove(HOME_WIN_COUNT)
remove(WINNER)
remove(to_write)
remove(current_pbp)
remove(last_season_pbp)