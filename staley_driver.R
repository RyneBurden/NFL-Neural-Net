# Load packages
library('ggplot2')
library('dplyr')
library('tibble')
library('qs')
library('reticulate')
library('magrittr')
library('glue')

# To run the script
# Rscript make_weekly_data.R <season> <week>

# Data still needed - nflSchedule

# Read in current data variables for logic processing
args <- commandArgs(trailingOnly = TRUE)
current_season <- args[1]
current_week <- args[2]
current_week <- as.integer(current_week)
current_season <- as.integer(current_season)

# Make data frames for looping
nfl_schedule_whole <- nflreadr::load_schedules(as.integer(current_season))
nfl_schedule <- nfl_schedule_whole[, c("season", "week","away_team", "home_team", "div_game")]
games <- nfl_schedule %>% filter(week == current_week, season == current_season)
current_week_data <- data.frame()
predictions <- data.frame()
current_week_predictions <- data.frame()
current_week_picks <- data.frame(matrix(ncol = 7))

# Sopranos reference
writeLines("Bada bing I'm on it boss")

# Loop through all games to make data frames
for(current_game in 1:nrow(games)) {

    # Get the current home and away teams
    current_away <- games[current_game, ] %>% select(away_team)
    current_home <- games[current_game, ] %>% select(home_team)
    div_game <- games[current_game, ] %>% select(div_game)
    
    # make variables for each data point needed away and home
    # calculate each variable using a for loop and current_pbp
    # NOTE: first 2 weeks will need to use <season-1> stats
    
    if (current_week == 1) {
        
        # Load previous season data
        current_pbp <- nflfastR::load_pbp(as.integer(current_season) - 1)
        
        # Rolling week so the remove statements below work
        rolling_week <- 0
        
        # I'm using last season here since it's the first week of the season
        last_season <- current_season - 1

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
            AWAY_TEAM <- current_away,
            OFF_RUSH_EPA_AWAY <- current_away_off_rush_epa$OFF_RUSH_EPA,
            OFF_PASS_EPA_AWAY <- current_away_off_pass_epa$OFF_PASS_EPA,
            OFF_FDR_AWAY <- current_away_off_fdr$n,
            DEF_RUSH_EPA_AWAY <- current_away_def_rush_epa$DEF_RUSH_EPA,
            DEF_PASS_EPA_AWAY <- current_away_def_pass_epa$DEF_PASS_EPA,
            DEF_FDR_AWAY <- current_away_def_fdr$n,
            OFF_TO_AWAY <- current_away_giveaways$n,
            DEF_TO_AWAY <- current_away_takeaways$n,
            OFF_EXP_RATE_AWAY <- current_away_off_exp_play_rate$n,
            DEF_EXP_RATE_AWAY <- current_away_def_exp_play_rate$n,
            PENS_AWAY <- current_away_total_penalties$n,
            PEN_YDS_AWAY <- current_away_total_penalty_yds$Off_Pen_Yds,
            OL_METRIC_AWAY <- current_away_off_line_metric$n,
            DL_METRIC_AWAY <- current_away_def_line_metric$n
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
            HOME_TEAM <- current_home,
            OFF_RUSH_EPA_HOME <- current_home_off_rush_epa$OFF_RUSH_EPA,
            OFF_PASS_EPA_HOME <- current_home_def_pass_epa$DEF_PASS_EPA,
            OFF_FDR_HOME <- current_home_off_fdr$n,
            DEF_RUSH_EPA_HOME <- current_home_def_rush_epa$DEF_RUSH_EPA,
            DEF_PASS_EPA_HOME <- current_home_def_pass_epa$DEF_PASS_EPA,
            DEF_FDR_HOME <- current_home_def_fdr$n,
            OFF_TO_HOME <- current_home_giveaways$n,
            DEF_TO_HOME <- current_home_takeaways$n,
            OFF_EXP_RATE_HOME <- current_home_off_exp_play_rate$n,
            DEF_EXP_RATE_HOME <- current_home_def_exp_play_rate$n,
            PENS_HOME <- current_home_total_penalties$n,
            PEN_YDS_HOME <- current_home_total_penalty_yds$Off_Pen_Yds,
            OL_METRIC_HOME <- current_home_off_line_metric$n,
            DL_METRIC_HOME <- current_home_def_line_metric$n,
            DIV_GAME_HOME <- div_game
        )  
    }
    else {

        # Load pbp
        current_pbp <- nflfastR::load_pbp(current_season)
        
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
            AWAY_TEAM <- current_away,
            OFF_RUSH_EPA_AWAY <- current_away_off_rush_epa$OFF_RUSH_EPA / (current_week - 1),
            OFF_PASS_EPA_AWAY <- current_away_off_pass_epa$OFF_PASS_EPA / (current_week - 1),
            OFF_FDR_AWAY <- current_away_off_fdr$n / (current_week - 1),
            DEF_RUSH_EPA_AWAY <- current_away_def_rush_epa$DEF_RUSH_EPA / (current_week - 1),
            DEF_PASS_EPA_AWAY <- current_away_def_pass_epa$DEF_PASS_EPA / (current_week - 1),
            DEF_FDR_AWAY <- current_away_def_fdr$n / (current_week - 1),
            OFF_TO_AWAY <- current_away_giveaways$n,
            DEF_TO_AWAY <- current_away_takeaways$n,
            OFF_EXP_RATE_AWAY <- current_away_off_exp_play_rate$n / (current_week - 1),
            DEF_EXP_RATE_AWAY <- current_away_def_exp_play_rate$n / (current_week - 1),
            PENS_AWAY <- current_away_total_penalties$n,
            PEN_YDS_AWAY <- current_away_total_penalty_yds$Off_Pen_Yds,
            OL_METRIC_AWAY <- current_away_off_line_metric$n,
            DL_METRIC_AWAY <- current_away_def_line_metric$n
        )
        
        ################### HOME STATS ###################
        # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
        home_general_data <- current_pbp %>% filter(week < current_week, posteam==current_home$home_team) %>% select(posteam_type, div_game) %>% unique()
        total_home_off_plays <- current_pbp %>% filter(week < current_week, posteam==current_home$home_team, play_type=="pass" | play_type=="run") %>% count()
        total_home_def_plays <- current_pbp %>% filter(week < current_week, defteam==current_home$home_team, play_type=="pass" | play_type=="run") %>% count()
        
        # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
        current_home_off_rush_epa <- current_pbp %>% filter(week < current_week, posteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(OFF_RUSH_EPA=mean(epa))
        current_home_off_pass_epa <- current_pbp %>% filter(week < current_week, posteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(OFF_PASS_EPA=mean(epa))
        current_home_first_downs_for <- current_pbp %>% filter(week < current_week, posteam==current_home$home_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
        current_home_off_fdr <- current_home_first_downs_for / total_home_off_plays
        
        # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
        current_home_def_rush_epa <- current_pbp %>% filter(week < current_week, defteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(DEF_RUSH_EPA=mean(epa))
        current_home_def_pass_epa <- current_pbp %>% filter(week < current_week, defteam==current_home$home_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(DEF_PASS_EPA=mean(epa))
        current_home_first_downs_allowed <- current_pbp %>% filter(week < current_week, defteam==current_home$home_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
        current_home_def_fdr <- current_home_first_downs_allowed / total_home_def_plays
        
        # ----- TURNOVER DATA ----- #
        current_home_giveaways <- (current_pbp %>% filter(week < current_week, posteam==current_home$home_team, fumble_lost==1 | interception==1) %>% count()) / (current_week - 1)
        current_home_takeaways <- (current_pbp %>% filter(week < current_week, defteam==current_home$home_team, fumble_lost==1 | interception==1) %>% count()) / (current_week - 1)
        
        # ----- EXPLOSIVE PLAY RATE ----- #
        current_home_off_exp_plays <- current_pbp %>% filter(week < current_week, posteam==current_home$home_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
        current_home_off_exp_play_rate <- current_home_off_exp_plays / total_home_off_plays
        current_home_def_exp_plays <- current_pbp %>% filter(week < current_week, defteam==current_home$home_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
        current_home_def_exp_play_rate <- current_home_def_exp_plays / total_home_def_plays
        
        # ----- PENALTIES & PENALTY YARDS ----- #
        current_home_off_penalties <- current_pbp %>% filter(week < current_week, posteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% count()
        current_home_off_penalty_yds <- current_pbp %>% filter(week < current_week, posteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% summarise(Off_Pen_Yds=sum(penalty_yards))
        current_home_def_penalties <- current_pbp %>% filter(week < current_week, defteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% count()
        current_home_def_penalty_yds <- current_pbp %>% filter(week < current_week, defteam==current_home$home_team, penalty==1 & penalty_team==current_home$home_team) %>% summarise(Def_Pen_Yds=sum(penalty_yards))
        current_home_total_penalties <- (current_home_off_penalties + current_home_def_penalties) / (current_week - 1)
        current_home_total_penalty_yds <- (current_home_off_penalty_yds + current_home_def_penalty_yds) / (current_week - 1)
        
        # ----- QB HIT + SACK DIFFERENTIAL ----- #
        # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
        # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
        current_home_off_line_metric <- (current_pbp %>% filter(week < current_week, posteam==current_home$home_team, qb_hit==1 | sack==1) %>% count()) / (current_week - 1)
        current_home_def_line_metric <- (current_pbp %>% filter(week < current_week, defteam==current_home$home_team, qb_hit==1 | sack==1) %>% count()) / (current_week - 1)
        
        # Make home data frame
        current_home_data <- data.frame(
            HOME_TEAM <- current_home,
            OFF_RUSH_EPA_HOME <- current_home_off_rush_epa$OFF_RUSH_EPA / (current_week - 1),
            OFF_PASS_EPA_HOME <- current_home_def_pass_epa$DEF_PASS_EPA / (current_week - 1),
            OFF_FDR_HOME <- current_home_off_fdr$n / (current_week - 1),
            DEF_RUSH_EPA_HOME <- current_home_def_rush_epa$DEF_RUSH_EPA / (current_week - 1),
            DEF_PASS_EPA_HOME <- current_home_def_pass_epa$DEF_PASS_EPA / (current_week - 1),
            DEF_FDR_HOME <- current_home_def_fdr$n / (current_week - 1),
            OFF_TO_HOME <- current_home_giveaways$n,
            DEF_TO_HOME <- current_home_takeaways$n,
            OFF_EXP_RATE_HOME <- current_home_off_exp_play_rate$n / (current_week - 1),
            DEF_EXP_RATE_HOME <- current_home_def_exp_play_rate$n / (current_week - 1),
            PENS_HOME <- current_home_total_penalties$n,
            PEN_YDS_HOME <- current_home_total_penalty_yds$Off_Pen_Yds,
            OL_METRIC_HOME <- current_home_off_line_metric$n,
            DL_METRIC_HOME <- current_home_def_line_metric$n,
            DIV_GAME_HOME <- div_game
        )
        
    }
   
    # Append current_game_data to current_week_data
    # This includes all stats needed for predictions (29 data points, 14 per team, 1 for divisional game)
    current_game_data <- append(current_away_data, current_home_data)
    current_week_data <- rbind(current_week_data, current_game_data)
    
}

# Set column names to be pretty
colnames(current_week_data) <- c(
    'away_team',
    'off_rush_epa_away',
    'off_pass_epa_away',
    'off_fdr_away',
    'def_rush_epa_away',
    'def_pass_epa_away',
    'def_fdr_away',
    'off_to_away',
    'def_to_away',
    'off_exp_rate_away',
    'def_exp_rate_away',
    'pens_away',
    'pen_yds_away',
    'ol_metric_away',
    'dl_metric_away',
    'home_team',
    'off_rush_epa_home',
    'off_pass_epa_home',
    'off_fdr_home',
    'def_rush_epa_home',
    'def_pass_epa_home',
    'def_fdr_home',
    'off_to_home',
    'def_to_home',
    'off_exp_rate_home',
    'def_exp_rate_home',
    'pens_home',
    'pen_yds_home',
    'ol_metric_home',
    'dl_metric_home',
    'div_game'
)

# Uncomment this line to debug the data going to the model
# write.csv(current_week_data, 'test.csv', row.names=FALSE)

# venv for dev purposes
reticulate::use_virtualenv("venv/") 
reticulate::py_run_file("staley_says_v3.py")
# Predictions will be made here and saved to the DB
reticulate::py$predict_games(reticulate::r_to_py(current_week_data), current_season, current_week) 