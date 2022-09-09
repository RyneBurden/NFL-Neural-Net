# Load packages
if (!require("dplyr")) {
    install.packages("dplyr", repos = "http://cran.us.r-project.org")
    library("dplyr")
}
if (!require("tibble")) {
    install.packages("tibble", repos = "http://cran.us.r-project.org")
    library("tibble")
}
if (!require("qs")) {
    install.packages("qs", repos = "http://cran.us.r-project.org")
    library("qs")
}
if (!require("reticulate")) {
    install.packages("reticulate", repos = "http://cran.us.r-project.org")
    library("reticulate")
}
if (!require("magrittr")) {
    install.packages("magrittr", repos = "http://cran.us.r-project.org")
    library("magrittr")
}
if (!require("glue")) {
    install.packages("glue", repos = "http://cran.us.r-project.org")
    library("glue")
}
if (!require("nflfastR")) {
    install.packages("nflfastR", repos = "http://cran.us.r-project.org")
    library("nflfastR")
}
if (!require("nflreadr")) {
    install.packages("nflreadr", repos = "http://cran.us.r-project.org")
    library("nflreadr")
}


# To run the script
# Rscript driver.R <season> <week>

# Data still needed - nflSchedule

# Read in current data variables for logic processing
args <- commandArgs(trailingOnly = TRUE)
current_season <- args[1]
current_week <- args[2]
current_week <- as.integer(current_week)
current_season <- as.integer(current_season)

# Make data frames for looping
nfl_schedule_whole <- nflreadr::load_schedules(as.integer(current_season))
nfl_schedule <- nfl_schedule_whole[, c("season", "week", "away_team", "home_team", "div_game")]
games <- nfl_schedule %>% filter(week == current_week, season == current_season)
current_week_data <- data.frame()
predictions <- data.frame()
current_week_predictions <- data.frame()
current_week_picks <- data.frame(matrix(ncol = 7))
rolling_week_modifier <- 4

# Sopranos reference
writeLines("Bada bing I'm on it boss")

# Loop through all games to make data frames
for (current_game in 1:nrow(games)) {

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
        away_general_data <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_away$away_team) %>%
            select(posteam_type, div_game) %>%
            unique()
        total_away_off_plays <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_away$away_team, play_type == "pass" | play_type == "run") %>%
            count()
        total_away_def_plays <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_away$away_team, play_type == "pass" | play_type == "run") %>%
            count()

        # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
        current_away_off_rush_epa <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_away$away_team, !is.na(epa), penalty == 0, play_type == "run") %>%
            summarise(OFF_RUSH_EPA = mean(epa))
        current_away_off_pass_epa <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_away$away_team, !is.na(epa), penalty == 0, play_type == "pass") %>%
            summarise(OFF_PASS_EPA = mean(epa))
        current_away_first_downs_for <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_away$away_team, first_down == 1, play_type == "pass" | play_type == "run") %>%
            count()
        current_away_off_fdr <- current_away_first_downs_for / total_away_off_plays

        # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
        current_away_def_rush_epa <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_away$away_team, !is.na(epa), penalty == 0, play_type == "run") %>%
            summarise(DEF_RUSH_EPA = mean(epa))
        current_away_def_pass_epa <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_away$away_team, !is.na(epa), penalty == 0, play_type == "pass") %>%
            summarise(DEF_PASS_EPA = mean(epa))
        current_away_first_downs_allowed <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_away$away_team, first_down == 1, play_type == "pass" | play_type == "run") %>%
            count()
        current_away_def_fdr <- current_away_first_downs_allowed / total_away_def_plays

        # ----- TURNOVER DATA ----- #
        current_away_giveaways <- (current_pbp %>% filter(season == last_season, week <= 18, posteam == current_away$away_team, fumble_lost == 1 | interception == 1) %>% count()) / 17
        current_away_takeaways <- (current_pbp %>% filter(season == last_season, week <= 18, defteam == current_away$away_team, fumble_lost == 1 | interception == 1) %>% count()) / 17

        # ----- EXPLOSIVE PLAY RATE ----- #
        current_away_off_exp_plays <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_away$away_team, yards_gained >= 15, interception == 0 & fumble_lost == 0, penalty == 0, play_type == "pass" | play_type == "run") %>%
            count()
        current_away_off_exp_play_rate <- current_away_off_exp_plays / total_away_off_plays
        current_away_def_exp_plays <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_away$away_team, yards_gained >= 15, interception == 0 & fumble_lost == 0, penalty == 0, play_type == "pass" | play_type == "run") %>%
            count()
        current_away_def_exp_play_rate <- current_away_def_exp_plays / total_away_def_plays

        # ----- PENALTIES & PENALTY YARDS ----- #
        current_away_off_penalties <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_away$away_team, penalty == 1 & penalty_team == current_away$away_team) %>%
            count()
        current_away_off_penalty_yds <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_away$away_team, penalty == 1 & penalty_team == current_away$away_team) %>%
            summarise(Off_Pen_Yds = sum(penalty_yards))
        current_away_def_penalties <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_away$away_team, penalty == 1 & penalty_team == current_away$away_team) %>%
            count()
        current_away_def_penalty_yds <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_away$away_team, penalty == 1 & penalty_team == current_away$away_team) %>%
            summarise(Def_Pen_Yds = sum(penalty_yards))
        current_away_total_penalties <- (current_away_off_penalties + current_away_def_penalties) / 17
        current_away_total_penalty_yds <- (current_away_off_penalty_yds + current_away_def_penalty_yds) / 17

        # ----- QB HIT + SACK DIFFERENTIAL ----- #
        # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
        # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
        current_away_off_line_metric <- (current_pbp %>% filter(season == last_season, week <= 18, posteam == current_away$away_team, qb_hit == 1 | sack == 1) %>% count()) / 17
        current_away_def_line_metric <- (current_pbp %>% filter(season == last_season, week <= 18, defteam == current_away$away_team, qb_hit == 1 | sack == 1) %>% count()) / 17

        # Make away data frame
        current_away_data <- data.frame(
            AWAY_TEAM <- current_away,
            AWAY_OFF_RUSH_EPA <- current_away_off_rush_epa$OFF_RUSH_EPA,
            AWAY_OFF_PASS_EPA <- current_away_off_pass_epa$OFF_PASS_EPA,
            AWAY_OFF_FDR <- current_away_off_fdr$n,
            AWAY_DEF_RUSH_EPA <- current_away_def_rush_epa$DEF_RUSH_EPA,
            AWAY_DEF_PASS_EPA <- current_away_def_pass_epa$DEF_PASS_EPA,
            AWAY_DEF_FDR <- current_away_def_fdr$n,
            AWAY_OFF_TO <- current_away_giveaways$n,
            AWAY_DEF_TO <- current_away_takeaways$n,
            AWAY_OFF_EXP_RATE <- current_away_off_exp_play_rate$n,
            AWAY_DEF_EXP_RATE <- current_away_def_exp_play_rate$n,
            AWAY_PENS <- current_away_total_penalties$n,
            AWAY_PEN_YDS <- current_away_total_penalty_yds$Off_Pen_Yds,
            AWAY_OL_METRIC <- current_away_off_line_metric$n,
            AWAY_DL_METRIC <- current_away_def_line_metric$n
        )


        ################### HOME STATS ###################
        # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
        home_general_data <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_home$home_team) %>%
            select(posteam_type, div_game) %>%
            unique()
        total_home_off_plays <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_home$home_team, play_type == "pass" | play_type == "run") %>%
            count()
        total_home_def_plays <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_home$home_team, play_type == "pass" | play_type == "run") %>%
            count()

        # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
        current_home_off_rush_epa <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_home$home_team, !is.na(epa), penalty == 0, play_type == "run") %>%
            summarise(OFF_RUSH_EPA = mean(epa))
        current_home_off_pass_epa <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_home$home_team, !is.na(epa), penalty == 0, play_type == "pass") %>%
            summarise(OFF_PASS_EPA = mean(epa))
        current_home_first_downs_for <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_home$home_team, first_down == 1, play_type == "pass" | play_type == "run") %>%
            count()
        current_home_off_fdr <- current_home_first_downs_for / total_home_off_plays

        # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
        current_home_def_rush_epa <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_home$home_team, !is.na(epa), penalty == 0, play_type == "run") %>%
            summarise(DEF_RUSH_EPA = mean(epa))
        current_home_def_pass_epa <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_home$home_team, !is.na(epa), penalty == 0, play_type == "pass") %>%
            summarise(DEF_PASS_EPA = mean(epa))
        current_home_first_downs_allowed <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_home$home_team, first_down == 1, play_type == "pass" | play_type == "run") %>%
            count()
        current_home_def_fdr <- current_home_first_downs_allowed / total_home_def_plays

        # ----- TURNOVER DATA ----- #
        current_home_giveaways <- (current_pbp %>% filter(season == last_season, week <= 18, posteam == current_home$home_team, fumble_lost == 1 | interception == 1) %>% count()) / 17
        current_home_takeaways <- (current_pbp %>% filter(season == last_season, week <= 18, defteam == current_home$home_team, fumble_lost == 1 | interception == 1) %>% count()) / 17

        # ----- EXPLOSIVE PLAY RATE ----- #
        current_home_off_exp_plays <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_home$home_team, yards_gained >= 15, interception == 0 & fumble_lost == 0, penalty == 0, play_type == "pass" | play_type == "run") %>%
            count()
        current_home_off_exp_play_rate <- current_home_off_exp_plays / total_home_off_plays
        current_home_def_exp_plays <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_home$home_team, yards_gained >= 15, interception == 0 & fumble_lost == 0, penalty == 0, play_type == "pass" | play_type == "run") %>%
            count()
        current_home_def_exp_play_rate <- current_home_def_exp_plays / total_home_def_plays

        # ----- PENALTIES & PENALTY YARDS ----- #
        current_home_off_penalties <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_home$home_team, penalty == 1 & penalty_team == current_home$home_team) %>%
            count()
        current_home_off_penalty_yds <- current_pbp %>%
            filter(season == last_season, week <= 18, posteam == current_home$home_team, penalty == 1 & penalty_team == current_home$home_team) %>%
            summarise(Off_Pen_Yds = sum(penalty_yards))
        current_home_def_penalties <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_home$home_team, penalty == 1 & penalty_team == current_home$home_team) %>%
            count()
        current_home_def_penalty_yds <- current_pbp %>%
            filter(season == last_season, week <= 18, defteam == current_home$home_team, penalty == 1 & penalty_team == current_home$home_team) %>%
            summarise(Def_Pen_Yds = sum(penalty_yards))
        current_home_total_penalties <- (current_home_off_penalties + current_home_def_penalties) / 17
        current_home_total_penalty_yds <- (current_home_off_penalty_yds + current_home_def_penalty_yds) / 17

        # ----- QB HIT + SACK DIFFERENTIAL ----- #
        # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
        # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
        current_home_off_line_metric <- (current_pbp %>% filter(season == last_season, week <= 18, posteam == current_home$home_team, qb_hit == 1 | sack == 1) %>% count()) / 17
        current_home_def_line_metric <- (current_pbp %>% filter(season == last_season, week <= 18, defteam == current_home$home_team, qb_hit == 1 | sack == 1) %>% count()) / 17

        # Make home data frame
        current_home_data <- data.frame(
            HOME_TEAM <- current_home,
            HOME_OFF_RUSH_EPA <- current_home_off_rush_epa$OFF_RUSH_EPA,
            HOME_OFF_PASS_EPA <- current_home_def_pass_epa$DEF_PASS_EPA,
            HOME_OFF_FDR <- current_home_off_fdr$n,
            HOME_DEF_RUSH_EPA <- current_home_def_rush_epa$DEF_RUSH_EPA,
            HOME_DEF_PASS_EPA <- current_home_def_pass_epa$DEF_PASS_EPA,
            HOME_DEF_FDR <- current_home_def_fdr$n,
            HOME_OFF_TO <- current_home_giveaways$n,
            HOME_DEF_TO <- current_home_takeaways$n,
            HOME_OFF_EXP_RATE <- current_home_off_exp_play_rate$n,
            HOME_DEF_EXP_RATE <- current_home_def_exp_play_rate$n,
            HOME_PENS <- current_home_total_penalties$n,
            HOME_PEN_YDS <- current_home_total_penalty_yds$Off_Pen_Yds,
            HOME_OL_METRIC <- current_home_off_line_metric$n,
            HOME_DL_METRIC <- current_home_def_line_metric$n,
            DIV_GAME <- div_game
        )
    } else if (current_week <= rolling_week_modifier) {

        # Load pbp
        current_pbp <- nflfastR::load_pbp(current_season)

        ################### AWAY STATS ###################
        # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
        away_general_data <- current_pbp %>%
            filter(week < current_week, posteam == current_away$away_team) %>%
            select(posteam_type, div_game) %>%
            unique()
        total_away_off_plays <- current_pbp %>%
            filter(week < current_week, posteam == current_away$away_team, play_type == "pass" | play_type == "run") %>%
            count()
        total_away_def_plays <- current_pbp %>%
            filter(week < current_week, defteam == current_away$away_team, play_type == "pass" | play_type == "run") %>%
            count()

        # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
        current_away_off_rush_epa <- current_pbp %>%
            filter(week < current_week, posteam == current_away$away_team, !is.na(epa), penalty == 0, play_type == "run") %>%
            summarise(OFF_RUSH_EPA = mean(epa))
        current_away_off_pass_epa <- current_pbp %>%
            filter(week < current_week, posteam == current_away$away_team, !is.na(epa), penalty == 0, play_type == "pass") %>%
            summarise(OFF_PASS_EPA = mean(epa))
        current_away_first_downs_for <- current_pbp %>%
            filter(week < current_week, posteam == current_away$away_team, first_down == 1, play_type == "pass" | play_type == "run") %>%
            count()
        current_away_off_fdr <- current_away_first_downs_for / total_away_off_plays

        # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
        current_away_def_rush_epa <- current_pbp %>%
            filter(week < current_week, defteam == current_away$away_team, !is.na(epa), penalty == 0, play_type == "run") %>%
            summarise(DEF_RUSH_EPA = mean(epa))
        current_away_def_pass_epa <- current_pbp %>%
            filter(week < current_week, defteam == current_away$away_team, !is.na(epa), penalty == 0, play_type == "pass") %>%
            summarise(DEF_PASS_EPA = mean(epa))
        current_away_first_downs_allowed <- current_pbp %>%
            filter(week < current_week, defteam == current_away$away_team, first_down == 1, play_type == "pass" | play_type == "run") %>%
            count()
        current_away_def_fdr <- current_away_first_downs_allowed / total_away_def_plays

        # ----- TURNOVER DATA ----- #
        current_away_giveaways <- (current_pbp %>% filter(week < current_week, posteam == current_away$away_team, fumble_lost == 1 | interception == 1) %>% count()) / (current_week - 1)
        current_away_takeaways <- (current_pbp %>% filter(week < current_week, defteam == current_away$away_team, fumble_lost == 1 | interception == 1) %>% count()) / (current_week - 1)

        # ----- EXPLOSIVE PLAY RATE ----- #
        current_away_off_exp_plays <- current_pbp %>%
            filter(week < current_week, posteam == current_away$away_team, yards_gained >= 15, interception == 0 & fumble_lost == 0, penalty == 0, play_type == "pass" | play_type == "run") %>%
            count()
        current_away_off_exp_play_rate <- current_away_off_exp_plays / total_away_off_plays
        current_away_def_exp_plays <- current_pbp %>%
            filter(week < current_week, defteam == current_away$away_team, yards_gained >= 15, interception == 0 & fumble_lost == 0, penalty == 0, play_type == "pass" | play_type == "run") %>%
            count()
        current_away_def_exp_play_rate <- current_away_def_exp_plays / total_away_def_plays

        # ----- PENALTIES & PENALTY YARDS ----- #
        current_away_off_penalties <- current_pbp %>%
            filter(week < current_week, posteam == current_away$away_team, penalty == 1 & penalty_team == current_away$away_team) %>%
            count()
        current_away_off_penalty_yds <- current_pbp %>%
            filter(week < current_week, posteam == current_away$away_team, penalty == 1 & penalty_team == current_away$away_team) %>%
            summarise(Off_Pen_Yds = sum(penalty_yards))
        current_away_def_penalties <- current_pbp %>%
            filter(week < current_week, defteam == current_away$away_team, penalty == 1 & penalty_team == current_away$away_team) %>%
            count()
        current_away_def_penalty_yds <- current_pbp %>%
            filter(week < current_week, defteam == current_away$away_team, penalty == 1 & penalty_team == current_away$away_team) %>%
            summarise(Def_Pen_Yds = sum(penalty_yards))
        current_away_total_penalties <- (current_away_off_penalties + current_away_def_penalties) / (current_week - 1)
        current_away_total_penalty_yds <- (current_away_off_penalty_yds + current_away_def_penalty_yds) / (current_week - 1)

        # ----- QB HIT + SACK DIFFERENTIAL ----- #
        # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
        # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
        current_away_off_line_metric <- (current_pbp %>% filter(week < current_week, posteam == current_away$away_team, qb_hit == 1 | sack == 1) %>% count()) / (current_week - 1)
        current_away_def_line_metric <- (current_pbp %>% filter(week < current_week, defteam == current_away$away_team, qb_hit == 1 | sack == 1) %>% count()) / (current_week - 1)

        # Make away data frame
        current_away_data <- data.frame(
            AWAY_TEAM <- current_away,
            AWAY_OFF_RUSH_EPA <- current_away_off_rush_epa$OFF_RUSH_EPA,
            AWAY_OFF_PASS_EPA <- current_away_off_pass_epa$OFF_PASS_EPA,
            AWAY_OFF_FDR <- current_away_off_fdr$n,
            AWAY_DEF_RUSH_EPA <- current_away_def_rush_epa$DEF_RUSH_EPA,
            AWAY_DEF_PASS_EPA <- current_away_def_pass_epa$DEF_PASS_EPA,
            AWAY_DEF_FDR <- current_away_def_fdr$n,
            AWAY_OFF_TO <- current_away_giveaways$n,
            AWAY_DEF_TO <- current_away_takeaways$n,
            AWAY_OFF_EXP_RATE <- current_away_off_exp_play_rate$n,
            AWAY_DEF_EXP_RATE <- current_away_def_exp_play_rate$n,
            AWAY_PENS <- current_away_total_penalties$n,
            AWAY_PEN_YDS <- current_away_total_penalty_yds$Off_Pen_Yds,
            AWAY_OL_METRIC <- current_away_off_line_metric$n,
            AWAY_DL_METRIC <- current_away_def_line_metric$n
        )

        ################### HOME STATS ###################
        # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
        home_general_data <- current_pbp %>%
            filter(week < current_week, posteam == current_home$home_team) %>%
            select(posteam_type, div_game) %>%
            unique()
        total_home_off_plays <- current_pbp %>%
            filter(week < current_week, posteam == current_home$home_team, play_type == "pass" | play_type == "run") %>%
            count()
        total_home_def_plays <- current_pbp %>%
            filter(week < current_week, defteam == current_home$home_team, play_type == "pass" | play_type == "run") %>%
            count()

        # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
        current_home_off_rush_epa <- current_pbp %>%
            filter(week < current_week, posteam == current_home$home_team, !is.na(epa), penalty == 0, play_type == "run") %>%
            summarise(OFF_RUSH_EPA = mean(epa))
        current_home_off_pass_epa <- current_pbp %>%
            filter(week < current_week, posteam == current_home$home_team, !is.na(epa), penalty == 0, play_type == "pass") %>%
            summarise(OFF_PASS_EPA = mean(epa))
        current_home_first_downs_for <- current_pbp %>%
            filter(week < current_week, posteam == current_home$home_team, first_down == 1, play_type == "pass" | play_type == "run") %>%
            count()
        current_home_off_fdr <- current_home_first_downs_for / total_home_off_plays

        # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
        current_home_def_rush_epa <- current_pbp %>%
            filter(week < current_week, defteam == current_home$home_team, !is.na(epa), penalty == 0, play_type == "run") %>%
            summarise(DEF_RUSH_EPA = mean(epa))
        current_home_def_pass_epa <- current_pbp %>%
            filter(week < current_week, defteam == current_home$home_team, !is.na(epa), penalty == 0, play_type == "pass") %>%
            summarise(DEF_PASS_EPA = mean(epa))
        current_home_first_downs_allowed <- current_pbp %>%
            filter(week < current_week, defteam == current_home$home_team, first_down == 1, play_type == "pass" | play_type == "run") %>%
            count()
        current_home_def_fdr <- current_home_first_downs_allowed / total_home_def_plays

        # ----- TURNOVER DATA ----- #
        current_home_giveaways <- (current_pbp %>% filter(week < current_week, posteam == current_home$home_team, fumble_lost == 1 | interception == 1) %>% count()) / (current_week - 1)
        current_home_takeaways <- (current_pbp %>% filter(week < current_week, defteam == current_home$home_team, fumble_lost == 1 | interception == 1) %>% count()) / (current_week - 1)

        # ----- EXPLOSIVE PLAY RATE ----- #
        current_home_off_exp_plays <- current_pbp %>%
            filter(week < current_week, posteam == current_home$home_team, yards_gained >= 15, interception == 0 & fumble_lost == 0, penalty == 0, play_type == "pass" | play_type == "run") %>%
            count()
        current_home_off_exp_play_rate <- current_home_off_exp_plays / total_home_off_plays
        current_home_def_exp_plays <- current_pbp %>%
            filter(week < current_week, defteam == current_home$home_team, yards_gained >= 15, interception == 0 & fumble_lost == 0, penalty == 0, play_type == "pass" | play_type == "run") %>%
            count()
        current_home_def_exp_play_rate <- current_home_def_exp_plays / total_home_def_plays

        # ----- PENALTIES & PENALTY YARDS ----- #
        current_home_off_penalties <- current_pbp %>%
            filter(week < current_week, posteam == current_home$home_team, penalty == 1 & penalty_team == current_home$home_team) %>%
            count()
        current_home_off_penalty_yds <- current_pbp %>%
            filter(week < current_week, posteam == current_home$home_team, penalty == 1 & penalty_team == current_home$home_team) %>%
            summarise(Off_Pen_Yds = sum(penalty_yards))
        current_home_def_penalties <- current_pbp %>%
            filter(week < current_week, defteam == current_home$home_team, penalty == 1 & penalty_team == current_home$home_team) %>%
            count()
        current_home_def_penalty_yds <- current_pbp %>%
            filter(week < current_week, defteam == current_home$home_team, penalty == 1 & penalty_team == current_home$home_team) %>%
            summarise(Def_Pen_Yds = sum(penalty_yards))
        current_home_total_penalties <- (current_home_off_penalties + current_home_def_penalties) / (current_week - 1)
        current_home_total_penalty_yds <- (current_home_off_penalty_yds + current_home_def_penalty_yds) / (current_week - 1)

        # ----- QB HIT + SACK DIFFERENTIAL ----- #
        # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
        # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
        current_home_off_line_metric <- (current_pbp %>% filter(week < current_week, posteam == current_home$home_team, qb_hit == 1 | sack == 1) %>% count()) / (current_week - 1)
        current_home_def_line_metric <- (current_pbp %>% filter(week < current_week, defteam == current_home$home_team, qb_hit == 1 | sack == 1) %>% count()) / (current_week - 1)

        # Make home data frame
        current_home_data <- data.frame(
            HOME_TEAM <- current_home,
            HOME_OFF_RUSH_EPA <- current_home_off_rush_epa$OFF_RUSH_EPA,
            HOME_OFF_PASS_EPA <- current_home_def_pass_epa$DEF_PASS_EPA,
            HOME_OFF_FDR <- current_home_off_fdr$n,
            HOME_DEF_RUSH_EPA <- current_home_def_rush_epa$DEF_RUSH_EPA,
            HOME_DEF_PASS_EPA <- current_home_def_pass_epa$DEF_PASS_EPA,
            HOME_DEF_FDR <- current_home_def_fdr$n,
            HOME_OFF_TO <- current_home_giveaways$n,
            HOME_DEF_TO <- current_home_takeaways$n,
            HOME_OFF_EXP_RATE <- current_home_off_exp_play_rate$n,
            HOME_DEF_EXP_RATE <- current_home_def_exp_play_rate$n,
            HOME_PENS <- current_home_total_penalties$n,
            HOME_PEN_YDS <- current_home_total_penalty_yds$Off_Pen_Yds,
            HOME_OL_METRIC <- current_home_off_line_metric$n,
            HOME_DL_METRIC <- current_home_def_line_metric$n,
            DIV_GAME <- div_game
        )
    } else {

        # Load pbp
        current_pbp <- nflfastR::load_pbp(current_season)
        rolling_week <- current_week - rolling_week_modifier

        ################### AWAY STATS ###################
        # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
        away_general_data <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_away$away_team) %>%
            select(posteam_type, div_game) %>%
            unique()
        total_away_off_plays <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_away$away_team, play_type == "pass" | play_type == "run") %>%
            count()
        total_away_def_plays <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_away$away_team, play_type == "pass" | play_type == "run") %>%
            count()

        # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
        current_away_off_rush_epa <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_away$away_team, !is.na(epa), penalty == 0, play_type == "run") %>%
            summarise(OFF_RUSH_EPA = mean(epa))
        current_away_off_pass_epa <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_away$away_team, !is.na(epa), penalty == 0, play_type == "pass") %>%
            summarise(OFF_PASS_EPA = mean(epa))
        current_away_first_downs_for <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_away$away_team, first_down == 1, play_type == "pass" | play_type == "run") %>%
            count()
        current_away_off_fdr <- current_away_first_downs_for / total_away_off_plays

        # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
        current_away_def_rush_epa <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_away$away_team, !is.na(epa), penalty == 0, play_type == "run") %>%
            summarise(DEF_RUSH_EPA = mean(epa))
        current_away_def_pass_epa <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_away$away_team, !is.na(epa), penalty == 0, play_type == "pass") %>%
            summarise(DEF_PASS_EPA = mean(epa))
        current_away_first_downs_allowed <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_away$away_team, first_down == 1, play_type == "pass" | play_type == "run") %>%
            count()
        current_away_def_fdr <- current_away_first_downs_allowed / total_away_def_plays

        # ----- TURNOVER DATA ----- #
        current_away_giveaways <- (current_pbp %>% filter(week >= rolling_week & week < current_week, posteam == current_away$away_team, fumble_lost == 1 | interception == 1) %>% count()) / rolling_week_modifier
        current_away_takeaways <- (current_pbp %>% filter(week >= rolling_week & week < current_week, defteam == current_away$away_team, fumble_lost == 1 | interception == 1) %>% count()) / rolling_week_modifier

        # ----- EXPLOSIVE PLAY RATE ----- #
        current_away_off_exp_plays <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_away$away_team, yards_gained >= 15, interception == 0 & fumble_lost == 0, penalty == 0, play_type == "pass" | play_type == "run") %>%
            count()
        current_away_off_exp_play_rate <- current_away_off_exp_plays / total_away_off_plays
        current_away_def_exp_plays <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_away$away_team, yards_gained >= 15, interception == 0 & fumble_lost == 0, penalty == 0, play_type == "pass" | play_type == "run") %>%
            count()
        current_away_def_exp_play_rate <- current_away_def_exp_plays / total_away_def_plays

        # ----- PENALTIES & PENALTY YARDS ----- #
        current_away_off_penalties <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_away$away_team, penalty == 1 & penalty_team == current_away$away_team) %>%
            count()
        current_away_off_penalty_yds <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_away$away_team, penalty == 1 & penalty_team == current_away$away_team) %>%
            summarise(Off_Pen_Yds = sum(penalty_yards))
        current_away_def_penalties <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_away$away_team, penalty == 1 & penalty_team == current_away$away_team) %>%
            count()
        current_away_def_penalty_yds <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_away$away_team, penalty == 1 & penalty_team == current_away$away_team) %>%
            summarise(Def_Pen_Yds = sum(penalty_yards))
        current_away_total_penalties <- (current_away_off_penalties + current_away_def_penalties) / rolling_week_modifier
        current_away_total_penalty_yds <- (current_away_off_penalty_yds + current_away_def_penalty_yds) / rolling_week_modifier

        # ----- QB HIT + SACK DIFFERENTIAL ----- #
        # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
        # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
        current_away_off_line_metric <- (current_pbp %>% filter(week >= rolling_week & week < current_week, posteam == current_away$away_team, qb_hit == 1 | sack == 1) %>% count()) / rolling_week_modifier
        current_away_def_line_metric <- (current_pbp %>% filter(week >= rolling_week & week < current_week, defteam == current_away$away_team, qb_hit == 1 | sack == 1) %>% count()) / rolling_week_modifier

        # Make away data frame
        current_away_data <- data.frame(
            AWAY_TEAM <- current_away,
            AWAY_OFF_RUSH_EPA <- current_away_off_rush_epa$OFF_RUSH_EPA,
            AWAY_OFF_PASS_EPA <- current_away_off_pass_epa$OFF_PASS_EPA,
            AWAY_OFF_FDR <- current_away_off_fdr$n,
            AWAY_DEF_RUSH_EPA <- current_away_def_rush_epa$DEF_RUSH_EPA,
            AWAY_DEF_PASS_EPA <- current_away_def_pass_epa$DEF_PASS_EPA,
            AWAY_DEF_FDR <- current_away_def_fdr$n,
            AWAY_OFF_TO <- current_away_giveaways$n,
            AWAY_DEF_TO <- current_away_takeaways$n,
            AWAY_OFF_EXP_RATE <- current_away_off_exp_play_rate$n,
            AWAY_DEF_EXP_RATE <- current_away_def_exp_play_rate$n,
            AWAY_PENS <- current_away_total_penalties$n,
            AWAY_PEN_YDS <- current_away_total_penalty_yds$Off_Pen_Yds,
            AWAY_OL_METRIC <- current_away_off_line_metric$n,
            AWAY_DL_METRIC <- current_away_def_line_metric$n
        )

        ################### HOME STATS ###################
        # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
        home_general_data <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_home$home_team) %>%
            select(posteam_type, div_game) %>%
            unique()
        total_home_off_plays <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_home$home_team, play_type == "pass" | play_type == "run") %>%
            count()
        total_home_def_plays <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_home$home_team, play_type == "pass" | play_type == "run") %>%
            count()

        # ----- OFFENSIVE EPA/GAME AND FIRST DOWN RATE----- #
        current_home_off_rush_epa <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_home$home_team, !is.na(epa), penalty == 0, play_type == "run") %>%
            summarise(OFF_RUSH_EPA = mean(epa))
        current_home_off_pass_epa <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_home$home_team, !is.na(epa), penalty == 0, play_type == "pass") %>%
            summarise(OFF_PASS_EPA = mean(epa))
        current_home_first_downs_for <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_home$home_team, first_down == 1, play_type == "pass" | play_type == "run") %>%
            count()
        current_home_off_fdr <- current_home_first_downs_for / total_home_off_plays

        # ----- DEFENSIVE EPA/GAME AND FIRST DOWN RATE ----- #
        current_home_def_rush_epa <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_home$home_team, !is.na(epa), penalty == 0, play_type == "run") %>%
            summarise(DEF_RUSH_EPA = mean(epa))
        current_home_def_pass_epa <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_home$home_team, !is.na(epa), penalty == 0, play_type == "pass") %>%
            summarise(DEF_PASS_EPA = mean(epa))
        current_home_first_downs_allowed <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_home$home_team, first_down == 1, play_type == "pass" | play_type == "run") %>%
            count()
        current_home_def_fdr <- current_home_first_downs_allowed / total_home_def_plays

        # ----- TURNOVER DATA ----- #
        current_home_giveaways <- (current_pbp %>% filter(week >= rolling_week & week < current_week, posteam == current_home$home_team, fumble_lost == 1 | interception == 1) %>% count()) / (current_week - 1)
        current_home_takeaways <- (current_pbp %>% filter(week >= rolling_week & week < current_week, defteam == current_home$home_team, fumble_lost == 1 | interception == 1) %>% count()) / (current_week - 1)

        # ----- EXPLOSIVE PLAY RATE ----- #
        current_home_off_exp_plays <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_home$home_team, yards_gained >= 15, interception == 0 & fumble_lost == 0, penalty == 0, play_type == "pass" | play_type == "run") %>%
            count()
        current_home_off_exp_play_rate <- current_home_off_exp_plays / total_home_off_plays
        current_home_def_exp_plays <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_home$home_team, yards_gained >= 15, interception == 0 & fumble_lost == 0, penalty == 0, play_type == "pass" | play_type == "run") %>%
            count()
        current_home_def_exp_play_rate <- current_home_def_exp_plays / total_home_def_plays

        # ----- PENALTIES & PENALTY YARDS ----- #
        current_home_off_penalties <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_home$home_team, penalty == 1 & penalty_team == current_home$home_team) %>%
            count()
        current_home_off_penalty_yds <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, posteam == current_home$home_team, penalty == 1 & penalty_team == current_home$home_team) %>%
            summarise(Off_Pen_Yds = sum(penalty_yards))
        current_home_def_penalties <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_home$home_team, penalty == 1 & penalty_team == current_home$home_team) %>%
            count()
        current_home_def_penalty_yds <- current_pbp %>%
            filter(week >= rolling_week & week < current_week, defteam == current_home$home_team, penalty == 1 & penalty_team == current_home$home_team) %>%
            summarise(Def_Pen_Yds = sum(penalty_yards))
        current_home_total_penalties <- (current_home_off_penalties + current_home_def_penalties) / (current_week - 1)
        current_home_total_penalty_yds <- (current_home_off_penalty_yds + current_home_def_penalty_yds) / (current_week - 1)

        # ----- QB HIT + SACK DIFFERENTIAL ----- #
        # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
        # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
        current_home_off_line_metric <- (current_pbp %>% filter(week >= rolling_week & week < current_week, posteam == current_home$home_team, qb_hit == 1 | sack == 1) %>% count()) / (current_week - 1)
        current_home_def_line_metric <- (current_pbp %>% filter(week >= rolling_week & week < current_week, defteam == current_home$home_team, qb_hit == 1 | sack == 1) %>% count()) / (current_week - 1)

        # Make home data frame
        current_home_data <- data.frame(
            HOME_TEAM <- current_home,
            HOME_OFF_RUSH_EPA <- current_home_off_rush_epa$OFF_RUSH_EPA,
            HOME_OFF_PASS_EPA <- current_home_def_pass_epa$DEF_PASS_EPA,
            HOME_OFF_FDR <- current_home_off_fdr$n,
            HOME_DEF_RUSH_EPA <- current_home_def_rush_epa$DEF_RUSH_EPA,
            HOME_DEF_PASS_EPA <- current_home_def_pass_epa$DEF_PASS_EPA,
            HOME_DEF_FDR <- current_home_def_fdr$n,
            HOME_OFF_TO <- current_home_giveaways$n,
            HOME_DEF_TO <- current_home_takeaways$n,
            HOME_OFF_EXP_RATE <- current_home_off_exp_play_rate$n,
            HOME_DEF_EXP_RATE <- current_home_def_exp_play_rate$n,
            HOME_PENS <- current_home_total_penalties$n,
            HOME_PEN_YDS <- current_home_total_penalty_yds$Off_Pen_Yds,
            HOME_OL_METRIC <- current_home_off_line_metric$n,
            HOME_DL_METRIC <- current_home_def_line_metric$n,
            DIV_GAME <- div_game
        )
    }

    # Append current_game_data to current_week_data
    # This includes all stats needed for predictions (29 data points, 14 per team, 1 for divisional game)
    current_game_data <- append(current_away_data, current_home_data)
    current_week_data <- rbind(current_week_data, current_game_data)
}

# Set column names to be pretty
colnames(current_week_data) <- c(
    "AWAY_TEAM",
    "AWAY_OFF_RUSH_EPA",
    "AWAY_OFF_PASS_EPA",
    "AWAY_OFF_FDR",
    "AWAY_DEF_RUSH_EPA",
    "AWAY_DEF_PASS_EPA",
    "AWAY_DEF_FDR",
    "AWAY_OFF_TO",
    "AWAY_DEF_TO",
    "AWAY_OFF_EXP_RATE",
    "AWAY_DEF_EXP_RATE",
    "AWAY_PENS",
    "AWAY_PEN_YDS",
    "AWAY_OL_METRIC",
    "AWAY_DL_METRIC",
    "HOME_TEAM",
    "HOME_OFF_RUSH_EPA",
    "HOME_OFF_PASS_EPA",
    "HOME_OFF_FDR",
    "HOME_DEF_RUSH_EPA",
    "HOME_DEF_PASS_EPA",
    "HOME_DEF_FDR",
    "HOME_OFF_TO",
    "HOME_DEF_TO",
    "HOME_OFF_EXP_RATE",
    "HOME_DEF_EXP_RATE",
    "HOME_PENS",
    "HOME_PEN_YDS",
    "HOME_OL_METRIC",
    "HOME_DL_METRIC",
    "DIV_GAME"
)

# Uncomment this line to debug the data going to the model
# write.csv(current_week_data, 'test.csv', row.names=FALSE)

# venv for dev purposes
reticulate::use_virtualenv("venv/")
reticulate::py_run_file("staley_says_v3.py")
# reticulate::source_python("staley_says_v3.py")
# Predictions will be made here and saved to the DB
# predict_games(test_set = reticulate::r_to_py(current_week_data), current_seaso = current_season, current_week = current_week)
reticulate::py$predict_games(reticulate::r_to_py(current_week_data), current_season, current_week)