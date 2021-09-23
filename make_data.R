# offensive epa/game
# filter by 16 to get the epa per game
#staley_pbp %>% filter(season==season) %>% filter(posteam==team) %>% summarise(sum(epa)/num_games)
# I want to use a 4 game rolling average for epa/game

##### ---------- FILTER DATA FROM STALEY_PBP ---------- #####

source("load_packages.R")
print(glue("Packages loaded, data creation started\n"))

staley_data <- data.frame()

for (current_team_index in 1:nrow(teams)){
  current_team <- teams[current_team_index,]$team_abbr
  
  print(glue("\n\n----- ", current_team, " STARTED -----\n"))
  
  for (current_year in 1999:2020) {
    for (current_week in 1:17) {
      
      # ----- GENERAL DATA POINTS FROM NFLFASTR ----- #
      general_data <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team) %>% select(posteam_type, div_game) %>% unique()
      total_off_plays <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, play_type=="pass" | play_type=="run") %>% count()
      total_def_plays <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, play_type=="pass" | play_type=="run") %>% count()
      
      # ----- OFFENSIVE EPA/GAME  AND FIRST DOWN RATE----- #
      current_off_rush_epa <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(OFF_RUSH_EPA=mean(epa))
      current_off_pass_epa <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(OFF_PASS_EPA=mean(epa))
      current_first_downs_for <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
      current_off_fdr <- current_first_downs_for / total_off_plays
      
      # ----- DEFENSIVE EPA/GAME ----- #
      current_def_rush_epa <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, !is.na(epa), penalty==0, play_type=="run") %>% summarise(DEF_RUSH_EPA=mean(epa))
      current_def_pass_epa <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, !is.na(epa), penalty==0, play_type=="pass") %>% summarise(DEF_PASS_EPA=mean(epa))
      current_first_downs_allowed <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, first_down==1, play_type=="pass" | play_type=="run") %>% count()
      current_def_fdr <- current_first_downs_allowed / total_def_plays
      
      # ----- TURNOVER DIFFERENTIAL ----- #
      ## get giveaways
      current_giveaways <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, fumble_lost==1 | interception==1) %>% count()
      
      ## get takeaways
      current_takeaways <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, fumble_lost==1 | interception==1) %>% count()
      
      ## subtract the two values for the differential
      ### positive value means you have more takeaways than giveaways, obviously a good thing
      current_turnover_diff <- current_takeaways - current_giveaways
      
      # ----- EXPLOSIVE PLAY RATE ----- #
      ## get offensive explosive plays
      current_off_exp_plays <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
      current_off_exp_play_rate <- current_off_exp_plays / total_off_plays
      
      ## get defensive explosive plays allowed
      current_def_exp_plays <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, yards_gained>=15, interception==0 & fumble_lost==0, penalty==0, play_type=="pass" | play_type=="run") %>% count()
      current_def_exp_play_rate <- current_def_exp_plays / total_def_plays
      
      ## Subtract the two values for the differential
      ### a positive value means your offense has more exp plays than your defense allows
      current_exp_play_diff <- current_off_exp_plays - current_def_exp_plays
      
      # ----- POINT DIFFERENTIAL ----- #
      ## offensive points
      current_td_for <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, touchdown==1, interception==0 & fumble_lost==0) %>% count()
      current_fg_for <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, field_goal_result=="made") %>% count()
      current_xp_for <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, extra_point_result=="good") %>% count()
      current_safety_for <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, safety==1) %>% count()
      current_points_for <- (current_td_for * 6) + (current_fg_for * 3) + (current_safety_for * 2) + current_xp_for
      
      ## defensive points allowed
      current_td_allowed <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, touchdown==1, interception==0 & fumble_lost==0) %>% count()
      current_fg_allowed <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, field_goal_result=="made") %>% count()
      current_xp_allowed <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, extra_point_result=="good") %>% count()
      current_safety_allowed <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, safety==1) %>% count()
      current_points_allowed <- (current_td_allowed * 6) + (current_fg_allowed * 3) + (current_safety_allowed * 2) + current_xp_allowed
      
      ## calculate the differential
      current_point_diff <- current_points_for - current_points_allowed
      
      # ----- PENALTIES & PENALTY YARDS ----- #
      current_off_penalties <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, penalty==1 & penalty_team==current_team) %>% count()
      current_off_penalty_yds <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, penalty==1 & penalty_team==current_team) %>% summarise(Off_Pen_Yds=sum(penalty_yards))
      current_def_penalties <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, penalty==1 & penalty_team==current_team) %>% count()
      current_def_penalty_yds <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, penalty==1 & penalty_team==current_team) %>% summarise(Def_Pen_Yds=sum(penalty_yards))
      current_total_penalties <- current_off_penalties + current_def_penalties
      current_total_penalty_yds <- current_off_penalty_yds + current_def_penalty_yds
      
      # ----- QB HIT + SACK DIFFERENTIAL ----- #
      # A positive value means the defense produces more QB Hits and Sacks than are allowed on offense
      # This seems like a decent measure of offensive/defensive line play that can be inferred from the nflfastR data sets
      current_off_line_metric <- staley_pbp %>% filter(season==current_year, week==current_week, posteam==current_team, qb_hit==1 | sack==1) %>% count()
      current_def_line_metric <- staley_pbp %>% filter(season==current_year, week==current_week, defteam==current_team, qb_hit==1 | sack==1) %>% count()
      
      #line_metric_diff <- current_def_line_metric - current_off_line_metric
      
      # Add all data points to a separate data frame 
      current_week_data <- data.frame(
        TEAM=current_team,
        SEASON=current_year,
        WEEK=current_week,
        OFF_RUSH_EPA=current_off_rush_epa$OFF_RUSH_EPA,
        OFF_PASS_EPA=current_off_pass_epa$OFF_PASS_EPA,
        OFF_FDR=current_off_fdr$n,
        DEF_RUSH_EPA=current_def_rush_epa$DEF_RUSH_EPA,
        DEF_PASS_EPA=current_def_pass_epa$DEF_PASS_EPA,
        DEF_FDR=current_def_fdr$n,
        OFF_TO=current_giveaways$n,
        DEF_TO=current_takeaways$n,
        #TO_DIFF=current_turnover_diff$n,
        #TO_DIFF_POS=if_else(current_turnover_diff$n >= 0, current_turnover_diff$n, as.integer(0)),
        #TO_DIFF_NEG=if_else(current_turnover_diff$n < 0, (current_turnover_diff$n * -1), as.integer(0)),
        OFF_EXP_RATE=current_off_exp_play_rate$n,
        DEF_EXP_RATE=current_def_exp_play_rate$n,
        PENS=current_total_penalties$n,
        PEN_YDS=current_total_penalty_yds$Off_Pen_Yds,
        OL_METRIC=current_off_line_metric$n,
        DL_METRIC=current_def_line_metric$n,
        HOA=ifelse(nrow(general_data)==0, "BYE", if_else(general_data$posteam_type=="home", 1, 0)),
        DIV=ifelse(nrow(general_data)==0, as.integer(0), general_data$div_game),
        PTS_DIFF=current_point_diff$n
      )
      # Add the data frame to the main data frame with all staley data
      staley_data <- staley_data %>% rbind(current_week_data)
    }
    
    print(glue(current_year, " finished"))
    
  }
}

# Cleanup variables
remove(current_team_index)
remove(current_team)
remove(current_year)
remove(current_week)
remove(general_data)
remove(total_def_plays)
remove(total_off_plays)
remove(current_off_rush_epa)
remove(current_off_pass_epa)
remove(current_off_fdr)
remove(current_first_downs_for)
remove(current_def_rush_epa)
remove(current_def_pass_epa)
remove(current_def_fdr)
remove(current_first_downs_allowed)
remove(current_giveaways)
remove(current_takeaways)
remove(current_turnover_diff)
remove(current_off_exp_plays)
remove(current_def_exp_plays)
remove(current_def_exp_play_rate)
remove(current_off_exp_play_rate)
remove(current_exp_play_diff)
remove(current_td_for)
remove(current_fg_for)
remove(current_xp_for)
remove(current_safety_for)
remove(current_safety_allowed)
remove(current_points_for)
remove(current_td_allowed)
remove(current_fg_allowed)
remove(current_xp_allowed)
remove(current_points_allowed)
remove(current_point_diff)
remove(current_off_penalties)
remove(current_off_penalty_yds)
remove(current_def_penalties)
remove(current_def_penalty_yds)
remove(current_total_penalties)
remove(current_total_penalty_yds)
remove(current_off_line_metric)
remove(current_def_line_metric)
remove(current_week_data)