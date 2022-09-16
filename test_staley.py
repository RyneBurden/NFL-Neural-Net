import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


class PredictionCounter:
    correct = 0
    incorrect = 0
    ties = 0

    def increment_correct(self):
        self.correct += 1

    def increment_incorrect(self):
        self.incorrect += 1

    def increment_ties(self):
        self.ties += 1


def get_winner(row) -> str:
    away_score = row[1]
    home_score = row[3]

    if away_score > home_score:
        return "away"
    elif home_score > away_score:
        return "home"
    else:
        return "tie"


def predict_games(
    test_set: pd.DataFrame,
    current_season: int,
    current_week: int,
    validation_set: pd.DataFrame,
):

    num_games = test_set.shape[0]

    validation_set = validation_set[
        ["away_team", "away_score", "home_team", "home_score"]
    ].to_numpy()

    models = {}
    models["staley_1"] = joblib.load("models/2022_1_new.staley")
    models["staley_2"] = joblib.load("models/2022_2_new.staley")
    models["staley_3"] = joblib.load("models/2022_3_new.staley")
    models["staley_4"] = joblib.load("models/2022_4_new.staley")
    models["staley_5"] = joblib.load("models/2022_5_new.staley")

    num_models = len(models.keys())

    scaler = StandardScaler(copy=False)

    # Split data into home and away datasets and then
    test_data_away = test_set[
        [
            "AWAY_OFF_RUSH_EPA",
            "AWAY_OFF_PASS_EPA",
            "AWAY_OFF_FDR",
            "AWAY_OFF_TO",
            "AWAY_OFF_EXP_RATE",
            "AWAY_OL_METRIC",
            "AWAY_DEF_RUSH_EPA",
            "AWAY_DEF_PASS_EPA",
            "AWAY_DEF_FDR",
            "AWAY_DEF_TO",
            "AWAY_DEF_EXP_RATE",
            "AWAY_DL_METRIC",
            "DIV",
        ]
    ]

    test_data_home = test_set[
        [
            "HOME_OFF_RUSH_EPA",
            "HOME_OFF_PASS_EPA",
            "HOME_OFF_FDR",
            "HOME_OFF_TO",
            "HOME_OFF_EXP_RATE",
            "HOME_OL_METRIC",
            "HOME_DEF_RUSH_EPA",
            "HOME_DEF_PASS_EPA",
            "HOME_DEF_FDR",
            "HOME_DEF_TO",
            "HOME_DEF_EXP_RATE",
            "HOME_DL_METRIC",
            "DIV",
        ]
    ]

    test_data_stacked = np.concatenate(
        (
            test_data_away.to_numpy(dtype=np.float64),
            test_data_home.to_numpy(dtype=np.float64),
        ),
        axis=0,
    )
    test_data_scaled = scaler.fit_transform(test_data_stacked)

    index_modifier = test_data_away.shape[0]

    predictions = {}
    predictions[1] = models["staley_1"].predict(test_data_scaled)
    predictions[2] = models["staley_2"].predict(test_data_scaled)
    predictions[3] = models["staley_3"].predict(test_data_scaled)
    predictions[4] = models["staley_4"].predict(test_data_scaled)
    predictions[5] = models["staley_5"].predict(test_data_scaled)

    avg_predictions = np.zeros_like(predictions[1], dtype=np.float64)

    # The predictions for each game need to be averaged together and added to a numpy array
    # With the same shape as the stacked array
    for game_index in range(predictions[1].shape[0]):
        for column_index in range(predictions[1].shape[1]):
            for model_index in range(1, num_models + 1):
                current_model = predictions[model_index]
                avg_predictions[game_index][column_index] += (
                    current_model[game_index][column_index]
                ) * 100

    counter = PredictionCounter()

    for x in range(num_games):
        away_team_win_pred = np.argmax(avg_predictions[x]) > np.argmax(
            avg_predictions[x + index_modifier]
        )

        home_team_win_pred = np.argmax(avg_predictions[x]) < np.argmax(
            avg_predictions[x + index_modifier]
        )

        tie_pred = np.argmax(avg_predictions[x]) == np.argmax(
            avg_predictions[x + index_modifier]
        )

        away_index = np.argmax(avg_predictions[x])
        home_index = np.argmax(avg_predictions[x + index_modifier])
        current_away = test_set.iloc[x]["AWAY_TEAM"]
        current_home = test_set.iloc[x]["HOME_TEAM"]
        away_edge = avg_predictions[x][away_index] > avg_predictions[x][home_index]
        home_edge = avg_predictions[x][away_index] < avg_predictions[x][home_index]

        correct_winner = get_winner(validation_set[x, :])
        match correct_winner:
            case "home":
                if home_team_win_pred:
                    counter.increment_correct()
                elif away_team_win_pred:
                    counter.increment_incorrect()
                elif tie_pred:
                    calculated_edge = get_game_edge(
                        away_edge=away_edge,
                        home_edge=home_edge,
                        data=test_set[test_set["AWAY_TEAM"] == current_away],
                    )
                    if calculated_edge == "home":
                        counter.increment_correct()
                    else:
                        counter.increment_incorrect()
            case "away":
                if home_team_win_pred:
                    counter.increment_incorrect()
                elif away_team_win_pred:
                    counter.increment_correct()
                elif tie_pred:
                    calculated_edge = get_game_edge(
                        away_edge=away_edge,
                        home_edge=home_edge,
                        data=test_set[test_set["AWAY_TEAM"] == current_away],
                    )
                    if calculated_edge == "away":
                        counter.increment_correct()
                    else:
                        counter.increment_incorrect()
            case "tie":
                counter.increment_ties()

    print(
        f"\n\
        {current_season} Week {current_week}\
        {counter.correct} - {counter.incorrect} - {counter.ties} ({(counter.correct / (counter.correct + counter.incorrect)) * 100}%)\
        \n"
    )


def get_game_edge(
    away_edge: float,
    home_edge: float,
    data: pd.DataFrame,
) -> str:
    equal_edge = away_edge == home_edge
    data_copy = data.copy(deep=True)

    # Negative EPA is good for defense, so we need it to count positively towards the aggregate
    flip_sign = lambda x: x * -1
    data_copy[
        [
            "AWAY_DEF_RUSH_EPA",
            "AWAY_DEF_PASS_EPA",
            "HOME_DEF_RUSH_EPA",
            "HOME_DEF_PASS_EPA",
        ]
    ] = data_copy[
        [
            "AWAY_DEF_RUSH_EPA",
            "AWAY_DEF_PASS_EPA",
            "HOME_DEF_RUSH_EPA",
            "HOME_DEF_PASS_EPA",
        ]
    ].apply(
        flip_sign
    )

    # Calculate home and away aggregates
    away_aggregate = (
        data_copy[
            [
                "AWAY_OFF_FDR",
                "AWAY_OFF_RUSH_EPA",
                "AWAY_OFF_PASS_EPA",
                "AWAY_OFF_EXP_RATE",
                "AWAY_DEF_FDR",
                "AWAY_DEF_RUSH_EPA",
                "AWAY_DEF_PASS_EPA",
                "AWAY_DEF_EXP_RATE",
            ]
        ]
        .sum(axis=1)
        .to_numpy()
    )

    home_aggregate = (
        data_copy[
            [
                "HOME_OFF_FDR",
                "HOME_OFF_RUSH_EPA",
                "HOME_OFF_PASS_EPA",
                "HOME_OFF_EXP_RATE",
                "HOME_DEF_FDR",
                "HOME_DEF_RUSH_EPA",
                "HOME_DEF_PASS_EPA",
                "HOME_DEF_EXP_RATE",
            ]
        ]
        .sum(axis=1)
        .to_numpy()
        .item()
    )

    if equal_edge:
        return "away" if away_aggregate > home_aggregate else "home"

    return "away" if away_edge > home_edge else "home"
