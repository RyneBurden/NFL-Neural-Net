import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def predict_games(test_set: pd.DataFrame, current_season: int, current_week: int):

    num_games = test_set.shape[0]

    models = {}
    models["staley_1"] = joblib.load("models/2022_1.staley")
    models["staley_2"] = joblib.load("models/2022_2.staley")
    models["staley_3"] = joblib.load("models/2022_3.staley")
    models["staley_4"] = joblib.load("models/2022_4.staley")
    models["staley_5"] = joblib.load("models/2022_5.staley")

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
        ]
    ].to_numpy(dtype=np.float64)

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
        ]
    ].to_numpy(dtype=np.float64)

    test_data_stacked = np.concatenate((test_data_away, test_data_home), axis=0)
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
            for model_index in range(1, 6):
                current_model = predictions[model_index]
                avg_predictions[game_index][column_index] += current_model[game_index][
                    column_index
                ]

    for x in range(int(test_set.shape[0])):
        if np.argmax(avg_predictions[x]) != np.argmax(
            avg_predictions[x + index_modifier]
        ):
            if len(test_set.iloc[x]["AWAY_TEAM"]) == 2:
                if np.argmax(avg_predictions[x]) + 1 < 10:
                    print(
                        f"{test_set.iloc[x]['AWAY_TEAM']}  - {np.argmax(avg_predictions[x]) + 1}  || {test_set.iloc[x]['HOME_TEAM']} - {np.argmax(avg_predictions[x+index_modifier]) + 1}"
                    )
                else:
                    print(
                        f"{test_set.iloc[x]['AWAY_TEAM']}  - {np.argmax(avg_predictions[x]) + 1} || {test_set.iloc[x]['HOME_TEAM']} - {np.argmax(avg_predictions[x+index_modifier]) + 1}"
                    )
            else:
                if np.argmax(avg_predictions[x]) + 1 < 10:
                    print(
                        f"{test_set.iloc[x]['AWAY_TEAM']} - {np.argmax(avg_predictions[x]) + 1}  || {test_set.iloc[x]['HOME_TEAM']} - {np.argmax(avg_predictions[x+index_modifier]) + 1}"
                    )
                else:
                    print(
                        f"{test_set.iloc[x]['AWAY_TEAM']} - {np.argmax(avg_predictions[x]) + 1} || {test_set.iloc[x]['HOME_TEAM']} - {np.argmax(avg_predictions[x+index_modifier]) + 1}"
                    )
        else:
            max_away_prediction_index = np.argmax(avg_predictions[x])
            max_home_prediction_index = np.argmax(avg_predictions[x + index_modifier])
            if (
                avg_predictions[x][max_away_prediction_index]
                > avg_predictions[x + index_modifier][max_home_prediction_index]
            ):
                print(
                    f"{test_set.iloc[x]['AWAY_TEAM']} - {np.argmax(avg_predictions[x]) + 1} || {test_set.iloc[x]['HOME_TEAM']} - {np.argmax(avg_predictions[x+index_modifier]) + 1} \t {test_set.iloc[x]['AWAY_TEAM']} has the edge"
                )
            else:
                print(
                    f"{test_set.iloc[x]['AWAY_TEAM']} - {np.argmax(avg_predictions[x]) + 1} || {test_set.iloc[x]['HOME_TEAM']} - {np.argmax(avg_predictions[x+index_modifier]) + 1} \t {test_set.iloc[x]['HOME_TEAM']} has the edge"
                )
