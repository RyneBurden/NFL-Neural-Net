import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def predict_games(test_set: pd.DataFrame, current_season: int, current_week: int):

    models = {}
    models["staley_1"] = joblib.load("models/2022_1.staley")

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

    predictions = models["staley_1"].predict(test_data_scaled)

    for x in range(test_set.shape[0]):
        if np.argmax(predictions[x]) != np.argmax(predictions[x + index_modifier]):
            if len(test_set.iloc[x]["AWAY_TEAM"]) == 2:
                if np.argmax(predictions[x]) + 1 < 10:
                    print(
                        f"{test_set.iloc[x]['AWAY_TEAM']}  - {np.argmax(predictions[x]) + 1}  || {test_set.iloc[x]['HOME_TEAM']} - {np.argmax(predictions[x+index_modifier]) + 1}"
                    )
                else:
                    print(
                        f"{test_set.iloc[x]['AWAY_TEAM']}  - {np.argmax(predictions[x]) + 1} || {test_set.iloc[x]['HOME_TEAM']} - {np.argmax(predictions[x+index_modifier]) + 1}"
                    )
            else:
                if np.argmax(predictions[x]) + 1 < 10:
                    print(
                        f"{test_set.iloc[x]['AWAY_TEAM']} - {np.argmax(predictions[x]) + 1}  || {test_set.iloc[x]['HOME_TEAM']} - {np.argmax(predictions[x+index_modifier]) + 1}"
                    )
                else:
                    print(
                        f"{test_set.iloc[x]['AWAY_TEAM']} - {np.argmax(predictions[x]) + 1} || {test_set.iloc[x]['HOME_TEAM']} - {np.argmax(predictions[x+index_modifier]) + 1}"
                    )
        else:
            max_away_prediction_index = np.argmax(predictions[x])
            max_home_prediction_index = np.argmax(predictions[x + index_modifier])
            if (
                predictions[x][max_away_prediction_index]
                > predictions[x + index_modifier][max_home_prediction_index]
            ):
                print(
                    f"{test_set.iloc[x]['AWAY_TEAM']} - {np.argmax(predictions[x]) + 1} || {test_set.iloc[x]['HOME_TEAM']} - {np.argmax(predictions[x+index_modifier]) + 1} \t {test_set.iloc[x]['AWAY_TEAM']} has the edge"
                )
            else:
                print(
                    f"{test_set.iloc[x]['AWAY_TEAM']} - {np.argmax(predictions[x]) + 1} || {test_set.iloc[x]['HOME_TEAM']} - {np.argmax(predictions[x+index_modifier]) + 1} \t {test_set.iloc[x]['HOME_TEAM']} has the edge"
                )
