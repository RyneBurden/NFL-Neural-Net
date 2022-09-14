import os

import gspread
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

load_dotenv()


def get_game_edge(
    away_team: str,
    home_team: str,
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
        .item()
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
        return away_team if away_aggregate > home_aggregate else home_team

    return away_team if away_edge > home_edge else home_team


def predict_games(
    input_test_set: pd.DataFrame,
    current_season: int,
    current_week: int,
):

    num_games = input_test_set.shape[0]

    models = {}
    models["staley_1"] = joblib.load("models/2022_1.staley")
    models["staley_2"] = joblib.load("models/2022_2.staley")
    models["staley_3"] = joblib.load("models/2022_3.staley")
    models["staley_4"] = joblib.load("models/2022_4.staley")
    models["staley_5"] = joblib.load("models/2022_5.staley")

    num_models = len(models.keys())

    scaler = StandardScaler(copy=False)

    # Split data into home and away datasets and then
    test_data_away = input_test_set[
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

    test_data_home = input_test_set[
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
                avg_predictions[game_index][column_index] += current_model[game_index][
                    column_index
                ]

    formatted_predictions = []
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
        current_away = input_test_set.iloc[x]["AWAY_TEAM"]
        current_home = input_test_set.iloc[x]["HOME_TEAM"]
        away_edge = avg_predictions[x][away_index]
        home_edge = avg_predictions[x][home_index]

        formatted_predictions.append(
            {
                "away_team": f"{current_away}",
                "away_score": away_index + 1,
                "home_team": f"{current_home}",
                "home_score": home_index + 1,
                "edge": ""
                if not tie_pred
                else get_game_edge(
                    away_team=current_away,
                    away_edge=away_edge,
                    home_team=current_home,
                    home_edge=home_edge,
                    data=input_test_set[input_test_set["AWAY_TEAM"] == current_away],
                ),
            }
        )

    prediction_viewer = pd.DataFrame.from_dict(formatted_predictions)
    to_save = input("Do you want to save to G Drive? ")
    if to_save == "yes" or to_save == "y":
        send_to_gdrive(current_week, prediction_viewer)
    print(prediction_viewer)


def send_to_gdrive(week_num: int, data: pd.DataFrame):
    gc = gspread.service_account()

    main_sheet = gc.open_by_url(os.getenv("SHEET_URL"))

    worksheet = main_sheet.worksheet(f"Week {week_num}")

    worksheet.update([data.columns.values.tolist()] + data.values.tolist())
