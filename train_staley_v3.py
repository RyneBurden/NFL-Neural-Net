import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder


def main():

    scaler = StandardScaler()

    # Load CSV file
    training_data = pd.read_csv("data/training_data_with_points.csv").dropna(how="any")
    training_data = training_data.drop(
        training_data[abs(training_data.HOME_PTS - training_data.AWAY_PTS) > 21].index
    )

    # Split data into home and away datasets and then
    training_data_away = training_data[
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
            "AWAY_PTS",
        ]
    ].to_numpy()

    training_data_home = training_data[
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
            "HOME_PTS",
        ]
    ].to_numpy()

    training_data_stacked = np.concatenate(
        (training_data_away, training_data_home), axis=0
    )

    # Get most points scored and set MAX_SCORE constant equal to
    pts = training_data_stacked[:, -1]
    MAX_SCORE = int(np.amax(pts))

    # Create testing and training sets
    train_set_split, validation_set_split = train_test_split(training_data_stacked)

    staley = train_model(
        train_set=train_set_split,
        validation_set=validation_set_split,
        max_score=MAX_SCORE,
    )

    test_data = pd.read_csv("data/training_data_2021_with_points.csv")

    # Split data into home and away datasets and then
    test_data_away = test_data[
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
            "AWAY_PTS",
        ]
    ].to_numpy(dtype=np.float64)

    test_data_home = test_data[
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
            "HOME_PTS",
        ]
    ].to_numpy(dtype=np.float64)

    test_data_stacked = np.concatenate((test_data_away, test_data_home), axis=0)[:, :-1]
    test_data_stacked_labels = np.concatenate((test_data_away, test_data_home), axis=0)[
        :, -1
    ]

    test_data_scaled = scaler.fit_transform(test_data_stacked)

    predictions = staley.predict(test_data_scaled)

    correct_predictions = 0
    index_modifier = test_data_away.shape[0]
    for x in range(int(len(predictions) / 2)):

        if (
            test_data_stacked_labels[x] > test_data_stacked_labels[x + index_modifier]
        ) and (np.argmax(predictions[x]) > np.argmax(predictions[x + index_modifier])):
            correct_predictions += 1
        if (
            test_data_stacked_labels[x] < test_data_stacked_labels[x + index_modifier]
        ) and (np.argmax(predictions[x]) < np.argmax(predictions[x + index_modifier])):
            correct_predictions += 1

    percent_correct_score = (
        correct_predictions / (len(test_data_stacked_labels) / 2) * 100
    )
    print(f"{round(percent_correct_score, 2)}%")

    to_save = input("Do you want to save this model? ")
    if to_save == "y" or to_save == "yes":
        model_name = input("Model name: ")
        output_filepath = f"models/{model_name}.staley"
        joblib.dump(staley, output_filepath)


def train_model(train_set: np.ndarray, validation_set: np.ndarray, max_score: int):

    scaler = StandardScaler()

    train_labels = train_set[:, -1]
    train_set = train_set[:, :-1]

    train_set_scaled = scaler.fit_transform(train_set)

    new_train_labels = np.zeros((train_labels.shape[0], max_score))
    for row_index in range(len(train_labels)):
        current_label = (
            int(train_labels[row_index] - 1)
            if (int(train_labels[row_index]) - 1 < max_score)
            else max_score - 1
        )
        new_train_labels[row_index][current_label] = 0.99

    validation_labels = validation_set[:, -1]
    validation_set = validation_set[:, :-1]

    xgb_estimator = xgb.XGBRegressor(
        objective="reg:squaredlogerror",
        # num_parallel_tree=5,
        subsample=0.5,
        gamma=0.5,
        eta=0.00001,
        eval_metric="logloss",
    )

    multilabel_model = MultiOutputRegressor(xgb_estimator)

    multilabel_model.fit(X=train_set_scaled, y=new_train_labels)

    return multilabel_model


def predict_games(bst: xgb.Booster, test_set: pd.DataFrame) -> None:
    # Shuffle set, isolate and drop labels
    test_set = test_set.sample(frac=1)
    test_labels = np.array(test_set["RESULT"].values.tolist())
    test_set = test_set.drop(["RESULT"], axis=1)
    dtest = xgb.DMatrix(test_set)
    # Perform predictions here
    predictions = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

    total_predictions = predictions.shape[0]
    correct_predictions = 0

    for x in range(0, total_predictions):
        current_prediction = 0 if predictions[x] < 0.5 else 1
        if current_prediction == test_labels[x]:
            correct_predictions += 1

    print(f"Correct prediction %: {(correct_predictions/total_predictions)*100}")


# Get CSV for 2021 season and test the model using last season's slate
if __name__ == "__main__":
    main()
