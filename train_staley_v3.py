import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def main():
    # Function to apply to dataframes
    def is_home_win(entry) -> bool:
        return 0 if entry <= 0 else 1

    # Load CSV file
    raw_data = pd.read_csv("data/raw_data.csv")
    # Get absolute value for PTS_DIFF to get rid of outliers
    raw_data["ABS_PTS_DIFF"] = abs(raw_data["PTS_DIFF"])
    # Get rid of rows that have an ABS_PTS_DIFF of larger than 2 touchdowns
    raw_data = raw_data[raw_data.ABS_PTS_DIFF <= 24]
    # Make new column to show home wins
    raw_data["RESULT"] = raw_data["PTS_DIFF"].apply(is_home_win)

    # Drop created columns
    raw_data = raw_data.drop(["PTS_DIFF", "ABS_PTS_DIFF", "id"], axis=1)
    # Create testing and training sets
    train_set, validation_set = train_test_split(raw_data)

    staley = train_model(train_set=train_set, validation_set=validation_set)

    # xgb.plot_importance(staley, importance_type="gain")
    # pyplot.show()

    test_set = pd.read_csv("data/2021_season_games.csv")
    test_set["RESULT"] = test_set["PTS_DIFF"].apply(is_home_win)
    test_set = test_set.drop(["id", "PTS_DIFF"], axis=1)
    predict_games(staley, test_set)

    save_choice = input("Save this model? ")
    if save_choice == "y" or save_choice == "yes":
        model_filename = input("filename to save: ")
        staley.save_model(f"models/{model_filename}.staley")


def train_model(train_set, validation_set):
    train_labels = train_set["RESULT"]
    validation_labels = validation_set["RESULT"]

    train_set = train_set.drop(["RESULT"], axis=1)
    validation_set = validation_set.drop(["RESULT"], axis=1)
    # test_set = test_set.drop(["RESULT"], axis=1)

    dtrain = xgb.DMatrix(train_set, label=train_labels)
    dvalid = xgb.DMatrix(validation_set, label=validation_labels)
    param = {
        "max_depth": 6,
        "eta": 0.3,
        "objective": "binary:logistic",
        # "eval_metric": "error",
        "subsample": 0.5,
        "gamma": 0.5,
        "num_parallel_tree": 500,
        "tree_method": "approx",
    }
    num_round = 65
    evallist = [(dvalid, "valid")]
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=num_round,
        evals=evallist,
        early_stopping_rounds=2,
    )

    return bst


# Test metrics
# print(raw_data[raw_data])
# print(f"PTS_DIFF max: {raw_data[raw_data.columns[-1]].max()}")
# print(f"PTS_DIFF mean: {raw_data[raw_data.columns[-1]].mean()}")
# print(f"PTS_DIFF median: {raw_data[raw_data.columns[-1]].median()}")
# print(f"PTS_DIFF std deviation: {raw_data[raw_data.columns[-1]].std()}")


def predict_games(bst, test_set):
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
