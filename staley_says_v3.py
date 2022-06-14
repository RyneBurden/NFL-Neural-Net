import pandas as pd
import numpy as np
import xgboost as xgb

def main():

    # Function to apply to dataframes
    def is_home_win(entry) -> bool:
        return 0 if entry <= 0 else 1

    models = {}

    # Load models
    models["staley_1"] = xgb.Booster(model_file="models/001.staley")
    models["staley_2"] = xgb.Booster(model_file="models/002.staley")
    models["staley_3"] = xgb.Booster(model_file="models/003.staley")
    models["staley_4"] = xgb.Booster(model_file="models/004.staley")
    models["staley_5"] = xgb.Booster(model_file="models/005.staley")

    test_set = pd.read_csv("data/2021_season_games.csv")
    test_set["RESULT"] = test_set["PTS_DIFF"].apply(is_home_win)

    # Shuffle set, isolate and drop labels
    test_set = test_set.sample(frac=1)
    test_labels = np.array(test_set["RESULT"].values.tolist())
    test_set = test_set.drop(["RESULT", "id", "PTS_DIFF"], axis=1)
    dtest = xgb.DMatrix(test_set)

    predictions_1 = models["staley_1"].predict(dtest, iteration_range=(0, models["staley_1"].best_iteration + 1))
    predictions_2 = models["staley_2"].predict(dtest, iteration_range=(0, models["staley_2"].best_iteration + 1))
    predictions_3 = models["staley_3"].predict(dtest, iteration_range=(0, models["staley_3"].best_iteration + 1))
    predictions_4 = models["staley_4"].predict(dtest, iteration_range=(0, models["staley_4"].best_iteration + 1))
    predictions_5 = models["staley_5"].predict(dtest, iteration_range=(0, models["staley_5"].best_iteration + 1))

if __name__ == "__main__":
    main()