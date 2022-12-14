import os
import pickle
from data_processing import get_train_data, get_test_data, get_submission
from constants import 착과량_MAX
import autosklearn.regression


def main():
    X_train, y_train = get_train_data()
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=60*60 * 24,  # 60min * 24
        per_run_time_limit=60*60,  # 30min
        n_jobs=os.cpu_count() * 3 // 3,
        memory_limit=30000,  # 30GB
        resampling_strategy="cv",
        resampling_strategy_arguments={"folds": 5},
    )
    automl.fit(X_train, y_train, dataset_name='tangerine_dataset')
    print(automl.sprint_statistics())
    print("Performing re-fit")
    X_test = get_test_data()
    predictions = automl.predict(X_test) * 착과량_MAX
    get_submission(predictions, submission_id=3)
    x = automl.show_models()
    results = {"ensemble": x}
    pickle.dump(results, open('tangerine_model.pickle', 'wb'))


if __name__ == "__main__":
    main()
