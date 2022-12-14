import numpy as np
import pandas as pd
from constants import 착과량_MAX, 수고_MAX, 수관폭_MAX, 새순_MAX, 엽록소_MAX


def min_max_norm(
    df: pd.DataFrame,
    is_test: bool = False
) -> pd.DataFrame:
    if not is_test:
        df['착과량(int)'] = df['착과량(int)'] / 착과량_MAX
    df['수고(m)'] = df['수고(m)'] / 수고_MAX
    수관폭_cols = [col for col in df.columns if '수관폭' in col]
    df[수관폭_cols] = df[수관폭_cols] / 수관폭_MAX
    새순_cols = [col for col in df.columns if '새순' in col]
    df[새순_cols] = df[새순_cols] / 새순_MAX
    엽록소_cols = [col for col in df.columns if '엽록소' in col]
    df[엽록소_cols] = df[엽록소_cols] / 엽록소_MAX
    return df


def get_train_data():
    df = pd.read_csv("./data/train.csv")
    df = min_max_norm(df)
    y = df['착과량(int)'].values.astype(np.float32)
    df = df.drop(columns=['ID', '착과량(int)'])
    X = df.values.astype(np.float32)
    return X, y


def get_test_data():
    df = pd.read_csv("./data/test.csv")
    df = min_max_norm(df, is_test=True)
    df = df.drop(columns=['ID'])
    X = df.values.astype(np.float32)
    return X


def get_submission(predictions, submission_id: int = 1):
    df = pd.read_csv("./data/sample_submission.csv")
    predictions = np.round(predictions)
    df['착과량(int)'] = predictions.astype(np.int32)
    df.to_csv(f"./data/submission_num{submission_id}.csv", index=False)
