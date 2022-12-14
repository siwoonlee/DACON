import pandas as pd


def get_min_max(df):
    착과량_min = df['착과량(int)'].min()
    착과량_max = df['착과량(int)'].max()
    수고_min = df['수고(m)'].min()
    수고_max = df['수고(m)'].max()
    수관폭_min = df[[col for col in df.columns if '수관폭' in col]].min().min()
    수관폭_max = df[[col for col in df.columns if '수관폭' in col]].max().max()
    새순_min = df[[col for col in df.columns if '새순' in col]].min().min()
    새순_max = df[[col for col in df.columns if '새순' in col]].max().max()
    엽록소_min = df[[col for col in df.columns if '엽록소' in col]].min().min()
    엽록소_max = df[[col for col in df.columns if '엽록소' in col]].max().max()
    return dict(
        착과량_min=착과량_min,
        착과량_max=착과량_max,
        수고_min=수고_min,
        수고_max=수고_max,
        수관폭_min=수관폭_min,
        수관폭_max=수관폭_max,
        새순_min=새순_min,
        새순_max=새순_max,
        엽록소_min=엽록소_min,
        엽록소_max=엽록소_max
    )


if __name__ == "__main__":
    df = pd.read_csv("./data/train.csv")
    print(df.describe())
    print(get_min_max(df))
