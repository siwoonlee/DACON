from collections import Counter
from pprint import pprint
import numpy as np
import pandas as pd

from utils.vectorizer import get_character_vector


def main():
    df = pd.read_csv("./data/train.csv")
    print(df.describe().to_string())
    inspect_sentence_char_length(df)
    inspect_labels(df)


def inspect_sentence_char_length(df: pd.DataFrame):
    char_lengths = []
    for sentence in df['문장'].values:
        char_lengths.append(len(get_character_vector(sentence)))
    print(f"Length mean: {np.mean(char_lengths)}, "
          f"max: {np.max(char_lengths)}, "
          f"min: {np.min(char_lengths)}, "
          f"95%: {np.percentile(char_lengths, 95)}")


def inspect_labels(df: pd.DataFrame):
    pprint(Counter(df['label'].values))
    for col in ('유형', '극성', '시제', '확실성'):
        print(Counter(df[col]))


if __name__ == "__main__":
    main()