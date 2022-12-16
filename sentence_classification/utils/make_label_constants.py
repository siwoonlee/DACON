from itertools import product
from pprint import pprint
import pandas as pd


def make_labels_dict():
    label_to_num_dict = dict()
    for i, (유형, 극성, 시제, 확실성) in enumerate(
        product(
            ['사실형', '추론형', '예측형', '대화형'],
            ['긍정', '부정', '미정'],
            ['현재', '과거', '미래'],
            ['확실', '불확실']
        )
    ):
        label = "-".join([유형, 극성, 시제, 확실성])
        label_to_num_dict[label] = i
    pprint(label_to_num_dict)


if __name__ == "__main__":
    make_labels_dict()