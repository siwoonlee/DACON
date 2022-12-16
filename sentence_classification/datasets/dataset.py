import copy
import os
import random

import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from sentence_classification.utils.vectorizer import (
    get_padded_character_vector,
)
from sentence_classification.constants import (
    유형_to_num_map,
    극성_to_num_map,
    시제_to_num_map,
    확실성_to_num_map,
    full_name_label_to_num_map,
)

class SingleInputDataLoader(Dataset):
    def __init__(
        self,
        dataset_df,
        max_char_length=300,
        apply_data_augmentation=False,
    ):
        self.apply_data_augmentation = apply_data_augmentation
        self.max_char_length = max_char_length
        self.dataset_df = dataset_df

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        문장 = self.dataset_df['문장'].iloc[idx]
        if self.apply_data_augmentation:
            if random.choice([True, False]):
                문장 = add_noise_to_sentence(문장)
        return (
            get_padded_character_vector(문장, self.max_char_length),
            유형_to_num_map[self.dataset_df['유형'].iloc[idx]],
            극성_to_num_map[self.dataset_df['극성'].iloc[idx]],
            시제_to_num_map[self.dataset_df['시제'].iloc[idx]],
            확실성_to_num_map[self.dataset_df['확실성'].iloc[idx]],
            full_name_label_to_num_map[self.dataset_df['label'].iloc[idx]],
        )


def add_noise_to_sentence(title: str) -> str:
    noise = random.choice([' ', '*', '**', '!', '@', '#', '$', '%', '^', '&', '(', ')', ',', '.'])
    position = random.choice(range(len(title)+1))
    decomped_title = list(title)
    decomped_title.insert(position, noise)
    return "".join(decomped_title)


def collate_batch(batch):
    inputs = []
    유형_labels = []
    극성_labels = []
    시제_labels = []
    확실성_labels = []
    final_labels = []
    for one_sample_input, 유형_label, 극성_label, 시제_label, 확실성_label, final_label in batch:
        inputs.append(one_sample_input)
        유형_labels.append(유형_label)
        극성_labels.append(극성_label)
        시제_labels.append(시제_label)
        확실성_labels.append(확실성_label)
        final_labels.append(final_label)
    inputs = np.array(inputs, dtype=np.int64)
    유형_labels = np.array(유형_labels, dtype=np.int64)
    극성_labels = np.array(극성_labels, dtype=np.int64)
    시제_labels = np.array(시제_labels, dtype=np.int64)
    확실성_labels = np.array(확실성_labels, dtype=np.int64)
    final_labels = np.array(final_labels, dtype=np.int64)
    return dict(
        inputs=torch.from_numpy(inputs),
        유형_labels=torch.from_numpy(유형_labels),
        극성_labels=torch.from_numpy(극성_labels),
        시제_labels=torch.from_numpy(시제_labels),
        확실성_labels=torch.from_numpy(확실성_labels),
        final_labels=torch.from_numpy(final_labels),
    )


def get_cls_weights(target_y_df):
    label_to_count = target_y_df.value_counts()
    sample_weights = 1.0 / label_to_count[target_y_df]
    return sample_weights


def split_dataset_equal_category_balance(
    dataset: pd.DataFrame, test_ratio=0.1, min_num_labels_per_category=3
):
    train_dfs = []
    test_dfs = []
    unique_categories = dataset['label'].unique()
    for category in tqdm.tqdm(unique_categories):
        single_category_df = dataset[
            dataset['label'] == category
        ]
        if len(single_category_df) == 0:
            continue
        elif len(single_category_df) < min_num_labels_per_category:
            train_dfs.append(single_category_df)
            continue
        train_df, test_df = train_test_split(
            single_category_df, test_size=test_ratio, random_state=32
        )
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    assert len(dataset) >= len(train_df) + len(test_df)
    return train_df, test_df


def get_dataloader(
    stage,
    dataset_df,
    max_char_length=300,
    batch_size=128,
    sampling_method='normal',
    apply_data_augmentation=False,
    num_workers=0,
):
    custom_dl = SingleInputDataLoader(
        dataset_df,
        max_char_length=max_char_length,
        apply_data_augmentation=apply_data_augmentation,
    )
    if stage == 'train':
        if sampling_method == 'weighted':
            labels = pd.DataFrame(custom_dl.child_label)
            sample_weights = get_cls_weights(labels)
            loader = DataLoader(
                custom_dl,
                sampler=WeightedRandomSampler(
                    sample_weights.values,
                    len(sample_weights),
                    replacement=True,
                ),
                batch_size=batch_size,
                collate_fn=collate_batch,
                num_workers=num_workers,
            )
        elif sampling_method == 'normal':
            loader = DataLoader(
                custom_dl,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=collate_batch,
                drop_last=True,
            )
        else:
            raise NotImplementedError(
                f"sampling_method: {sampling_method} Not Implemented!"
            )
        return loader
    else:
        loader = DataLoader(
            custom_dl,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_batch,
            drop_last=True,
        )
        return loader
