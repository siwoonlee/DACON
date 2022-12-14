import copy
import os
import random

import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from ..utils.vectorizer import (
    get_padded_character_vector,
)


class SingleInputDataLoader(Dataset):
    def __init__(
        self,
        stage,
        dataset_dir,
        is_production_mode,
        transaction_type,
        dt,
        max_char_length=75,
        mmap_mode='r',
        apply_data_augmentation=False,
    ):
        self.apply_data_augmentation = apply_data_augmentation
        self.default_category_df = (
            load_feature_engineered_default_category_df()
        )
        self.max_char_length = max_char_length
        self.title_df = pd.read_pickle(
            os.path.join(
                dataset_dir,
                f"{dt}_{transaction_type}_{is_production_mode}_title_{stage}.pkl",
            )
        )
        self.is_gold_standard = pd.read_pickle(
            os.path.join(
                dataset_dir,
                f"{dt}_{transaction_type}_{is_production_mode}_is_gold_standard_{stage}.pkl",
            )
        )
        self.child_label = np.load(
            os.path.join(
                dataset_dir,
                f"{dt}_{transaction_type}_{is_production_mode}_child_label_{stage}.npy",
            ),
            mmap_mode=mmap_mode,
        )
        self.parent_label = np.load(
            os.path.join(
                dataset_dir,
                f"{dt}_{transaction_type}_{is_production_mode}_parent_label_{stage}.npy",
            ),
            mmap_mode=mmap_mode,
        )
        assert len(self.title_df) == len(self.child_label)

    def __len__(self):
        return len(self.title_df)

    def __getitem__(self, idx):
        return (
            get_padded_character_vector(
                self.title_df.iloc[idx], self.max_char_length
            ),
            self.child_label[idx],
            self.parent_label[idx],
        )


class MultiInputDataLoader(SingleInputDataLoader):
    def __init__(
        self,
        stage,
        dataset_dir,
        is_production_mode,
        transaction_type,
        dt,
        max_char_length=75,
        mmap_mode='r',
        apply_data_augmentation=False,
    ):
        super().__init__(
            stage,
            dataset_dir,
            is_production_mode,
            transaction_type,
            dt,
            max_char_length=max_char_length,
            mmap_mode=mmap_mode,
            apply_data_augmentation=apply_data_augmentation
        )
        self.transaction_amount_vec = np.load(
            os.path.join(
                dataset_dir,
                f"{dt}_{transaction_type}_{is_production_mode}_TRANSACTION_AMOUNT_RANGE_LIST_{stage}.npy",
            ),
            mmap_mode=mmap_mode,
        )

    def __len__(self):
        return len(self.title_df)

    def __getitem__(self, idx):
        title = self.title_df.iloc[idx]
        if self.apply_data_augmentation:
            if random.choice([True, False]):
                title = add_noise_to_transaction_title(title)
        return (
            get_padded_character_vector(
                title, self.max_char_length
            ),
            self.transaction_amount_vec[idx],
            self.child_label[idx],
            self.parent_label[idx],
        )


def add_noise_to_transaction_title(title: str) -> str:
    noise = random.choice([' ', '*', '**', '!', '@', '#', '$', '%', '^', '&', '(', ')', ',', '.'])
    position = random.choice(range(len(title)+1))
    decomped_title = list(title)
    decomped_title.insert(position, noise)
    return "".join(decomped_title)


def collate_batch(batch):
    inputs = []
    child_labels = []
    parent_labels = []
    for one_sample_input, child_label, parent_label in batch:
        inputs.append(one_sample_input)
        child_labels.append(child_label)
        parent_labels.append(parent_label)
    inputs = np.array(inputs, dtype=np.int64)
    child_labels = np.array(child_labels, dtype=np.int64)
    parent_labels = np.array(parent_labels, dtype=np.int64)
    return dict(
        inputs=torch.from_numpy(inputs),
        child_labels=torch.from_numpy(child_labels),
        parent_labels=torch.from_numpy(parent_labels),
    )


def collate_batch_multi(batch):
    title_vec_inputs = []
    transaction_amount_inputs = []
    child_labels = []
    parent_labels = []
    for (
        one_sample_title_vec,
        one_sample_transaction_amount_one_hot,
        child_label,
        parent_label,
    ) in batch:
        title_vec_inputs.append(one_sample_title_vec)
        transaction_amount_inputs.append(one_sample_transaction_amount_one_hot)
        child_labels.append(child_label)
        parent_labels.append(parent_label)
    title_vec_inputs = np.array(title_vec_inputs, dtype=np.int64)
    transaction_amount_inputs = np.array(
        transaction_amount_inputs, dtype=np.float32
    )
    child_labels = np.array(child_labels, dtype=np.int64)
    parent_labels = np.array(parent_labels, dtype=np.int64)
    return dict(
        title_vec_inputs=torch.from_numpy(title_vec_inputs),
        transaction_amount_inputs=torch.from_numpy(transaction_amount_inputs),
        child_labels=torch.from_numpy(child_labels),
        parent_labels=torch.from_numpy(parent_labels),
    )


def get_cls_weights(target_y_df):
    label_to_count = target_y_df.value_counts()
    sample_weights = 1.0 / label_to_count[target_y_df]
    return sample_weights


def make_onehot_transaction_amount_range_df(df: pd.DataFrame) -> pd.DataFrame:
    transaction_amount_range_list = df['transaction_amount_range'].values
    onehot_df = pd.DataFrame(
        data=np.zeros(
            shape=(
                len(transaction_amount_range_list),
                len(TRANSACTION_AMOUNT_RANGE_LIST),
            ),
            dtype=np.int32,
        ),
        columns=TRANSACTION_AMOUNT_RANGE_LIST,
    )
    for i, transaction_amount_range in tqdm.tqdm(
        enumerate(transaction_amount_range_list),
        total=len(transaction_amount_range_list),
    ):
        target_column = f'transaction_amount_range_{transaction_amount_range}'
        onehot_df.loc[i, target_column] = 1
    df = df.drop(columns=['transaction_amount_range'])
    df = pd.concat([df, onehot_df], axis=1)
    return df



def split_dataset_equal_category_balance(
    dataset: pd.DataFrame, test_ratio=0.1, min_num_titles_per_category=2
):
    train_dfs = []
    test_dfs = []
    unique_categories = dataset['default_category_id'].unique()
    for category in tqdm.tqdm(unique_categories):
        single_category_df = dataset[
            dataset['default_category_id'] == category
        ]
        if len(single_category_df) < min_num_titles_per_category:
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


def get_single_transaction_type_train_val_test_data(
    transaction_type='card',
    dataset_dir='/home/dataset',
    is_production_mode=False,
    process_onehot=True,
    dt='2022-10-24',
):
    """
        transaction_type: card, bankaccount, payment, prepayment
    """
    gold_standard_df = load_gold_standard_dataset(
        transaction_type, dataset_dir, dt=dt
    )
    legacy_data_df = pd.read_pickle(
        os.path.join(
            dataset_dir,
            f"legacy_labeled_dataset.pkl",
        )
    )
    gold_standard_df = pd.concat([gold_standard_df, legacy_data_df], axis=0).reset_index(drop=True)
    gold_standard_df['is_gold_standard'] = True

    logger.info(f"loaded {transaction_type} gold standard dataset!")
    if transaction_type in ('card', 'bankaccount'):
        old_dataset_df = load_old_dataset(transaction_type, dataset_dir)
        logger.info("loaded old dataset!")
        df = pd.concat([old_dataset_df, gold_standard_df], axis=0).reset_index(
            drop=True
        )
    else:
        df = gold_standard_df

    df = make_full_category_name(df)
    df = make_child_parent_label(df)
    train_val_dataset_df, test_dataset_df = split_dataset_equal_category_balance(
        df, test_ratio=0.1, min_num_titles_per_category=2
    )
    train_dataset_df, validation_dataset_df = split_dataset_equal_category_balance(
        train_val_dataset_df, test_ratio=2 / 9, min_num_titles_per_category=2
    )
    logger.info(f"data split success!")
    backoffice_ruleset_df = load_backoffice_ruleset_df(
        transaction_type=f"electronic_{transaction_type}"
        if 'payment' in transaction_type
        else transaction_type,
        dataset_dir=dataset_dir,
        dt=dt,
    )
    default_category_df = load_default_category_df(
        dataset_dir=dataset_dir, dt=dt
    )
    backoffice_ruleset_df = feature_engineer_backoffice_ruleset_df(
        backoffice_ruleset_df, default_category_df=default_category_df
    )
    train_dataset_df = pd.concat(
        [train_dataset_df, backoffice_ruleset_df], axis=0
    ).reset_index(drop=True)
    if is_production_mode:
        train_dataset_df = pd.concat(
            [train_dataset_df, validation_dataset_df, test_dataset_df]
        ).reset_index(drop=True)
    if process_onehot:
        train_dataset_df = make_onehot_transaction_amount_range_df(
            train_dataset_df
        )
        validation_dataset_df = make_onehot_transaction_amount_range_df(
            validation_dataset_df
        )
        test_dataset_df = make_onehot_transaction_amount_range_df(
            test_dataset_df
        )
    return train_dataset_df, validation_dataset_df, test_dataset_df


def get_dataloader(
    stage,
    dataset_dir,
    is_production_mode,
    transaction_type,
    mmap_mode,
    dt,
    max_char_length=75,
    batch_size=128,
    sampling_method='normal',
    is_multi_input=False,
    apply_data_augmentation=False,
    num_workers=0,
):
    custom_dl = (
        MultiInputDataLoader(
            stage,
            dataset_dir,
            is_production_mode,
            transaction_type,
            dt,
            max_char_length=max_char_length,
            mmap_mode=mmap_mode,
            apply_data_augmentation=apply_data_augmentation,
        )
        if is_multi_input
        else SingleInputDataLoader(
            stage,
            dataset_dir,
            is_production_mode,
            transaction_type,
            dt,
            max_char_length=max_char_length,
            mmap_mode=mmap_mode,
            apply_data_augmentation=apply_data_augmentation,
        )
    )
    batcher = collate_batch_multi if is_multi_input else collate_batch
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
                collate_fn=batcher,
                num_workers=num_workers,
                prefetch_factor=10,
            )
        elif sampling_method == 'normal':
            loader = DataLoader(
                custom_dl,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=batcher,
                drop_last=True,
                prefetch_factor=10,
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
            collate_fn=batcher,
            drop_last=True,
        )
        return loader
