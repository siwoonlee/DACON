import copy
import datetime
import os
import random

import tqdm
import numpy as np
import pandas as pd
import torch
from loguru import logger
from pyhive import hive
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from laboratory.transaction_classification.utils.sampler import CustomWeightedRandomSampler
from sklearn.model_selection import train_test_split

from laboratory.transaction_classification.utils.vectorizer import (
    get_padded_character_vector,
)
from laboratory.transaction_classification.constants import (
    GOLD_STANDARD_QUERY,
    BACKOFFICE_RULESET_QUERY,
    DEFAULT_CATEGORY_QUERY,
    TRANSACTION_AMOUNT_RANGE_LIST,
    DEFAULT_CATEGORY_LIST,
    EFIN_BACKOFFICE_RULESET_QUERY,
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
                title = apply_replace_method_to_transaction_title(title)
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


def apply_replace_method_to_transaction_title(
    title: str,
    targets=['주식회사', '(주)', '(주', '주)'],
) -> str:
    substituent = random.choice(targets)
    substitute = random.choice(targets)
    title = title.replace(substituent, substitute)
    return title


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


def load_old_dataset(transaction_type='card', dataset_dir='/home/dataset'):
    if transaction_type == 'card':
        df = pd.read_pickle(
            os.path.join(
                dataset_dir,
                f"labeled_card_transaction_data_final.pkl",
            )
        )
    elif transaction_type == 'bankaccount':
        df1 = pd.read_pickle(
            os.path.join(
                dataset_dir,
                f"labeled_bankaccount_income_transaction_data_final.pkl",
            )
        )
        df2 = pd.read_pickle(
            os.path.join(
                dataset_dir,
                f"labeled_bankaccount_expense_transaction_data_final.pkl",
            )
        )
        df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    else:
        raise NotImplementedError
    df['is_gold_standard'] = False
    return df


def load_gold_standard_dataset(
    transaction_type='card', dataset_dir='/home/dataset', dt='2022-10-24'
):
    try:
        df = pd.read_pickle(
            os.path.join(dataset_dir, f"{dt}_{transaction_type}.pkl")
        )
    except:
        conn = hive.Connection(
            host="10.150.8.175", port=10001, username="swlee@rainist.com"
        )
        pd.read_sql(
            """
                set spark.sql.thriftserver.scheduler.pool=bulk
            """,
            conn,
        )
        df = pd.read_sql(
            GOLD_STANDARD_QUERY.format(
                transaction_type=transaction_type, target_date=dt
            ),
            conn,
        )
        df.to_pickle(os.path.join(dataset_dir, f"{dt}_{transaction_type}.pkl"))
    return df


def load_default_category_df(dataset_dir='/home/dataset', dt='2022-10-24'):
    try:
        df = pd.read_pickle(
            os.path.join(dataset_dir, f"{dt}_default_category.pkl")
        )
    except:
        conn = hive.Connection(
            host="10.150.8.175", port=10001, username="swlee@rainist.com"
        )
        df = pd.read_sql(DEFAULT_CATEGORY_QUERY.format(target_date=dt), conn)
        df.to_pickle(os.path.join(dataset_dir, f"{dt}_default_category.pkl"))
    return df


def load_backoffice_ruleset_df(
    transaction_type='card', dataset_dir='/home/dataset', dt='2022-10-24'
):
    try:
        df = pd.read_pickle(
            os.path.join(
                dataset_dir, f"{dt}_{transaction_type}_backoffice_ruleset.pkl"
            )
        )
    except:
        conn = hive.Connection(
            host="10.150.8.175", port=10001, username="swlee@rainist.com"
        )
        query = (
            BACKOFFICE_RULESET_QUERY
            if transaction_type in ('card', 'bankaccount')
            else EFIN_BACKOFFICE_RULESET_QUERY
        )
        df = pd.read_sql(
            query.format(transaction_type=transaction_type, target_date=dt),
            conn,
        )
        df.to_pickle(
            os.path.join(
                dataset_dir, f"{dt}_{transaction_type}_backoffice_ruleset.pkl"
            )
        )
    return df


def feature_engineer_backoffice_ruleset_df(
    backoffice_ruleset_df: pd.DataFrame, default_category_df: pd.DataFrame
) -> pd.DataFrame:
    default_category_dict = default_category_df.set_index(
        'default_category_id'
    ).to_dict()
    backoffice_ruleset_df[
        'accountbook_transaction_type'
    ] = backoffice_ruleset_df['default_category_id'].apply(
        lambda x: default_category_dict['accountbook_transaction_type'][x]
    )
    backoffice_ruleset_df['user_parent_category'] = backoffice_ruleset_df[
        'default_category_id'
    ].apply(lambda x: default_category_dict['parent_category_name'][x])
    backoffice_ruleset_df['user_child_category'] = backoffice_ruleset_df[
        'default_category_id'
    ].apply(lambda x: default_category_dict['child_category_name'][x])
    backoffice_ruleset_df['transaction_amount_range'] = backoffice_ruleset_df[
        'default_category_id'
    ].apply(lambda x: 'unknown')
    backoffice_ruleset_df = backoffice_ruleset_df[
        [
            'title',
            'user_parent_category',
            'user_child_category',
            'accountbook_transaction_type',
            'default_category_id',
            'transaction_amount_range',
        ]
    ]
    backoffice_ruleset_df = make_full_category_name(backoffice_ruleset_df)
    backoffice_ruleset_df = make_child_parent_label(backoffice_ruleset_df)
    return backoffice_ruleset_df


def make_full_category_name(df: pd.DataFrame) -> pd.DataFrame:
    df = copy.deepcopy(df)
    try:
        df['full_category_name'] = df.apply(
            lambda x: f"{x['accountbook_transaction_type']}-{x['user_parent_category']}-{x['user_child_category']}",
            axis=1,
        )
        df['type_parent_category_name'] = df.apply(
            lambda x: f"{x['accountbook_transaction_type']}-{x['user_parent_category']}",
            axis=1,
        )
    except KeyError:
        df['full_category_name'] = df.apply(
            lambda x: f"{x['accountbook_transaction_type']}-{x['parent_category_name']}-{x['child_category_name']}",
            axis=1,
        )
        df['type_parent_category_name'] = df.apply(
            lambda x: f"{x['accountbook_transaction_type']}-{x['parent_category_name']}",
            axis=1,
        )
    return df


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


def make_child_parent_label(df):
    full_category_name_to_child_label_dict = {
        key: val for val, key in enumerate(DEFAULT_CATEGORY_LIST)
    }
    df['child_label'] = df['full_category_name'].apply(
        lambda x: full_category_name_to_child_label_dict[x]
    )
    type_parent_category_name = set(
        ['-'.join(x.split('-')[:2]) for x in DEFAULT_CATEGORY_LIST]
    )
    type_parent_category_name_to_parent_label_dict = {
        key: val for val, key in enumerate(type_parent_category_name)
    }
    df['parent_label'] = df['type_parent_category_name'].apply(
        lambda x: type_parent_category_name_to_parent_label_dict[x]
    )
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


def load_feature_engineered_default_category_df():
    default_category_df = load_default_category_df()
    default_category_df = make_full_category_name(default_category_df)
    default_category_df = make_child_parent_label(default_category_df)
    return default_category_df


def save_data_to_disk(config, data, stage='train'):
    data['title'].to_pickle(
        os.path.join(
            config.experiment_config.dataset_dir,
            f"{config.experiment_config.dt}_"
            f"{config.experiment_config.transaction_type}_"
            f"{config.experiment_config.is_production_mode}_"
            f"title_"
            f"{stage}.pkl",
        )
    )
    data['is_gold_standard'].to_pickle(
        os.path.join(
            config.experiment_config.dataset_dir,
            f"{config.experiment_config.dt}_"
            f"{config.experiment_config.transaction_type}_"
            f"{config.experiment_config.is_production_mode}_"
            f"is_gold_standard_"
            f"{stage}.pkl",
        )
    )
    np.save(
        os.path.join(
            config.experiment_config.dataset_dir,
            f"{config.experiment_config.dt}_"
            f"{config.experiment_config.transaction_type}_"
            f"{config.experiment_config.is_production_mode}_"
            f"TRANSACTION_AMOUNT_RANGE_LIST_"
            f"{stage}.npy",
        ),
        data[TRANSACTION_AMOUNT_RANGE_LIST].values,
    )
    np.save(
        os.path.join(
            config.experiment_config.dataset_dir,
            f"{config.experiment_config.dt}_"
            f"{config.experiment_config.transaction_type}_"
            f"{config.experiment_config.is_production_mode}_"
            f"child_label_"
            f"{stage}.npy",
        ),
        data['child_label'].values,
    )
    np.save(
        os.path.join(
            config.experiment_config.dataset_dir,
            f"{config.experiment_config.dt}_"
            f"{config.experiment_config.transaction_type}_"
            f"{config.experiment_config.is_production_mode}_"
            f"parent_label_"
            f"{stage}.npy",
        ),
        data['parent_label'].values,
    )
    logger.info(f"saved {stage} data!")


def save_train_val_test_data(config):
    if (
        config.experiment_config.transaction_type in ('card', 'payment')
        and config.experiment_config.concat_cross_domain_dataset
    ):
        train_data, val_data, test_data = get_single_transaction_type_train_val_test_data(
            transaction_type='card',
            dataset_dir=config.experiment_config.dataset_dir,
            is_production_mode=config.experiment_config.is_production_mode,
            process_onehot=config.experiment_config.process_onehot,
            dt=config.experiment_config.dt,
        )
        _train_data, _val_data, _test_data = get_single_transaction_type_train_val_test_data(
            transaction_type='payment',
            dataset_dir=config.experiment_config.dataset_dir,
            is_production_mode=config.experiment_config.is_production_mode,
            process_onehot=config.experiment_config.process_onehot,
            dt=config.experiment_config.dt,
        )
        train_data = pd.concat([train_data, _train_data], axis=0).reset_index(
            drop=True
        )
        val_data = pd.concat([val_data, _val_data], axis=0).reset_index(
            drop=True
        )
        test_data = pd.concat([test_data, _test_data], axis=0).reset_index(
            drop=True
        )
    elif (
        config.experiment_config.transaction_type
        in ('bankaccount', 'prepayment')
        and config.experiment_config.concat_cross_domain_dataset
    ):
        train_data, val_data, test_data = get_single_transaction_type_train_val_test_data(
            transaction_type='bankaccount',
            dataset_dir=config.experiment_config.dataset_dir,
            is_production_mode=config.experiment_config.is_production_mode,
            process_onehot=config.experiment_config.process_onehot,
            dt=config.experiment_config.dt,
        )
        _train_data, _val_data, _test_data = get_single_transaction_type_train_val_test_data(
            transaction_type='prepayment',
            dataset_dir=config.experiment_config.dataset_dir,
            is_production_mode=config.experiment_config.is_production_mode,
            process_onehot=config.experiment_config.process_onehot,
            dt=config.experiment_config.dt,
        )
        train_data = pd.concat([train_data, _train_data], axis=0).reset_index(
            drop=True
        )
        val_data = pd.concat([val_data, _val_data], axis=0).reset_index(
            drop=True
        )
        test_data = pd.concat([test_data, _test_data], axis=0).reset_index(
            drop=True
        )
    else:
        train_data, val_data, test_data = get_single_transaction_type_train_val_test_data(
            transaction_type=config.experiment_config.transaction_type,
            dataset_dir=config.experiment_config.dataset_dir,
            is_production_mode=config.experiment_config.is_production_mode,
            process_onehot=config.experiment_config.process_onehot,
            dt=config.experiment_config.dt,
        )
    save_data_to_disk(config, train_data, stage='train')
    save_data_to_disk(config, val_data, stage='validation')
    save_data_to_disk(config, test_data, stage='test')


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
            logger.info("use oversampling!")
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
        elif sampling_method == 'gold_standard_weighted':
            logger.info("use gold standard weighted oversampling!")
            labels = custom_dl.is_gold_standard
            sample_weights = [5e-10 if is_gold_standard else 1e-10 for is_gold_standard in labels]
            loader = DataLoader(
                custom_dl,
                sampler=CustomWeightedRandomSampler(
                    sample_weights,
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
            raise Exception(
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
