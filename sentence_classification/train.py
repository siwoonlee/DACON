import os
import shutil

import pandas as pd
import torch
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import (
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from config import Config
from sentence_classification.datasets.dataset import get_dataloader, split_dataset_equal_category_balance
from sentence_classification.experiment import Experiment
from sentence_classification.models.model import load_model_from_config, transfer_weights


def create_directories(config):
    os.makedirs(config.experiment_config.path_to_result, exist_ok=True)
    os.makedirs(
        os.path.join(
            config.trainer_config.tb_log_dir,
            config.experiment_config.experiment_name,
        ),
        exist_ok=True,
    )
    os.makedirs(config.experiment_config.path_to_logs, exist_ok=True)
    os.makedirs(config.trainer_config.path_to_model, exist_ok=True)


def main():
    config = Config()
    create_directories(config)
    log_path = os.path.join(
        config.experiment_config.path_to_logs, "{time}_log.log"
    )
    logger.add(log_path)
    shutil.copy(
        config.config_path,
        os.path.join(
            config.trainer_config.path_to_model,
            os.path.basename(config.config_path),
        ),
    )
    logger.info("Copied config file!")
    df = pd.read_csv(os.path.join(config.experiment_config.dataset_dir, "train.csv"))
    train_dataset_df, val_dataset_df = split_dataset_equal_category_balance(df, test_ratio=0.2)

    train_dl = get_dataloader(
        stage='train',
        dataset_df=train_dataset_df,
        max_char_length=config.model_config.seq_len,
        batch_size=config.experiment_config.batch_size,
        sampling_method=config.experiment_config.sampling_method,
        num_workers=config.experiment_config.num_workers,
        apply_data_augmentation=config.experiment_config.apply_data_augmentation,
    )
    logger.info("Loaded train dataloader")
    val_dl = get_dataloader(
        stage='val',
        dataset_df=val_dataset_df,
        max_char_length=config.model_config.seq_len,
        batch_size=config.experiment_config.batch_size,
        sampling_method=config.experiment_config.sampling_method,
        num_workers=config.experiment_config.num_workers,
        apply_data_augmentation=config.experiment_config.apply_data_augmentation,
    )
    logger.info("Loaded validation dataloader")

    model = load_model_from_config(config)
    if config.experiment_config.load_pretrained_model:
        transfer_weights(
            model,
            config.experiment_config.pretrained_model_path,
            exclude_head=True,
        )
        logger.info("weight transfer success!")
    logger.info("Model Loaded")

    expt = Experiment(
        model=model,
        exp_config=config.experiment_config,
        train_dl=train_dl,
        is_inference=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        min_delta=0.00,
        patience=config.experiment_config.early_stop_patience,
        verbose=False,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss"
        if config.experiment_config.is_pretraining
        else "val/weighted_f1_score",
        mode="min" if config.experiment_config.is_pretraining else "max",
        dirpath=config.trainer_config.path_to_model,
        save_top_k=1,
        filename='model_{epoch:02d}_{step:10d}',
    )
    stochastic_weight_avg_callback = StochasticWeightAveraging()
    callbacks = (
        [
            lr_monitor,
            checkpoint_callback,
            early_stop_callback,
            stochastic_weight_avg_callback,
        ]
        if config.trainer_config.use_swa
        else [lr_monitor, checkpoint_callback, early_stop_callback]
    )
    tb_logger = TensorBoardLogger(
        config.trainer_config.tb_log_dir,
        name=config.experiment_config.experiment_name,
        log_graph=config.trainer_config.tb_log_graph,
    )
    trainer = Trainer(
        auto_lr_find=config.trainer_config.auto_lr_find,
        max_epochs=config.experiment_config.max_epochs,
        gpus=config.trainer_config.num_gpus
        if torch.cuda.is_available()
        else None,
        precision=16 if config.trainer_config.mixed_precision else 32,
        logger=tb_logger,
        gradient_clip_val=config.trainer_config.grad_clip,
        callbacks=callbacks,
        log_every_n_steps=config.trainer_config.log_cycle,
        val_check_interval=config.trainer_config.val_check_interval,
    )

    trainer.fit(expt, train_dl, val_dl)


if __name__ == "__main__":
    main()
