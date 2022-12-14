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

from config import Config
from datasets.dataset import get_dataloader, save_train_val_test_data
from experiment import Experiment, Experiment2
from laboratory.transaction_classification.constants import (
    CHILD_LABEL_NUM,
    PARENT_LABEL_NUM,
)
from models.model import load_model_from_config, transfer_weights
from pytorch_lightning.plugins import DDPPlugin


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
    os.makedirs(config.experiment_config.path_to_leaderboard, exist_ok=True)
    os.makedirs(config.trainer_config.path_to_model, exist_ok=True)
    os.makedirs("/home/dataset", exist_ok=True)


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
    if not config.experiment_config.load_data_from_disk:
        save_train_val_test_data(config)

    train_dl = get_dataloader(
        stage='train',
        dataset_dir=config.experiment_config.dataset_dir,
        is_production_mode=config.experiment_config.is_production_mode,
        transaction_type=config.experiment_config.transaction_type,
        mmap_mode=config.experiment_config.mmap_mode,
        dt=config.experiment_config.dt,
        max_char_length=config.model_config.seq_len,
        batch_size=config.experiment_config.batch_size,
        sampling_method=config.experiment_config.sampling_method,
        is_multi_input=config.experiment_config.multi_input,
        num_workers=config.experiment_config.num_workers,
        apply_data_augmentation=config.experiment_config.apply_data_augmentation,
    )
    logger.info("Loaded train dataloader")
    val_dl = get_dataloader(
        stage='validation',
        dataset_dir=config.experiment_config.dataset_dir,
        is_production_mode=config.experiment_config.is_production_mode,
        transaction_type=config.experiment_config.transaction_type,
        mmap_mode=config.experiment_config.mmap_mode,
        dt=config.experiment_config.dt,
        max_char_length=config.model_config.seq_len,
        batch_size=config.experiment_config.batch_size,
        sampling_method='normal',
        is_multi_input=config.experiment_config.multi_input,
        num_workers=config.experiment_config.num_workers,
        apply_data_augmentation=False
    )
    logger.info("Loaded validation dataloader")

    model = load_model_from_config(config)
    if config.experiment_config.load_pretrained_model:
        transfer_weights(
            model,
            config.experiment_config.pretrained_model_path,
            exclude_head=True,
            is_multi_input=config.experiment_config.multi_input,
        )
        logger.info("weight transfer success!")
    logger.info("Model Loaded")
    exp_cls = (
        Experiment2 if config.experiment_config.multi_input else Experiment
    )
    expt = exp_cls(
        model=model,
        num_child_class=CHILD_LABEL_NUM,
        num_parent_class=PARENT_LABEL_NUM,
        exp_config=config.experiment_config,
        train_dl=train_dl,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(
        monitor="val/loss"
        if config.experiment_config.is_pretraining
        else "val/child_loss",
        min_delta=0.00,
        patience=config.experiment_config.early_stop_patience,
        verbose=False,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss"
        if config.experiment_config.is_pretraining
        else "val/child_balanced_accuracy",
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
        accelerator=None if config.trainer_config.num_gpus == 1 else 'ddp',
        plugins=DDPPlugin(find_unused_parameters=False),
        replace_sampler_ddp=False,
    )

    trainer.fit(expt, train_dl, val_dl)

    test_dl = get_dataloader(
        stage='test',
        dataset_dir=config.experiment_config.dataset_dir,
        is_production_mode=config.experiment_config.is_production_mode,
        transaction_type=config.experiment_config.transaction_type,
        mmap_mode=config.experiment_config.mmap_mode,
        dt=config.experiment_config.dt,
        max_char_length=config.model_config.seq_len,
        batch_size=config.experiment_config.batch_size,
        sampling_method='normal',
        is_multi_input=config.experiment_config.multi_input,
        num_workers=config.experiment_config.num_workers,
        apply_data_augmentation=False,
    )
    trainer.val_check_interval = 1
    trainer.test(ckpt_path="best", dataloaders=test_dl)


if __name__ == "__main__":
    main()
