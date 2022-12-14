import datetime
import os
from dataclasses import asdict, astuple, dataclass, field
from pathlib import Path
from typing import List

from pydantic import BaseModel

from laboratory.transaction_classification.constants import (
    CHILD_LABEL_NUM,
    PARENT_LABEL_NUM,
)

now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
transaction_type = 'card'  # bankaccount, card, payment, prepayment
path_to_result = (
    f"/home/{transaction_type}_experiment_result"
)  # "/home" #  "/home/ubuntu/swlee"


@dataclass
class InceptionConfig:
    num_filters: int = 64
    depth: int = 6
    se: bool = True
    bn: bool = True
    fc_dropout: float = 0


@dataclass
class TSTConfig:
    n_layers: int = 6
    d_model: int = 128
    n_heads: int = 16
    d_ff: int = 256
    attn_dropout: float = 0
    dropout: float = 0
    fc_dropout: float = 0
    bn: bool = True


@dataclass
class LSTM_FCNConfig:
    hidden_size: int = 128
    rnn_layers: int = 3
    cell_dropout: float = 0.2
    rnn_dropout: float = 0.5
    bidirectional: bool = True
    shuffle: bool = True
    fc_dropout: float = 0.5
    conv_layers: List = field(default_factory=lambda: [256, 512, 256])
    kss: List = field(default_factory=lambda: [7, 5, 3])
    se: int = 1


@dataclass
class LSTMConfig:
    hidden_size: List = field(default_factory=lambda: [128, 128, 128])
    n_layers: int = 3
    bidirectional: bool = True


@dataclass
class GRUConfig:
    hidden_size: List = field(default_factory=lambda: [128, 128, 128])
    n_layers: int = 3
    bidirectional: bool = True


@dataclass
class mWDNPlusConfig:
    levels: int = 3
    wavelet: None = None
    base_model: None = None


@dataclass
class TSiTConfig:
    pass


@dataclass
class TSPerceiverConfig:
    pass


@dataclass
class TextCNNConfig:
    path_to_model_config: str = (
        '../legacy/bank_account_category_classification/config.yaml'
    )


@dataclass
class ModelConfig:
    model_name: str = 'TSTPlus'  # 'TextCNN', 'Inception', 'TST', 'TSTPlus', 'LSTM_FCN', 'LSTM', 'GRU', 'mWDNPlus', 'TSiT', 'TSPerceiver'
    c_in: int = 128
    mid_c_out: int = 32
    c_out: int = CHILD_LABEL_NUM
    seq_len: int = 75
    n_vocabs: int = 247
    inception_config: object = InceptionConfig()
    tst_config: object = TSTConfig()
    lstm_fcn_config: object = LSTM_FCNConfig()
    lstm_config: object = LSTMConfig()
    gru_config: object = GRUConfig()
    wWDNPlus_config: object = mWDNPlusConfig()
    tsit_config: object = TSiTConfig()
    ts_perceiver_config: object = TSPerceiverConfig()
    textcnn_config = TextCNNConfig()


@dataclass
class TabularModelConfig:
    model_name: str = 'Simple'  # Simple, TabNet, TabTransformer,
    c_in: int = 18
    mid_c_out: int = 16
    c_out: int = CHILD_LABEL_NUM


@dataclass
class TrainerConfig:
    path_to_model: str = os.path.join(
        path_to_result, 'model_weights', now_time
    )
    tb_log_dir: str = os.path.join(path_to_result, 'tb_logs', now_time)
    tb_log_graph: bool = False
    num_gpus: int = 1
    grad_clip: float = 0.5
    use_swa: bool = True
    log_cycle: int = 1000
    mixed_precision: bool = False
    auto_lr_find: bool = False
    val_check_interval: float = 1.0


@dataclass
class ExperimentConfig:
    path_to_result: str = path_to_result
    path_to_logs: str = os.path.join(path_to_result, 'logs')
    experiment_name = 'tst_multi_loss_multi_input_simple_ensemble_simple_self_supervised_AdamW'  # 'tst_multi_loss_multi_input_simple_ensemble_simple_with_card_dataset_only_expense'
    transaction_type: str = transaction_type
    path_to_leaderboard: str = (
        f'{path_to_result}/{transaction_type}_leaderboard'
    )
    use_implicit_parent_category: bool = True
    max_epochs: int = 125
    batch_size: int = 1024
    lr: float = 1e-4
    max_lr: float = 1e-4
    smoothing_factor: float = 0.1
    optimizer: str = 'AdamW'  # 'RAdam', 'AdamW'
    use_weighted_sampling: bool = False
    sampling_method: str = 'gold_standard_weighted'  # 'normal', 'weighted'
    loss: str = 'CE'  # 'CE', 'weightedCE', 'Focal'
    lr_sched: str = None  # None or 'one_cycle' or 'step' or 'cosine_anneal'
    cos_min_lr: float = 1e-7
    cos_max_lr: float = 1e-2
    focal_gamma: float = 2
    focal_alpha: List = None  # None, field(default_factory=lambda: [0.1, 0.5, 0.4])   # 0 - background, 1,2 - foreground
    early_stop_patience: int = 50  # max_epochs * 2 // 4
    multi_input: bool = True
    multi_input_model: str = 'custom'  # None, 'tsai', 'custom'
    use_child_parent_loss: bool = True
    ensemble_method: str = 'simple'  # 'simple', 'tabnet'
    dataset_dir: str = '/home/dataset'
    is_pretraining: bool = False
    load_pretrained_model: bool = True
    pretrained_model_path: str = "/home/electronic_payment_experiment_result/model_weights/20220914_094343/model_epoch=34_step=    289940.ckpt"
    is_production_mode: bool = True
    concat_cross_domain_dataset: bool = True
    apply_data_augmentation: bool = True
    process_onehot: bool = True
    dt: str = '2022-10-24'  # dt=(datetime.datetime.now() - datetime.timedelta(days=3)).strftime('%Y-%m-%d')
    num_workers: int = 32
    mmap_mode: str = 'r'
    load_data_from_disk: bool = False


@dataclass
class Config:
    model_config: object = ModelConfig()
    tabular_model_config: object = TabularModelConfig()
    trainer_config: object = TrainerConfig()
    experiment_config: object = ExperimentConfig()
    config_path: str = Path(__file__).absolute()


if __name__ == "__main__":
    config_args = Config()
    print(config_args)
