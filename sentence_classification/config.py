import datetime
import os
from dataclasses import asdict, astuple, dataclass, field
from pathlib import Path
from typing import List

from pydantic import BaseModel

from sentence_classification.constants import MAX_CHARACTER_VECTOR_LENGTH, OOV_INDEX, full_name_label_to_num_map

now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
path_to_result = (
    f"./experiment_result"
)


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
class ModelConfig:
    model_name: str = 'TSTPlus'  # 'Inception', 'TST', 'TSTPlus', 'LSTM_FCN', 'LSTM', 'GRU', 'mWDNPlus', 'TSiT', 'TSPerceiver'
    c_in: int = 128
    mid_c_out: int = 32
    c_out: int = len(full_name_label_to_num_map)
    seq_len: int = MAX_CHARACTER_VECTOR_LENGTH
    n_vocabs: int = OOV_INDEX + 1
    inception_config: object = InceptionConfig()
    tst_config: object = TSTConfig()
    lstm_fcn_config: object = LSTM_FCNConfig()
    lstm_config: object = LSTMConfig()
    gru_config: object = GRUConfig()
    wWDNPlus_config: object = mWDNPlusConfig()


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
    log_cycle: int = 100
    mixed_precision: bool = False
    auto_lr_find: bool = False
    val_check_interval: float = 1.0


@dataclass
class ExperimentConfig:
    path_to_result: str = path_to_result
    path_to_logs: str = os.path.join(path_to_result, 'logs')
    experiment_name = 'tst_pretraining'
    max_epochs: int = 125
    batch_size: int = 32
    lr: float = 1e-4
    max_lr: float = 1e-4
    smoothing_factor: float = 0.1
    optimizer: str = 'AdamW'  # 'RAdam', 'AdamW'
    use_weighted_sampling: bool = False
    sampling_method: str = 'normal'  # 'normal', 'weighted'
    loss: str = 'CE'  # 'CE', 'weightedCE', 'Focal'
    lr_sched: str = None  # None or 'one_cycle' or 'step' or 'cosine_anneal'
    cos_min_lr: float = 1e-7
    cos_max_lr: float = 1e-2
    focal_gamma: float = 2
    focal_alpha: List = None
    early_stop_patience: int = 50
    dataset_dir: str = './data'
    is_pretraining: bool = True
    load_pretrained_model: bool = False
    pretrained_model_path: str = ""
    apply_data_augmentation: bool = True
    num_workers: int = 0


@dataclass
class Config:
    model_config: object = ModelConfig()
    trainer_config: object = TrainerConfig()
    experiment_config: object = ExperimentConfig()
    config_path: str = Path(__file__).absolute()


if __name__ == "__main__":
    config_args = Config()
    print(config_args)
