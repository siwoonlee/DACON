import torch
import torch.nn as nn

from tsai.models.all import (
    LSTM_FCN,
    TST,
    GRUPlus,
    InceptionTimePlus,
    LSTMPlus,
    MultiInputNet,
    TSiTPlus,
    TSPerceiver,
    TSTPlus,
    mWDNPlus,
)


class CustomModel(nn.Module):
    def __init__(
        self,
        model,
        mid_c_out,
        n_out,
        device,
        is_pretraining=False,
        is_inference=False,
    ):
        super().__init__()
        self.is_inference = is_inference
        self.is_pretraining = is_pretraining
        self.model = model.to(device=device, dtype=torch.float32)
        if not is_pretraining:
            self.head_유형 = nn.Sequential(
                nn.BatchNorm1d(mid_c_out),
                nn.ReLU(),
                nn.Linear(mid_c_out, mid_c_out),
                nn.BatchNorm1d(mid_c_out),
                nn.ReLU(),
                nn.Linear(mid_c_out, 4),
            )
            self.head_극성 = nn.Sequential(
                nn.BatchNorm1d(mid_c_out),
                nn.ReLU(),
                nn.Linear(mid_c_out, mid_c_out),
                nn.BatchNorm1d(mid_c_out),
                nn.ReLU(),
                nn.Linear(mid_c_out, 3),
            )
            self.head_시제 = nn.Sequential(
                nn.BatchNorm1d(mid_c_out),
                nn.ReLU(),
                nn.Linear(mid_c_out, mid_c_out),
                nn.BatchNorm1d(mid_c_out),
                nn.ReLU(),
                nn.Linear(mid_c_out, 3),
            )
            self.head_확실성 = nn.Sequential(
                nn.BatchNorm1d(mid_c_out),
                nn.ReLU(),
                nn.Linear(mid_c_out, mid_c_out),
                nn.BatchNorm1d(mid_c_out),
                nn.ReLU(),
                nn.Linear(mid_c_out, 2),
            )
            self.head_final = nn.Sequential(
                nn.BatchNorm1d(12),
                nn.ReLU(),
                nn.Linear(12, mid_c_out),
                nn.BatchNorm1d(mid_c_out),
                nn.ReLU(),
                nn.Linear(mid_c_out, mid_c_out),
                nn.BatchNorm1d(mid_c_out),
                nn.ReLU(),
                nn.Linear(mid_c_out, n_out),
            )

    def forward(self, x):
        mid_vec = self.model(x)
        if self.is_pretraining:
            return mid_vec
        out_유형 = self.head_유형(mid_vec)
        out_극성 = self.head_극성(mid_vec)
        out_시제 = self.head_시제(mid_vec)
        out_확실성 = self.head_확실성(mid_vec)

        res = torch.cat([out_유형, out_극성, out_시제, out_확실성], dim=1)
        res = self.head_final(res)
        if self.is_inference:
            return torch.argmax(res, axis=1)
        else:
            return res, out_유형, out_극성, out_시제, out_확실성


class SwapAxis(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.swapaxes(x, 1, 2)


def get_model(
    config_args,
    is_pretrained_model=False,
):
    if is_pretrained_model:
        c_out = config_args.seq_len
    else:
        c_out = config_args.mid_c_out
    if config_args.model_name == 'Inception':
        model_config = config_args.inception_config
        model = InceptionTimePlus(
            c_in=config_args.c_in,
            c_out=c_out,
            seq_len=config_args.seq_len,
            bn=model_config.bn,
            nf=model_config.num_filters,
            depth=model_config.depth,
            se=model_config.se,
            fc_dropout=model_config.fc_dropout,
        )
    elif config_args.model_name == 'TST':
        model_config = config_args.tst_config
        model = TST(
            c_in=config_args.c_in,
            c_out=c_out,
            seq_len=config_args.seq_len,
            max_seq_len=config_args.seq_len,
            n_layers=model_config.n_layers,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
        )
    elif config_args.model_name == 'TSTPlus':
        model_config = config_args.tst_config
        model = TSTPlus(
            c_in=config_args.c_in,
            c_out=c_out,
            seq_len=config_args.seq_len,
            max_seq_len=config_args.seq_len,
            n_layers=model_config.n_layers,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            d_ff=model_config.d_ff,
            fc_dropout=model_config.fc_dropout,
            dropout=model_config.dropout,
            attn_dropout=model_config.attn_dropout,
            bn=model_config.bn,
        )
    elif config_args.model_name == 'TSiT':
        model_config = config_args.tsit_config
        model = TSiTPlus(
            c_in=config_args.c_in, c_out=c_out, seq_len=config_args.seq_len
        )
    elif config_args.model_name == 'TSPerceiver':
        model_config = config_args.ts_perceiver_config
        model = TSPerceiver(
            c_in=config_args.c_in, c_out=c_out, seq_len=config_args.seq_len
        )
    elif config_args.model_name == "LSTM":
        model_config = config_args.lstm_config
        model = LSTMPlus(
            c_in=config_args.c_in,
            c_out=c_out,
            seq_len=config_args.seq_len,
            hidden_size=model_config.hidden_size,
            n_layers=model_config.n_layers,
            bidirectional=model_config.bidirectional,
        )
    elif config_args.model_name == "GRU":
        model_config = config_args.gru_config
        model = GRUPlus(
            c_in=config_args.c_in,
            c_out=c_out,
            seq_len=config_args.seq_len,
            hidden_size=model_config.hidden_size,
            n_layers=model_config.n_layers,
            bidirectional=model_config.bidirectional,
        )
    elif config_args.model_name == "LSTM_FCN":
        model_config = config_args.lstm_fcn_config
        model = LSTM_FCN(
            c_in=config_args.c_in,
            c_out=c_out,
            seq_len=config_args.seq_len,
            hidden_size=model_config.hidden_size,
            rnn_layers=model_config.rnn_layers,
            cell_dropout=model_config.cell_dropout,
            rnn_dropout=model_config.rnn_dropout,
            bidirectional=model_config.bidirectional,
            shuffle=model_config.shuffle,
            fc_dropout=model_config.fc_dropout,
            conv_layers=model_config.conv_layers,
            kss=model_config.kss,
            se=model_config.se,
        )
    elif config_args.model_name == "mWDNPlus":
        model_config = config_args.wWDNPlus_config
        model = mWDNPlus(
            c_in=config_args.c_in,
            c_out=c_out,
            seq_len=config_args.seq_len,
            levels=model_config.levels,
            wavelet=model_config.wavelet,
            base_model=model_config.base_model,
            base_arch=InceptionTimePlus,
        )
    else:
        raise NotImplementedError
    embedding_table = nn.Embedding(config_args.n_vocabs, config_args.c_in)
    return nn.Sequential(embedding_table, SwapAxis(), model)


def load_model_from_config(config, is_inference=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(
        config.model_config,
        is_pretrained_model=config.experiment_config.is_pretraining,
    ).to(device=device, dtype=torch.float32)
    model = CustomModel(
        model,
        config.model_config.seq_len if config.experiment_config.is_pretraining else config.model_config.mid_c_out,
        config.model_config.c_out,
        device,
        is_pretraining=config.experiment_config.is_pretraining,
        is_inference=is_inference,
    ).to(device=device, dtype=torch.float32)
    return model


def transfer_weights(
    model, weights_path, exclude_head: bool = True,
):
    """
    Utility function that allows to easily transfer weights between models.
    Taken from the great self-supervised repository created by Kerem Turgutlu.
    https://github.com/KeremTurgutlu/self_supervised/blob/d87ebd9b4961c7da0efd6073c42782bbc61aaa2e/self_supervised/utils.py
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = model.state_dict()
    new_state_dict = torch.load(weights_path, map_location=device)[
        'state_dict'
    ]
    matched_layers = 0
    unmatched_layers = []
    for name, param in state_dict.items():
        if exclude_head and 'head' in name:
            continue
        tmp_name = f'model.{name}'
        if tmp_name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[tmp_name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass  # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")
