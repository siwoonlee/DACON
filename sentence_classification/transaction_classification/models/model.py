import torch
import torch.nn as nn

# from pytorch_tabnet.tab_network import AttentiveTransformer, TabNet
# from tab_transformer_pytorch.tab_transformer_pytorch import TabTransformer
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

from laboratory.legacy.bank_account_category_classification.models.textcnn import (
    TextCNN,
)
from laboratory.transaction_classification.utils.vectorizer import OOV_INDEX
from laboratory.utils import get_config


class ConcatEnsembleMultiInputModel(nn.Module):
    def __init__(
        self,
        model1,
        model2,
        c_out1,
        c_out2,
        n_out,
        device,
        method='simple',
        is_inference=False,
    ):
        super().__init__()
        self.is_inference = is_inference
        self.model1 = model1.to(device=device, dtype=torch.float32)
        self.model2 = model2.to(device=device, dtype=torch.float32)
        self.method = method
        concat_len = c_out1 + c_out2
        if method == 'simple':
            self.ensemble_model = nn.Sequential(
                nn.BatchNorm1d(concat_len),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(concat_len, concat_len),
                nn.BatchNorm1d(concat_len),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(concat_len, concat_len),
                nn.BatchNorm1d(concat_len),
                nn.ReLU(),
                nn.Linear(concat_len, n_out),
            )
        elif method == 'tabnet':
            self.ensemble_model = TabNet(
                input_dim=concat_len, output_dim=n_out
            ).to(device=device, dtype=torch.float32)
        else:
            raise NotImplementedError

    def forward(self, x_dict):
        res1 = self.model1(x_dict['title_vec_inputs'])
        res2 = self.model2(x_dict['transaction_amount_inputs'])
        if isinstance(res2, tuple):
            res2, M_loss1 = res2[0], res2[1]
        else:
            M_loss1 = 0
        res = [res1, res2]
        res = torch.cat(res, dim=1)
        res = self.ensemble_model(res)
        if isinstance(res, tuple):
            res, M_loss = res[0], res[1] + M_loss1
        else:
            M_loss = 0 - M_loss1
        if self.is_inference:
            return torch.argmax(res, axis=1)
        else:
            return res, M_loss


class SimpleTabularModel(nn.Module):
    def __init__(self, n_in, c_out):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, c_out),
        )
        self.c_out = c_out

    def forward(self, x):
        return self.model(x)


class SwapAxis(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.swapaxes(x, 1, 2)


def get_tabular_model(config_args, device):
    if config_args.model_name == 'Simple':
        model2 = SimpleTabularModel(
            config_args.c_in, config_args.mid_c_out
        ).to(device=device, dtype=torch.float32)
    elif config_args.model_name == 'TabNet':
        model2 = TabNet(config_args.c_in, config_args.mid_c_out).to(
            device=device, dtype=torch.float32
        )
    else:
        raise NotImplementedError
    return model2


def get_model(
    config_args,
    multi_input=False,
    multi_input_model=None,
    is_pretrained_model=False,
):
    if config_args.model_name == 'TextCNN':
        model_config = config_args.textcnn_config
        model = TextCNN(
            config=get_config(model_config.path_to_model_config),
            n_vocabs=config_args.n_vocabs,
            num_categories=config_args.c_out,
            vectors=None,
        )
        return model
    if multi_input and 'custom' in multi_input_model:
        c_out = config_args.mid_c_out
    elif is_pretrained_model:
        c_out = config_args.seq_len
    else:
        c_out = config_args.c_out
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
        multi_input=config.experiment_config.multi_input,
        multi_input_model=config.experiment_config.multi_input_model,
        is_pretrained_model=config.experiment_config.is_pretraining,
    ).to(device=device, dtype=torch.float32)
    if config.experiment_config.multi_input:
        model2 = get_tabular_model(
            config_args=config.tabular_model_config, device=device
        )
        if 'tsai' in config.experiment_config.multi_input_model:
            model = MultiInputNet(model, model2)
        elif 'custom' in config.experiment_config.multi_input_model:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
            model = ConcatEnsembleMultiInputModel(
                model,
                model2,
                c_out1=config.model_config.mid_c_out,
                c_out2=config.tabular_model_config.mid_c_out,
                n_out=config.model_config.c_out,
                device=device,
                method=config.experiment_config.ensemble_method,
                is_inference=is_inference,
            ).to(device=device, dtype=torch.float32)
    return model


def transfer_weights(
    model, weights_path, exclude_head: bool = True, is_multi_input=False
):
    """Utility function that allows to easily transfer weights between models.
    Taken from the great self-supervised repository created by Kerem Turgutlu.
    https://github.com/KeremTurgutlu/self_supervised/blob/d87ebd9b4961c7da0efd6073c42782bbc61aaa2e/self_supervised/utils.py"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if is_multi_input:
        state_dict = model.model1.state_dict()
    else:
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
