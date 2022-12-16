import os
import importlib.util
from pathlib import Path
import torch
import pandas as pd

from sentence_classification.experiment import Experiment
from sentence_classification.models.model import load_model_from_config
from sentence_classification.constants import num_to_full_name_label_map
from sentence_classification.utils.vectorizer import get_padded_character_vector

def get_config_args(path_to_config):
    spec = importlib.util.spec_from_file_location("Config", os.path.join(path_to_config, "config.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_args = module.Config()
    return config_args


def load_model(path_to_model_ckpt, config_args, device):
    model = load_model_from_config(config_args, is_inference=True)
    pl_model = Experiment.load_from_checkpoint(
        model=model,
        exp_config=None,
        train_dl=None,
        is_inference=True,
        checkpoint_path=path_to_model_ckpt,
        map_location=device,
    )
    pl_model.eval()
    pl_model.to(device=device, dtype=torch.float32)
    return pl_model


def get_inference_samples(
    test_data_df,
    batch_size=4,
    max_char_length=300,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
):
    for i in range(0, len(test_data_df), batch_size):
        sample_list = []
        for idx in range(i, i+batch_size, 1):
            문장 = test_data_df['문장'].iloc[idx]
            문장_vec = get_padded_character_vector(
                문장,
                max_char_length,
            )
            sample_list.append(문장_vec)
        yield torch.tensor(sample_list, dtype=torch.int64).to(device=device)


def get_inference_result(
    pl_model,
    test_data_df,
    batch_size=4,
    max_char_length=300,
):
    overall_y_preds = []
    for batch in get_inference_samples(
        test_data_df,
        batch_size=4,
        max_char_length=300,
    ):
        y_preds = pl_model.model(batch)
        overall_y_preds.extend(y_preds.cpu().detach().numpy().tolist())
    return overall_y_preds


def main():
    path_to_model_ckpt = "../experiment_result/model_weights/20221216_230705/model_epoch=07_step=      3304.ckpt"
    path_to_config = str(Path(path_to_model_ckpt).parent)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config_args = get_config_args(path_to_config)
    pl_model = load_model(path_to_model_ckpt, config_args, device)
    test_data_df = pd.read_csv("../data/test.csv")
    overall_y_preds = get_inference_result(
        pl_model,
        test_data_df,
        batch_size=4,
        max_char_length=config_args.model_config.seq_len,
    )
    submission_df = test_data_df.drop(columns=['문장'])
    submission_df['label'] = [num_to_full_name_label_map[label] for label in overall_y_preds]
    submission_df.to_csv("../data/submission1.csv", index=False)


if __name__ == "__main__":
    main()