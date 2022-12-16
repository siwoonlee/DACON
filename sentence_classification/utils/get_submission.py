import os
import importlib.util
from pathlib import Path
import pandas as pd

from sentence_classification.experiment import Experiment
from sentence_classification.models.model import load_model_from_config
from sentence_classification.constants import num_to_full_name_label_map


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
):
    for i in range(0, len(test_data_df), batch_size):
        sample_list = []
        for idx in range(i, i+batch_size, 1):
            문장 = test_data_df['문장'].iloc[idx]
            문장_vec = get_padded_character_vector(
                문장,
                max_char_length,
            )
            문장_vec_inputs = torch.tensor(문장_vec, dtype=torch.int64)
            sample_list.append(문장_vec_inputs)
        yield sample_list


def get_inference_result(
    pl_model,
    test_data_df,
    batch_size=4,
    max_char_length=300,
):
    overall_y_preds = []
    for sample_list in get_inference_samples(
        test_data_df,
        batch_size=4,
        max_char_length=300,
    ):
        for batch in sample_list:
            logits = pl_model.model(batch)
            _, y_preds = torch.max(logits, 1)
            overall_y_preds.extend(y_preds.cpu().detach().numpy().tolist())
    return overall_y_preds


def main():
    path_to_model_ckpt = "/Users/swlee/Downloads/models/20220915_050623/model_epoch=32_step=    267762.ckpt"
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