import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime
import torch
from mlflow.models.signature import ModelSignature
from mlflow.onnx import save_model
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from onnx import load_model as load_onnx_model

from laboratory.transaction_classification.constants import (
    CHILD_LABEL_NUM,
    PARENT_LABEL_NUM,
)
from laboratory.transaction_classification.experiment import (
    Experiment,
    Experiment2,
)
from laboratory.transaction_classification.models.model import (
    load_model_from_config,
)
from laboratory.transaction_classification.utils.vectorizer import (
    get_padded_character_vector,
)


def get_config_args(path_to_config):
    sys.path.insert(0, path_to_config)
    from config import Config

    config_args = Config()
    return config_args


def load_model(path_to_model_ckpt, config_args, device):
    model = load_model_from_config(config_args, is_inference=True)
    exp_cls = (
        Experiment2
        if config_args.experiment_config.multi_input
        else Experiment
    )
    pl_model = exp_cls.load_from_checkpoint(
        model=model,
        num_child_class=CHILD_LABEL_NUM,
        num_parent_class=PARENT_LABEL_NUM,
        exp_config=None,
        train_dl=None,
        is_inference=True,
        checkpoint_path=path_to_model_ckpt,
        map_location=device,
    )
    pl_model.eval()
    pl_model.to(device=device, dtype=torch.float32)
    return pl_model


def onnx_to_mlflow(path_to_onnx_model, path_to_mlflow_model):
    onnx_model = load_onnx_model(path_to_onnx_model)
    signature = get_mlflow_signature()
    save_model(onnx_model, path_to_mlflow_model, signature=signature)


def get_mlflow_signature():
    input_schema = Schema(
        [
            TensorSpec(
                type=np.dtype(np.int64), shape=(-1, 75), name='title_vec'
            ),
            TensorSpec(
                type=np.dtype(np.float32),
                shape=(-1, 18),
                name='transaction_amount_vec',
            ),
        ]
    )
    output_schema = Schema(
        [
            TensorSpec(
                type=np.dtype(np.int64), shape=(-1, 1), name='category_idx'
            )
        ]
    )
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    return signature


def simple_test_onnx(path_to_onnx, torch_model, input_sample):
    session = onnxruntime.InferenceSession(path_to_onnx)
    input_sample_dict = {
        input_layer.name: v.tolist()
        for (k, v), input_layer in zip(
            input_sample.items(), session.get_inputs()
        )
    }
    onnx_child_y_preds = session.run(None, input_sample_dict)[0]
    print(f'ONNX category_idx : {onnx_child_y_preds}')
    torch_child_y_preds = torch_model(input_sample)
    print(f'torch category_idx : {torch_child_y_preds}')
    assert onnx_child_y_preds.item() == torch_child_y_preds.item()


def main():
    path_to_model_ckpt = "/Users/swlee/Downloads/models/20220915_050623/model_epoch=32_step=    267762.ckpt"
    path_to_config = str(Path(path_to_model_ckpt).parent)
    device = torch.device("cpu")
    config_args = get_config_args(path_to_config)
    pl_model = load_model(path_to_model_ckpt, config_args, device)
    to_be_vectorized = "Sample Title"
    title_vec_inputs = get_padded_character_vector(
        to_be_vectorized, config_args.model_config.seq_len
    )
    title_vec_inputs = np.array(title_vec_inputs, dtype=np.int64)
    title_vec_inputs = torch.from_numpy(title_vec_inputs).unsqueeze(dim=0)
    tabular_vec_inputs = torch.randn(
        (1, config_args.tabular_model_config.c_in)
    )
    input_sample = dict(
        title_vec_inputs=title_vec_inputs,
        transaction_amount_inputs=tabular_vec_inputs,
    )
    pl_model.to_onnx(
        path_to_model_ckpt.replace("ckpt", "onnx"),
        input_sample,
        input_names=['title_vec', 'transaction_amount_vec'],
        output_names=['category_idx'],
        example_outputs=pl_model.model(input_sample),
        opset_version=11,
        export_params=True,
        do_constant_folding=True,
        strip_doc_string=True,
        dynamic_axes={
            'title_vec': {0: 'batch_size'},
            'transaction_amount_vec': {0: 'batch_size'},
            'category_idx': {0: 'batch_size'},
        },
    )
    simple_test_onnx(
        path_to_onnx=path_to_model_ckpt.replace("ckpt", "onnx"),
        torch_model=pl_model,
        input_sample=input_sample,
    )
    onnx_to_mlflow(
        path_to_onnx_model=path_to_model_ckpt.replace("ckpt", "onnx"),
        path_to_mlflow_model=os.path.join(path_to_config, 'mlflow_model'),
    )

    model = mlflow.pyfunc.load_model(
        os.path.join(path_to_config, 'mlflow_model')
    )
    print(model._model_meta._signature)


if __name__ == "__main__":
    main()
