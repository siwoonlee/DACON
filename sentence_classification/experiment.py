import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sn
import torch
import torch_optimizer as torch_optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch import nn

from laboratory.evaluation.custom_eval import (
    get_model_eval_metric_values,
    get_multiclass_eval_dict,
    merge_child_parent_result,
    update_leaderboard,
)
from laboratory.transaction_classification.utils.create_mask import (
    create_masked_titles,
)
from laboratory.transaction_classification.utils.focal_loss import FocalLoss
from laboratory.transaction_classification.utils.label_smoothing_loss import (
    LabelSmoothingLoss,
)
from laboratory.transaction_classification.utils.normalization import (
    min_max_norm_title_vec,
)


class Experiment(pl.LightningModule):
    def __init__(
        self,
        model,
        num_child_class,
        num_parent_class,
        exp_config,
        train_dl,
        is_inference=False,
    ):
        super().__init__()
        self.model = model
        if is_inference:
            return
        self.experiment_name = exp_config.experiment_name
        self.use_implicit_parent_category = (
            exp_config.use_implicit_parent_category
        )
        self.path_to_leaderboard = (
            f"{exp_config.path_to_leaderboard}/leaderboard.pkl"
        )
        self.optimizer = exp_config.optimizer
        self.softmax = nn.Softmax(dim=1)
        self.epoch_steps = train_dl.__len__()
        self.max_epochs = exp_config.max_epochs
        self.lr_sched = exp_config.lr_sched
        self.num_child_class = num_child_class
        self.num_parent_class = num_parent_class
        self.default_category_df = train_dl.dataset.default_category_df
        self.use_child_parent_loss = exp_config.use_child_parent_loss
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.lr_sched == 'cosine_anneal':
            self.learning_rate = exp_config.cos_min_lr
            self.max_learning_rate = exp_config.cos_max_lr
        else:
            self.learning_rate = exp_config.lr
            self.max_learning_rate = exp_config.max_lr
        if exp_config.loss == "weightedCE":
            labels = train_dl.dataset.dataset_df['child_label'].values
            (unique, counts) = np.unique(labels, return_counts=True)
            cls_weight = 1 / counts
            cls_weight = cls_weight / sum(cls_weight)
            cls_weight = torch.tensor(cls_weight).to(
                device=device, dtype=torch.float32
            )
        else:
            cls_weight = None
        self.loss = LabelSmoothingLoss(
            classes=num_child_class,
            smoothing=exp_config.smoothing_factor,
            weight=cls_weight,
        )
        if "Focal" == exp_config.loss:
            self.loss = FocalLoss(
                gamma=exp_config.focal_gamma, alpha=exp_config.focal_alpha
            )
        self.is_pretraining = exp_config.is_pretraining
        if self.is_pretraining:
            self.loss = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate
            )
        elif self.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.learning_rate
            )
        elif self.optimizer == 'RAdam':
            optimizer = torch_optim.RAdam(
                self.parameters(), lr=self.learning_rate
            )
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=0.9
            )
        elif self.optimizer == 'QHAdam':
            optimizer = torch_optim.QHAdam(
                self.parameters(), lr=self.learning_rate
            )
        elif self.optimizer == 'RMSProp':
            optimizer = torch.optim.RMSprop(
                self.parameters(), lr=self.learning_rate
            )
        else:
            raise Exception("Not Available Optimizer!")
        if self.lr_sched == 'one_cycle':
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    max_lr=self.max_learning_rate,
                    optimizer=optimizer,
                    epochs=self.max_epochs,
                    total_steps=self.epoch_steps * self.max_epochs,
                    pct_start=0.3,
                    div_factor=25,
                    final_div_factor=1e4,
                    base_momentum=0.90,
                    max_momentum=0.95,
                ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1,
            }
        elif self.lr_sched == 'step':
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.max_epochs // 3, gamma=0.1
                ),
                'name': 'learning_rate',
                'interval': 'epoch',
                'frequency': 1,
            }
        elif self.lr_sched == 'cosine_anneal':
            lr_scheduler = {
                'scheduler': CosineAnnealingWarmupRestarts(
                    optimizer,
                    first_cycle_steps=self.epoch_steps,
                    cycle_mult=2,
                    max_lr=self.max_learning_rate,
                    min_lr=self.learning_rate,
                    warmup_steps=self.epoch_steps // 10,
                    gamma=0.5,
                ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1,
            }
        else:
            return optimizer
        return [optimizer], [lr_scheduler]

    def get_confusion_mat_img(self, confusion_mat, to_CWH=True):
        df_cm = pd.DataFrame(
            confusion_mat,
            index=[i for i in range(self.num_child_classes)],
            columns=[i for i in range(self.num_child_classes)],
        )
        fig, ax = plt.subplots(figsize=(10, 10), dpi=70)
        sn.heatmap(df_cm, annot=True, fmt='d')
        plt.xlabel("Prediction")
        plt.ylabel("Ground Truth")
        fig.canvas.draw()
        image_from_plot = np.frombuffer(
            fig.canvas.tostring_rgb(), dtype=np.uint8
        )
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        if to_CWH:
            image_from_plot = np.swapaxes(image_from_plot, 2, 0)
            image_from_plot = np.swapaxes(image_from_plot, 2, 1)
        return image_from_plot

    def make_implicit_parent_category_predictions(self, child_logits):
        parent_logits = []
        for parent_label in range(
            0, self.default_category_df['parent_label'].max() + 1
        ):
            target_child_idxs = self.default_category_df['child_label'][
                self.default_category_df['parent_label'] == parent_label
            ]
            parent_logits.append(
                torch.sum(
                    child_logits[:, target_child_idxs.values], axis=1
                ).unsqueeze(dim=1)
            )
        parent_logits = torch.cat(parent_logits, axis=1)
        return parent_logits

    def training_step(self, batch, batch_idx):
        x, child_y, parent_y = (
            batch['inputs'],
            batch['child_labels'],
            batch['parent_labels'],
        )
        if self.is_pretraining:
            mask = create_masked_titles(x)
            _x = x - x * mask
            y_pred = self.model(_x)
            loss = self.loss(
                min_max_norm_title_vec(y_pred),
                min_max_norm_title_vec(x.to(torch.float32)),
            )
            _metrics = {'loss': loss, 'train_loss_step': loss.item()}
            return _metrics
        if self.use_implicit_parent_category:
            child_logits = self.model(x)
            parent_logits = self.make_implicit_parent_category_predictions(
                child_logits
            )
        else:
            child_logits, parent_logits = self.model(x)
        child_loss = self.loss(child_logits, child_y)
        parent_loss = self.loss(parent_logits, parent_y)
        _, child_y_preds = torch.max(child_logits, 1)
        _, parent_y_preds = torch.max(parent_logits, 1)
        loss = (
            child_loss + parent_loss
            if self.use_child_parent_loss
            else child_loss
        )
        _metrics = {
            'loss': loss,
            'train_child_loss_step': child_loss.item(),
            'train_parent_loss_step': parent_loss.item(),
            'train_child_y_gts': child_y.cpu().detach().numpy(),
            'train_parent_y_gts': parent_y.cpu().detach().numpy(),
            'train_child_y_preds': child_y_preds.cpu().detach().numpy(),
            'train_parent_y_preds': parent_y_preds.cpu().detach().numpy(),
        }
        return _metrics

    def validation_step(self, val_batch, batch_idx):
        x, child_y, parent_y = (
            val_batch['inputs'],
            val_batch['child_labels'],
            val_batch['parent_labels'],
        )
        if self.is_pretraining:
            mask = create_masked_titles(x)
            _x = x - x * mask
            y_pred = self.model(_x)
            loss = self.loss(
                min_max_norm_title_vec(y_pred),
                min_max_norm_title_vec(x.to(torch.float32)),
            )
            _metrics = {'loss': loss, 'val_loss_step': loss.item()}
            return _metrics
        if self.use_implicit_parent_category:
            child_logits = self.model(x)
            parent_logits = self.make_implicit_parent_category_predictions(
                child_logits
            )
        else:
            child_logits, parent_logits = self.model(x)
        child_loss = self.loss(child_logits, child_y)
        parent_loss = self.loss(parent_logits, parent_y)
        _, child_y_preds = torch.max(child_logits, 1)
        _, parent_y_preds = torch.max(parent_logits, 1)
        loss = (
            child_loss + parent_loss
            if self.use_child_parent_loss
            else child_loss
        )
        _metrics = {
            'loss': loss,
            'val_child_loss_step': child_loss.item(),
            'val_parent_loss_step': parent_loss.item(),
            'val_child_y_gts': child_y.cpu().detach().numpy(),
            'val_parent_y_gts': parent_y.cpu().detach().numpy(),
            'val_child_y_preds': child_y_preds.cpu().detach().numpy(),
            'val_parent_y_preds': parent_y_preds.cpu().detach().numpy(),
        }
        return _metrics

    def test_step(self, test_batch, batch_idx):
        x, child_y, parent_y = (
            test_batch['inputs'],
            test_batch['child_labels'],
            test_batch['parent_labels'],
        )
        if self.is_pretraining:
            mask = create_masked_titles(x)
            _x = x - x * mask
            y_pred = self.model(_x)
            loss = self.loss(
                min_max_norm_title_vec(y_pred),
                min_max_norm_title_vec(x.to(torch.float32)),
            )
            _metrics = {'loss': loss, 'test_loss_step': loss.item()}
            return _metrics
        if self.use_implicit_parent_category:
            child_logits = self.model(x)
            parent_logits = self.make_implicit_parent_category_predictions(
                child_logits
            )
        else:
            child_logits, parent_logits = self.model(x)
        child_loss = self.loss(child_logits, child_y)
        parent_loss = self.loss(parent_logits, parent_y)
        _, child_y_preds = torch.max(child_logits, 1)
        _, parent_y_preds = torch.max(parent_logits, 1)
        loss = (
            child_loss + parent_loss
            if self.use_child_parent_loss
            else child_loss
        )
        _metrics = {
            'loss': loss,
            'test_child_loss_step': child_loss.item(),
            'test_parent_loss_step': parent_loss.item(),
            'test_child_y_gts': child_y.cpu().detach().numpy(),
            'test_parent_y_gts': parent_y.cpu().detach().numpy(),
            'test_child_y_preds': child_y_preds.cpu().detach().numpy(),
            'test_parent_y_preds': parent_y_preds.cpu().detach().numpy(),
        }
        return _metrics

    def common_epoch_end(self, outputs, stage):
        if self.trainer.sanity_checking:
            return
        elif self.is_pretraining:
            loss_steps = []
            for output in outputs:
                loss_steps.append(output[f'{stage}_loss_step'])
            avg_loss = np.mean(loss_steps)
            metrics = dict({f'{stage}/loss': avg_loss})
            self.log_dict(metrics, rank_zero_only=True)
            return
        child_loss_steps = []
        parent_loss_steps = []
        child_y_gts = []
        parent_y_gts = []
        child_y_preds = []
        parent_y_preds = []
        for output in outputs:
            child_loss_steps.append(output[f'{stage}_child_loss_step'])
            parent_loss_steps.append(output[f'{stage}_parent_loss_step'])
            child_y_gts.append(output[f'{stage}_child_y_gts'])
            parent_y_gts.append(output[f'{stage}_parent_y_gts'])
            child_y_preds.append(output[f'{stage}_child_y_preds'])
            parent_y_preds.append(output[f'{stage}_parent_y_preds'])
        child_avg_loss = np.mean(child_loss_steps)
        parent_avg_loss = np.mean(parent_loss_steps)
        child_y_gts = np.concatenate(child_y_gts)
        parent_y_gts = np.concatenate(parent_y_gts)
        child_y_preds = np.concatenate(child_y_preds)
        parent_y_preds = np.concatenate(parent_y_preds)
        child_eval_result = get_multiclass_eval_dict(
            self.num_child_class, child_y_gts, child_y_preds
        )
        parent_eval_result = get_multiclass_eval_dict(
            self.num_parent_class, parent_y_gts, parent_y_preds
        )
        overall_eval_result = merge_child_parent_result(
            child_eval_result, parent_eval_result
        )
        metrics = dict(
            {f'{stage}/{key}': val for key, val in overall_eval_result.items()}
        )
        metrics[f'{stage}/child_loss'] = child_avg_loss
        metrics[f'{stage}/parent_loss'] = parent_avg_loss
        if stage == 'test':
            model_eval_res_df = get_model_eval_metric_values(
                self.experiment_name,
                self.num_child_class,
                self.num_parent_class,
                child_y_gts,
                child_y_preds,
                parent_y_gts,
                parent_y_preds,
            )
            try:
                leaderboard = pd.read_pickle(self.path_to_leaderboard)
            except:
                leaderboard = pd.DataFrame(
                    data=[[0 for _ in model_eval_res_df.columns]],
                    columns=model_eval_res_df.columns,
                    index=["Dummy Model"],
                )
            leaderboard = update_leaderboard(model_eval_res_df, leaderboard)
            leaderboard.to_pickle(self.path_to_leaderboard)
        self.log_dict(metrics)

    def training_epoch_end(self, outputs):
        self.common_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.common_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self.common_epoch_end(outputs, 'test')

    def predict_step(self, batch):  # need to work on it
        x = self._prepare_batch(batch)
        return self(x)


class Experiment2(Experiment):
    def __init__(
        self,
        model,
        num_child_class,
        num_parent_class,
        exp_config,
        train_dl,
        is_inference=False,
    ):
        super().__init__(
            model,
            num_child_class,
            num_parent_class,
            exp_config,
            train_dl,
            is_inference=is_inference,
        )
        self.lambda_sparse = 1e-3  # for TabNet training

    def training_step(self, batch, batch_idx):
        child_y = batch.pop('child_labels')
        parent_y = batch.pop('parent_labels')
        x = batch
        if self.use_implicit_parent_category:
            child_logits, M_loss = self.model(x)
            parent_logits = self.make_implicit_parent_category_predictions(
                child_logits
            )
        else:
            child_logits, parent_logits = self.model(x)
        child_loss = self.loss(child_logits, child_y)
        parent_loss = self.loss(parent_logits, parent_y)
        _, child_y_preds = torch.max(child_logits, 1)
        _, parent_y_preds = torch.max(parent_logits, 1)
        loss = (
            child_loss + parent_loss
            if self.use_child_parent_loss
            else child_loss
        )
        loss -= self.lambda_sparse * M_loss
        _metrics = {
            'loss': loss,
            'train_child_loss_step': child_loss.item(),
            'train_parent_loss_step': parent_loss.item(),
            'train_child_y_gts': child_y.cpu().detach().numpy(),
            'train_parent_y_gts': parent_y.cpu().detach().numpy(),
            'train_child_y_preds': child_y_preds.cpu().detach().numpy(),
            'train_parent_y_preds': parent_y_preds.cpu().detach().numpy(),
        }
        return _metrics

    def validation_step(self, val_batch, batch_idx):
        child_y = val_batch.pop('child_labels')
        parent_y = val_batch.pop('parent_labels')
        x = val_batch
        if self.use_implicit_parent_category:
            child_logits, M_loss = self.model(x)
            parent_logits = self.make_implicit_parent_category_predictions(
                child_logits
            )
        else:
            child_logits, parent_logits = self.model(x)
        child_loss = self.loss(child_logits, child_y)
        parent_loss = self.loss(parent_logits, parent_y)
        _, child_y_preds = torch.max(child_logits, 1)
        _, parent_y_preds = torch.max(parent_logits, 1)
        loss = (
            child_loss + parent_loss
            if self.use_child_parent_loss
            else child_loss
        )
        loss -= self.lambda_sparse * M_loss
        _metrics = {
            'loss': loss,
            'val_child_loss_step': child_loss.item(),
            'val_parent_loss_step': parent_loss.item(),
            'val_child_y_gts': child_y.cpu().detach().numpy(),
            'val_parent_y_gts': parent_y.cpu().detach().numpy(),
            'val_child_y_preds': child_y_preds.cpu().detach().numpy(),
            'val_parent_y_preds': parent_y_preds.cpu().detach().numpy(),
        }
        return _metrics

    def test_step(self, test_batch, batch_idx):
        child_y = test_batch.pop('child_labels')
        parent_y = test_batch.pop('parent_labels')
        x = test_batch
        if self.use_implicit_parent_category:
            child_logits, M_loss = self.model(x)
            parent_logits = self.make_implicit_parent_category_predictions(
                child_logits
            )
        else:
            child_logits, parent_logits = self.model(x)
        child_loss = self.loss(child_logits, child_y)
        parent_loss = self.loss(parent_logits, parent_y)
        _, child_y_preds = torch.max(child_logits, 1)
        _, parent_y_preds = torch.max(parent_logits, 1)
        loss = (
            child_loss + parent_loss
            if self.use_child_parent_loss
            else child_loss
        )
        loss -= self.lambda_sparse * M_loss
        _metrics = {
            'loss': loss,
            'test_child_loss_step': child_loss.item(),
            'test_parent_loss_step': parent_loss.item(),
            'test_child_y_gts': child_y.cpu().detach().numpy(),
            'test_parent_y_gts': parent_y.cpu().detach().numpy(),
            'test_child_y_preds': child_y_preds.cpu().detach().numpy(),
            'test_parent_y_preds': parent_y_preds.cpu().detach().numpy(),
        }
        return _metrics
