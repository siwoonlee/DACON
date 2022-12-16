import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch_optimizer as torch_optim
from torch import nn
from sklearn import metrics

from sentence_classification.utils.create_mask import (
    create_masked_titles,
)
from sentence_classification.utils.focal_loss import FocalLoss
from sentence_classification.utils.label_smoothing_loss import (
    LabelSmoothingLoss,
)
from sentence_classification.utils.normalization import (
    min_max_norm_title_vec,
)
from sentence_classification.utils.cosine_annealing_with_restarts import CosineAnnealingWarmupRestarts
from sentence_classification.constants import full_name_label_to_num_map


class Experiment(pl.LightningModule):
    def __init__(
        self,
        model,
        exp_config,
        train_dl,
        is_inference=False,
    ):
        super().__init__()
        self.model = model
        if is_inference:
            return
        self.experiment_name = exp_config.experiment_name
        self.optimizer = exp_config.optimizer
        self.softmax = nn.Softmax(dim=1)
        self.epoch_steps = train_dl.__len__()
        self.max_epochs = exp_config.max_epochs
        self.lr_sched = exp_config.lr_sched
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
            classes=len(full_name_label_to_num_map),
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

    def training_step(self, batch, batch_idx):
        x, 유형_labels, 극성_labels, 시제_labels, 확실성_labels, final_labels = (
            batch['inputs'],
            batch['유형_labels'],
            batch['극성_labels'],
            batch['시제_labels'],
            batch['확실성_labels'],
            batch['final_labels'],
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
        out_total, out_유형, out_극성, out_시제, out_확실성 = self.model(x)
        total_loss = self.loss(out_total, final_labels)
        유형_loss = self.loss(out_유형, 유형_labels)
        극성_loss = self.loss(out_극성, 극성_labels)
        시제_loss = self.loss(out_시제, 시제_labels)
        확실성_loss = self.loss(out_확실성, 확실성_labels)
        _, total_y_preds = torch.max(out_total, 1)
        _, 유형_y_preds = torch.max(out_유형, 1)
        _, 극성_y_preds = torch.max(out_극성, 1)
        _, 시제_y_preds = torch.max(out_시제, 1)
        _, 확실성_y_preds = torch.max(out_확실성, 1)
        loss = total_loss + 유형_loss + 극성_loss + 시제_loss + 확실성_loss
        _metrics = {
            'loss': loss,
            'train_total_loss_step': total_loss.item(),
            'train_final_labels': final_labels.cpu().detach().numpy(),
            'train_유형_labels': 유형_labels.cpu().detach().numpy(),
            'train_극성_labels': 극성_labels.cpu().detach().numpy(),
            'train_시제_labels': 시제_labels.cpu().detach().numpy(),
            'train_확실성_labels': 확실성_labels.cpu().detach().numpy(),
            'train_total_y_preds': total_y_preds.cpu().detach().numpy(),
            'train_유형_y_preds': 유형_y_preds.cpu().detach().numpy(),
            'train_극성_y_preds': 극성_y_preds.cpu().detach().numpy(),
            'train_시제_y_preds': 시제_y_preds.cpu().detach().numpy(),
            'train_확실성_y_preds': 확실성_y_preds.cpu().detach().numpy(),
        }
        return _metrics

    def validation_step(self, val_batch, batch_idx):
        x, 유형_labels, 극성_labels, 시제_labels, 확실성_labels, final_labels = (
            val_batch['inputs'],
            val_batch['유형_labels'],
            val_batch['극성_labels'],
            val_batch['시제_labels'],
            val_batch['확실성_labels'],
            val_batch['final_labels'],
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
        out_total, out_유형, out_극성, out_시제, out_확실성 = self.model(x)
        total_loss = self.loss(out_total, final_labels)
        유형_loss = self.loss(out_유형, 유형_labels)
        극성_loss = self.loss(out_극성, 극성_labels)
        시제_loss = self.loss(out_시제, 시제_labels)
        확실성_loss = self.loss(out_확실성, 확실성_labels)
        _, total_y_preds = torch.max(out_total, 1)
        _, 유형_y_preds = torch.max(out_유형, 1)
        _, 극성_y_preds = torch.max(out_극성, 1)
        _, 시제_y_preds = torch.max(out_시제, 1)
        _, 확실성_y_preds = torch.max(out_확실성, 1)
        loss = total_loss + 유형_loss + 극성_loss + 시제_loss + 확실성_loss
        _metrics = {
            'val_total_loss_step': total_loss.item(),
            'val_final_labels': final_labels.cpu().detach().numpy(),
            'val_유형_labels': 유형_labels.cpu().detach().numpy(),
            'val_극성_labels': 극성_labels.cpu().detach().numpy(),
            'val_시제_labels': 시제_labels.cpu().detach().numpy(),
            'val_확실성_labels': 확실성_labels.cpu().detach().numpy(),
            'val_total_y_preds': total_y_preds.cpu().detach().numpy(),
            'val_유형_y_preds': 유형_y_preds.cpu().detach().numpy(),
            'val_극성_y_preds': 극성_y_preds.cpu().detach().numpy(),
            'val_시제_y_preds': 시제_y_preds.cpu().detach().numpy(),
            'val_확실성_y_preds': 확실성_y_preds.cpu().detach().numpy(),
        }
        return _metrics

    def test_step(self, test_batch, batch_idx):
        x, 유형_labels, 극성_labels, 시제_labels, 확실성_labels, final_labels = (
            test_batch['inputs'],
            test_batch['유형_labels'],
            test_batch['극성_labels'],
            test_batch['시제_labels'],
            test_batch['확실성_labels'],
            test_batch['final_labels'],
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
        out_total, out_유형, out_극성, out_시제, out_확실성 = self.model(x)
        total_loss = self.loss(out_total, final_labels)
        유형_loss = self.loss(out_유형, 유형_labels)
        극성_loss = self.loss(out_극성, 극성_labels)
        시제_loss = self.loss(out_시제, 시제_labels)
        확실성_loss = self.loss(out_확실성, 확실성_labels)
        _, total_y_preds = torch.max(out_total, 1)
        _, 유형_y_preds = torch.max(out_유형, 1)
        _, 극성_y_preds = torch.max(out_극성, 1)
        _, 시제_y_preds = torch.max(out_시제, 1)
        _, 확실성_y_preds = torch.max(out_확실성, 1)
        loss = total_loss + 유형_loss + 극성_loss + 시제_loss + 확실성_loss
        _metrics = {
            'test_total_loss_step': total_loss.item(),
            'test_final_labels': final_labels.cpu().detach().numpy(),
            'test_유형_labels': 유형_labels.cpu().detach().numpy(),
            'test_극성_labels': 극성_labels.cpu().detach().numpy(),
            'test_시제_labels': 시제_labels.cpu().detach().numpy(),
            'test_확실성_labels': 확실성_labels.cpu().detach().numpy(),
            'test_total_y_preds': total_y_preds.cpu().detach().numpy(),
            'test_유형_y_preds': 유형_y_preds.cpu().detach().numpy(),
            'test_극성_y_preds': 극성_y_preds.cpu().detach().numpy(),
            'test_시제_y_preds': 시제_y_preds.cpu().detach().numpy(),
            'test_확실성_y_preds': 확실성_y_preds.cpu().detach().numpy(),
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
            _metrics = dict({f'{stage}/loss': avg_loss})
            self.log_dict(_metrics, rank_zero_only=True)
            return
        total_loss_steps = []
        final_labels = []
        total_y_preds = []
        for output in outputs:
            total_loss_steps.append(output[f'{stage}_total_loss_step'])
            final_labels.append(output[f'{stage}_final_labels'])
            total_y_preds.append(output[f'{stage}_total_y_preds'])
        total_avg_loss = np.mean(total_loss_steps)
        final_labels = np.concatenate(final_labels)
        total_y_preds = np.concatenate(total_y_preds)
        f1 = metrics.f1_score(final_labels, total_y_preds, average='weighted')
        b_acc = metrics.balanced_accuracy_score(final_labels, total_y_preds)
        acc = metrics.accuracy_score(final_labels, total_y_preds)
        _metrics = dict()
        _metrics[f'{stage}/loss'] = total_avg_loss
        _metrics[f'{stage}/weighted_f1_score'] = f1
        _metrics[f'{stage}/balanced_accuracy'] = b_acc
        _metrics[f'{stage}/accuracy'] = acc
        self.log_dict(_metrics)

    def training_epoch_end(self, outputs):
        self.common_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self.common_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self.common_epoch_end(outputs, 'test')
