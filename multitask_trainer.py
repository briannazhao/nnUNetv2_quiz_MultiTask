import os, math, json
import numpy as np 
import pandas as pd
from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import autocast
from importlib import import_module

from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.residual import BasicBlockD

# Network with ROI-masked GeM pooling for cls head (NaN-safe)
class MultiTaskResEnc(nn.Module):
    """Shared encoder + Segmentation decoder + Classification head"""

    @property
    def decoder(self):
        return self.segmentation_net.decoder

    def __init__(self, input_channels, n_stages, features_per_stage, conv_op,
                 kernel_sizes, strides, n_blocks_per_stage, num_segmentation_classes,
                 num_classification_classes=3, n_conv_per_stage_decoder=None,
                 conv_bias=False, norm_op=None, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None, nonlin=None,
                 nonlin_kwargs=None, deep_supervision=False, block=None,
                 cls_stopgrad_through_encoder=True):
        super().__init__()
        block = block or BasicBlockD

        self.segmentation_net = ResidualEncoderUNet(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_segmentation_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=deep_supervision,
            block=block
        )

        bottleneck_features = features_per_stage[-1] if isinstance(features_per_stage, (list, tuple)) else features_per_stage
        self.cls_stopgrad_through_encoder = cls_stopgrad_through_encoder

        # GeM pooling exponent for masked pooling
        self.gem_p = nn.Parameter(torch.tensor(3.0))

        # Fallback GAP (3D)
        self.global_pool = nn.AdaptiveAvgPool3d(1) if conv_op == nn.Conv3d else nn.AdaptiveAvgPool2d(1)

        self.classification_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(bottleneck_features, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, num_classification_classes)
        )

    @staticmethod
    def _nan_to_num(x, lim=1e6):
        # replace NaN/Inf and clamp to avoid overflow in downstream losses
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.clamp(min=-lim, max=lim)

    def _masked_gem_pool(self, bottleneck, seg_logits, eps=1e-6, tiny_roi_frac=5e-4):
        """
        bottleneck: [B, C, d, h, w]
        seg_logits: [B, Cseg, D, H, W] (main head resolution)
        Returns pooled: [B, C, 1, 1, 1]
        """
        seg_logits = self._nan_to_num(seg_logits)
        # soft foreground prob
        if seg_logits.shape[1] > 1:
            fg_prob = torch.softmax(seg_logits, dim=1)[:, 1:, ...].max(1, keepdim=True)[0]
        else:
            fg_prob = torch.sigmoid(seg_logits)

        fg_prob = self._nan_to_num(fg_prob).clamp(0, 1)

        # downsample mask to bottleneck size
        if fg_prob.shape[2:] != bottleneck.shape[2:]:
            fg_prob = F.adaptive_avg_pool3d(fg_prob, bottleneck.shape[2:])
            fg_prob = self._nan_to_num(fg_prob).clamp(0, 1)

        # safe dtype math in fp32, cast back at end
        orig_dtype = bottleneck.dtype
        x = self._nan_to_num(bottleneck.float())
        m = fg_prob.float()

        # tiny ROI -> fall back to GAP
        valid = m.sum(dim=(2, 3, 4), keepdim=True)
        total = float(np.prod(bottleneck.shape[2:]))
        tiny = (valid / (total + eps)) < tiny_roi_frac  # [B,1,1,1,1] boolean

        # GeM over ROI
        p = torch.clamp(self.gem_p, 1.0, 6.0)
        x_pos = x.clamp_min(0)
        num = ((x_pos ** p) * m).sum(dim=(2, 3, 4), keepdim=True)
        den = valid.clamp_min(1.0)
        gem_roi = (num / den).clamp_min(eps) ** (1.0 / p)

        # GAP fallback
        gap = F.adaptive_avg_pool3d(x, 1)

        pooled = torch.where(tiny, gap, gem_roi).to(orig_dtype)
        return self._nan_to_num(pooled)

    def forward(self, x, return_both: bool = False):
        seg_logits = self.segmentation_net(x)
        seg_logits_main = seg_logits[0] if isinstance(seg_logits, (list, tuple)) else seg_logits
        seg_logits_main = self._nan_to_num(seg_logits_main)

        # encoder features
        enc_out = self.segmentation_net.encoder(x)
        bottleneck = enc_out[-1] if isinstance(enc_out, (list, tuple)) else enc_out

        # stop-grad for cls path (optional)
        bn_for_cls = bottleneck.detach() if self.cls_stopgrad_through_encoder else bottleneck

        pooled = self._masked_gem_pool(bn_for_cls, seg_logits_main)  # [B,C,1,1,1]
        cls_logits = self.classification_head(pooled.view(pooled.size(0), -1))
        cls_logits = self._nan_to_num(cls_logits)

        if return_both or self.training:
            return seg_logits, cls_logits

        # Predictor expects segmentation only
        return seg_logits

class MultiTaskTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, 
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # classification weight ramps up
        self.cls_weight_start = 0.10 
        self.cls_weight_end = 0.55
        self.cls_weight_ramp_epochs = 25  # linear ramp

        # classification loss knobs
        self.cls_label_smoothing = 0.05
        self.cls_logit_adj_tau = 1.0

        # mapping & priors
        self.case_to_subtype = self._load_classification_labels()
        print(f"Loaded {len(self.case_to_subtype)} classification labels")

        counts = np.bincount(list(self.case_to_subtype.values()), minlength=3)
        priors = counts / max(1, counts.sum())
        inv = 1.0 / np.clip(counts, 1, None)
        inv = inv / inv.mean() if inv.sum() > 0 else np.ones_like(inv, dtype=float)

        self.cls_class_weights = torch.tensor(inv, dtype=torch.float32, device=self.device)
        self.cls_log_prior = torch.log(torch.tensor(np.clip(priors, 1e-8, 1.0), dtype=torch.float32, device=self.device))

        self.classification_loss = nn.CrossEntropyLoss(
            weight=self.cls_class_weights, 
            label_smoothing=self.cls_label_smoothing
        )

        # keep DS off: we feed only the main seg map to the loss
        self.enable_deep_supervision = False

        # bookkeeping
        self.classification_metrics = {'train_acc': [], 'val_acc': []}
        self._epoch_idx = -1
        self._val_cls_acc_epoch = []
        self.best_val_cls_acc = -1.0
        self.best_cls_head_path = os.path.join(self.output_folder, "checkpoint_best_cls_head.pth")

        # stability: warm-up in fp32 for a few steps to avoid AMP NaNs at start
        self._train_step_count = 0
        self._fp32_warmup_steps = 6

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                arch_init_kwargs: dict,  
                                arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                num_input_channels: int,
                                num_output_channels: int,
                                enable_deep_supervision: bool = True) -> nn.Module:
        kwargs = dict(arch_init_kwargs)
        if arch_init_kwargs_req_import:
            for k in arch_init_kwargs_req_import:
                v = kwargs.get(k)
                if isinstance(v, str):
                    mod, attr = v.rsplit('.', 1)
                    kwargs[k] = getattr(import_module(mod), attr)

        return MultiTaskResEnc(
            input_channels=num_input_channels,
            n_stages=kwargs['n_stages'],
            features_per_stage=kwargs['features_per_stage'],
            conv_op=kwargs['conv_op'],
            kernel_sizes=kwargs['kernel_sizes'],
            strides=kwargs['strides'],
            n_blocks_per_stage=kwargs['n_blocks_per_stage'],
            num_segmentation_classes=num_output_channels,
            num_classification_classes=3,
            n_conv_per_stage_decoder=kwargs['n_conv_per_stage_decoder'],
            conv_bias=kwargs['conv_bias'],
            norm_op=kwargs['norm_op'],
            norm_op_kwargs=kwargs['norm_op_kwargs'],
            dropout_op=kwargs.get('dropout_op'),
            dropout_op_kwargs=kwargs.get('dropout_op_kwargs'),
            nonlin=kwargs['nonlin'],
            nonlin_kwargs=kwargs['nonlin_kwargs'],
            deep_supervision=enable_deep_supervision,
            cls_stopgrad_through_encoder=True
        )

    def _load_classification_labels(self):
        try:
            df = pd.read_csv('/content/case_subtype_mapping.csv')
            col = 'case_id' if 'case_id' in df.columns else ('case' if 'case' in df.columns else None)
            assert col is not None, f"Mapping CSV must have case_id or case; got columns {df.columns.tolist()}"
            df[col] = df[col].astype(str)
            df['subtype'] = df['subtype'].astype(int)
            return dict(zip(df[col], df['subtype']))
        except Exception as e:
            print("Warning: Could not load case_subtype_mapping.csv; falling back to dummy mapping.", e)
            return {f"case_{i:04d}": i % 3 for i in range(300)}

    def get_network(self):
        return self.build_network_architecture(
            architecture_class_name=self.network_arch_class_name,
            arch_init_kwargs=self.network_arch_init_kwargs,
            arch_init_kwargs_req_import=self.network_arch_init_kwargs_req_import,
            num_input_channels=self.num_input_channels,
            num_output_channels=self.label_manager.num_segmentation_heads,
            enable_deep_supervision=self.enable_deep_supervision
        )

    def on_train_start(self):
        super().on_train_start()
        # ensure tensors are on the right device
        self.cls_class_weights = self.cls_class_weights.to(self.device)
        self.cls_log_prior = self.cls_log_prior.to(self.device)
        if isinstance(getattr(self.classification_loss, "weight", None), torch.Tensor):
            self.classification_loss.weight = self.cls_class_weights.to(self.device)

    def on_epoch_start(self):
        super().on_epoch_start()
        self._epoch_idx += 1
        self._val_cls_acc_epoch = []

    # ---------- helpers ----------
    @staticmethod
    def _nan_to_num(x, lim=1e6):
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.clamp(min=-lim, max=lim)

    def _align_target_to_output(self, target, seg_logits: torch.Tensor) -> torch.Tensor:
        t = target[0] if isinstance(target, (list, tuple)) else target
        if not torch.is_tensor(t):
            t = torch.as_tensor(t)
        t = t.to(device=seg_logits.device, non_blocking=True)

        # squeeze extra singletons
        while t.ndim > seg_logits.ndim:
            reduced = False
            for dim in (1, 2, 0):
                if dim < t.ndim and t.shape[dim] == 1:
                    t = t.squeeze(dim); reduced = True; break
            if not reduced:
                for dim in range(t.ndim):
                    if t.shape[dim] == 1:
                        t = t.squeeze(dim); reduced = True; break
            if not reduced:
                break

        # add channel if missing
        if t.ndim == seg_logits.ndim - 1:
            t = t.unsqueeze(1)

        # clamp labels into valid range (defensive)
        num_classes = seg_logits.shape[1]
        t = t.long().clamp(min=0, max=max(0, num_classes - 1))

        if t.ndim != seg_logits.ndim:
            raise RuntimeError(f"Target rank {t.ndim} mismatches output rank {seg_logits.ndim}. "
                               f"target {tuple(t.shape)} vs output {tuple(seg_logits.shape)}")
        return t

    def _ensure_bc_first(self, seg_logits: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        if seg_logits.ndim >= 2:
            B_data = data.shape[0]
            if seg_logits.shape[0] != B_data and seg_logits.shape[1] == B_data:
                perm = list(range(seg_logits.ndim))
                perm[0], perm[1] = 1, 0
                seg_logits = seg_logits.permute(*perm).contiguous()
        return seg_logits

    def _cls_targets_and_mask(self, batch):
        """
        Returns:
          cls_target: LongTensor [B] with -1 where the label is unknown
          mask:       BoolTensor [B] True where label is known
        """
        bs = batch['data'].shape[0]
        if 'keys' in batch:
            keys = list(batch['keys'])
            t = []
            for k in keys:
                t.append(self.case_to_subtype[k] if k in self.case_to_subtype else -1)
            tgt = torch.as_tensor(t, dtype=torch.long, device=self.device)
        else:
            tgt = torch.full((bs,), -1, dtype=torch.long, device=self.device)
        mask = tgt >= 0
        return tgt, mask

    def _cls_weight_schedule(self) -> float:
        e = max(0, self._epoch_idx)
        if self.cls_weight_ramp_epochs <= 0:
            return float(self.cls_weight_end)
        frac = min(1.0, e / float(self.cls_weight_ramp_epochs))
        return float(self.cls_weight_start + frac * (self.cls_weight_end - self.cls_weight_start))

    # ---------- train/val ----------
    def train_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # FP32 warmup for first few steps (avoid AMP NaNs at start)
        use_amp = (self.device.type == 'cuda') and (self._train_step_count >= self._fp32_warmup_steps)
        ctx = autocast(self.device.type, enabled=use_amp) if use_amp else dummy_context()

        with ctx:
            seg_out, cls_out = self.network(data, return_both=True)
            seg_logits = seg_out[0] if isinstance(seg_out, (list, tuple)) else seg_out
            seg_logits = self._nan_to_num(self._ensure_bc_first(seg_logits, data))
            target = self._align_target_to_output(batch['target'], seg_logits)

            # sanitize cls logits too
            cls_out = self._nan_to_num(cls_out)

            seg_loss = self.loss(seg_logits, target)

            # classification (masked) + logit adjustment
            cls_target, cls_mask = self._cls_targets_and_mask(batch)
            if cls_mask.any():
                cls_logits_adj = cls_out - (self.cls_log_prior.to(cls_out.dtype))[None, :] * self.cls_logit_adj_tau
                cls_logits_adj = self._nan_to_num(cls_logits_adj)
                cls_loss = self.classification_loss(cls_logits_adj[cls_mask], cls_target[cls_mask])
                cls_acc  = (torch.argmax(cls_logits_adj[cls_mask], dim=1) == cls_target[cls_mask]).float().mean().item()
            else:
                cls_loss = torch.zeros((), device=self.device)
                cls_acc  = float('nan')

            total_loss = seg_loss + self._cls_weight_schedule() * cls_loss

        # if NaN/Inf slipped through, try a last-chance FP32 recompute
        if not torch.isfinite(total_loss):
            print("⚠️ Non-finite loss encountered. Retrying batch in FP32 fall-back.")
            with dummy_context():
                seg_out, cls_out = self.network(data, return_both=True)
                seg_logits = seg_out[0] if isinstance(seg_out, (list, tuple)) else seg_out
                seg_logits = self._nan_to_num(self._ensure_bc_first(seg_logits, data).float())
                target = self._align_target_to_output(batch['target'], seg_logits)
                cls_out = self._nan_to_num(cls_out.float())

                seg_loss = self.loss(seg_logits, target)
                cls_target, cls_mask = self._cls_targets_and_mask(batch)
                if cls_mask.any():
                    cls_logits_adj = self._nan_to_num(cls_out - (self.cls_log_prior.to(cls_out.dtype))[None, :] * self.cls_logit_adj_tau)
                    cls_loss = self.classification_loss(cls_logits_adj[cls_mask], cls_target[cls_mask])
                    cls_acc  = (torch.argmax(cls_logits_adj[cls_mask], dim=1) == cls_target[cls_mask]).float().mean().item()
                else:
                    cls_loss = torch.zeros((), device=self.device)
                    cls_acc  = float('nan')
                total_loss = seg_loss + self._cls_weight_schedule() * cls_loss

        if not torch.isfinite(total_loss):
            print("⚠️ Non-finite loss persists. Skipping batch.")
            return {'loss': np.array(float('nan'))}

        if self.grad_scaler is not None and use_amp:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        self._train_step_count += 1

        if not np.isnan(cls_acc):
            self.classification_metrics['train_acc'].append(cls_acc)

        return {
            'loss': total_loss.detach().cpu().numpy(),
            'seg_loss': seg_loss.detach().cpu().numpy(),
            'cls_loss': cls_loss.detach().cpu().numpy(),
            'cls_acc': cls_acc,
        }

    def validation_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        with torch.no_grad():
            seg_out, cls_out = self.network(data, return_both=True)
            seg_logits = seg_out[0] if isinstance(seg_out, (list, tuple)) else seg_out
            seg_logits = self._nan_to_num(self._ensure_bc_first(seg_logits, data))
            target = self._align_target_to_output(batch['target'], seg_logits)
            target_for_metrics = target

            cls_out = self._nan_to_num(cls_out)

            seg_loss = self.loss(seg_logits, target)

            # classification (masked) + logit adjustment
            cls_target, cls_mask = self._cls_targets_and_mask(batch)
            if cls_mask.any():
                cls_logits_adj = self._nan_to_num(cls_out - (self.cls_log_prior.to(cls_out.dtype))[None, :] * self.cls_logit_adj_tau)
                cls_loss = self.classification_loss(cls_logits_adj[cls_mask], cls_target[cls_mask])
                cls_preds = torch.argmax(cls_logits_adj[cls_mask], dim=1)
                cls_acc = (cls_preds == cls_target[cls_mask]).float().mean().item()
                self.classification_metrics['val_acc'].append(cls_acc)
                self._val_cls_acc_epoch.append(cls_acc)
            else:
                cls_loss = torch.zeros((), device=self.device)
                cls_acc = float('nan')

            total_loss = seg_loss + self._cls_weight_schedule() * cls_loss

            # dice bookkeeping as in nnU-Net
            if self.label_manager.has_regions:
                predicted_segmentation_onehot = (torch.sigmoid(seg_logits) > 0.5).long()
            else:
                output_seg = seg_logits.argmax(1)[:, None]
                predicted_segmentation_onehot = torch.zeros(
                    seg_logits.shape, device=seg_logits.device, dtype=torch.float32
                )
                predicted_segmentation_onehot.scatter_(1, output_seg, 1)
                del output_seg

            if self.label_manager.has_ignore_label:
                if not self.label_manager.has_regions:
                    mask = (target_for_metrics != self.label_manager.ignore_label).float()
                    target_for_metrics = target_for_metrics.clone()
                    target_for_metrics[target_for_metrics == self.label_manager.ignore_label] = 0
                else:
                    if target_for_metrics.dtype == torch.bool:
                        mask = ~target_for_metrics[:, -1:]
                    else:
                        mask = 1 - target_for_metrics[:, -1:]
                    target_for_metrics = target_for_metrics[:, :-1]
            else:
                mask = None

            axes = [0] + list(range(2, seg_logits.ndim))
            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot,
                                            target_for_metrics, axes=axes, mask=mask)
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()
            if not self.label_manager.has_regions:
                tp_hard, fp_hard, fn_hard = tp_hard[1:], fp_hard[1:], fn_hard[1:]

        return {
            'loss': total_loss.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'seg_loss': seg_loss.detach().cpu().numpy(),
            'cls_loss': cls_loss.detach().cpu().numpy(),
            'cls_acc': cls_acc,
        }

    def on_epoch_end(self):
        super().on_epoch_end()
        if self.classification_metrics['train_acc']:
            train_acc = float(np.mean(self.classification_metrics['train_acc'][-min(50, len(self.classification_metrics['train_acc'])):]))
            print(f"Classification Train Acc: {train_acc:.4f}")
        if self.classification_metrics['val_acc']:
            val_acc_hist = float(np.mean(self.classification_metrics['val_acc'][-min(50, len(self.classification_metrics['val_acc'])):]))
            print(f"Classification Val Acc: {val_acc_hist:.4f}")

        if self._val_cls_acc_epoch:
            epoch_val_acc = float(np.mean(self._val_cls_acc_epoch))
            print(f"(cls) epoch val acc: {epoch_val_acc:.3f}")
            if epoch_val_acc > self.best_val_cls_acc + 1e-6:
                self.best_val_cls_acc = epoch_val_acc
                try:
                    torch.save(self.network.classification_head.state_dict(), self.best_cls_head_path)
                    print(f"  ↳ saved best classification head → {self.best_cls_head_path}")
                except Exception as e:
                    print("  ↳ failed to save best classification head:", e)

    def initialize_val_metrics(self):
        if not hasattr(self, 'val_metrics'):
            self.val_metrics = []