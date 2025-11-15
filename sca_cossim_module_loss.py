import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.registry import MODELS


@MODELS.register_module()
class SCACosSimModuleLoss(nn.Module):

    def __init__(self, loss_weight=1.0):
        super(SCACosSimModuleLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, feature, target, weight=None):

        if isinstance(feature, torch.Tensor):
            target = target.type_as(feature)
            num_classes = feature.size(2)
            num_instances = feature.size(1)
            target = F.one_hot(
                target.to(torch.int64), num_classes=num_classes + 1
            ).float()

            # ---- SODA-A
            large_col = feature[:, :, 3]  # LV on Feature map
            container_col = feature[:, :, 5]  # CT on Feature Map
            large_target = target[:, :, 3]  # LV on GT
            container_target = target[:, :, 5]  # CT on GT

            a = torch.stack([large_col, container_col], dim=-1)
            L = torch.stack([large_target, container_target], dim=-1)

            cossim = F.cosine_similarity(
                a, L, dim=-1
            )
            a_m = torch.sqrt(
                large_col**2 + container_col**2
            )

            roll_all_zero = torch.all(
                L == 0, dim=-1
            )
            device = roll_all_zero.device
            w = torch.where(
                roll_all_zero,
                torch.full((1, num_instances), 0.1, device=device),
                torch.full((1, num_instances), 0.9, device=device),
            )

            loss = (1 - cossim) * (
                1 - a_m
            )
            loss_angle = ((w * loss).sum()) / num_instances

            if weight is not None:
                loss_angle = loss_angle * weight

            loss_angle = self.loss_weight * loss_angle

        else:
            loss_angle = torch.tensor(0.0)

        return loss_angle
