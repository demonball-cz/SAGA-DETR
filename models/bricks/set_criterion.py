import copy
from typing import Dict

import torch
import torch.distributed
from torch import nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from models.bricks.losses import sigmoid_focal_loss, vari_sigmoid_focal_loss
from util.utils import get_world_size, is_dist_avail_and_initialized


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict,
        alpha: float = 0.25,
        gamma: float = 2.0,
        two_stage_binary_cls=False,
        use_acml: bool = False,
        acml_topk: int = 5,
        acml_tau0: float = 0.2,
        acml_alpha: float = 0.05,
        acml_adaptive_margin: bool = True,
    ):
        """Create the criterion.

        :param num_classes: number of object categories, omitting the special no-object category
        :param matcher: module able to compute a matching between targets and proposals
        :param weight_dict: dict containing as key the names of the losses and as values their relative weight
        :param alpha: alpha in Focal Loss, defaults to 0.25
        :param gamma: gamma in Focal loss, defaults to 2.0
        :param two_stage_binary_cls: Whether to use two-stage binary classification loss, defaults to False
        """        
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.gamma = gamma
        self.two_stage_binary_cls = two_stage_binary_cls
        self.use_acml = use_acml
        self.acml_topk = acml_topk
        self.acml_tau0 = acml_tau0
        self.acml_alpha = acml_alpha
        self.acml_adaptive_margin = acml_adaptive_margin

    def loss_labels(self, outputs, targets, num_boxes, indices, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_class = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.alpha,
                gamma=self.gamma,
            ) * src_logits.shape[1]
        )
        losses = {"loss_class": loss_class}
        return losses

    def loss_boxes(self, outputs, targets, num_boxes, indices, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops._box_cxcywh_to_xyxy(src_boxes),
                box_ops._box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def calculate_acml_cost(self, pred_boxes, pred_logits, gt_boxes, gt_labels):
        out_prob = pred_logits.sigmoid()
        neg_cost_class = (
            -(1 - self.matcher.focal_alpha)
            * out_prob**self.matcher.focal_gamma
            * (1 - out_prob + 1e-6).log()
        )
        pos_cost_class = (
            -self.matcher.focal_alpha
            * (1 - out_prob)**self.matcher.focal_gamma
            * (out_prob + 1e-6).log()
        )
        cost_class = pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]
        cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1)
        cost_giou = -box_ops.generalized_box_iou(
            box_ops._box_cxcywh_to_xyxy(pred_boxes),
            box_ops._box_cxcywh_to_xyxy(gt_boxes),
        )
        return (
            self.matcher.cost_class * cost_class
            + self.matcher.cost_bbox * cost_bbox
            + self.matcher.cost_giou * cost_giou
        )

    def loss_acml(self, outputs, targets, indices):
        pred_boxes = outputs["pred_boxes"]
        pred_logits = outputs["pred_logits"]
        losses = []
        margins = []
        violations = []

        for boxes_i, logits_i, target_i, (src_idx, tgt_idx) in zip(pred_boxes, pred_logits, targets, indices):
            if len(tgt_idx) == 0 or len(target_i["labels"]) == 0:
                continue
            gt_boxes = target_i["boxes"]
            gt_labels = target_i["labels"]
            cost = self.calculate_acml_cost(boxes_i, logits_i, gt_boxes, gt_labels)
            num_queries = cost.shape[0]

            for matched_src, matched_tgt in zip(src_idx.to(cost.device), tgt_idx.to(cost.device)):
                competitor_mask = torch.ones(num_queries, dtype=torch.bool, device=cost.device)
                competitor_mask[matched_src] = False
                competitor_cost = cost[competitor_mask, matched_tgt]
                if competitor_cost.numel() == 0:
                    continue
                topk = min(self.acml_topk, competitor_cost.numel())
                hard_cost = torch.topk(competitor_cost, k=topk, largest=False).values
                matched_cost = cost[matched_src, matched_tgt]
                margin = hard_cost - matched_cost

                if self.acml_adaptive_margin:
                    wh = gt_boxes[matched_tgt, 2:].clamp(min=1e-4)
                    tau = self.acml_tau0 + self.acml_alpha / torch.sqrt((wh[0] * wh[1]).clamp(min=1e-4))
                else:
                    tau = cost.new_tensor(self.acml_tau0)

                acml = F.relu(tau - margin)
                losses.append(acml.mean())
                margins.append(margin.detach().mean())
                violations.append((margin.detach() < tau.detach()).float().mean())

        if len(losses) == 0:
            zero = pred_boxes.sum() * 0.0
            return {
                "loss_acml": zero,
                "metric_acml_margin": zero.detach(),
                "metric_acml_violation_rate": zero.detach(),
            }

        return {
            "loss_acml": torch.stack(losses).mean(),
            "metric_acml_margin": torch.stack(margins).mean(),
            "metric_acml_violation_rate": torch.stack(violations).mean(),
        }

    def calculate_loss(self, outputs, targets, num_boxes, indices=None, include_acml=False, **kwargs):
        losses = {}
        # get matching results for each image
        if not indices:
            gt_boxes, gt_labels = list(zip(*map(lambda x: (x["boxes"], x["labels"]), targets)))
            pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
            indices = list(map(self.matcher, pred_boxes, pred_logits, gt_boxes, gt_labels))
        loss_class = self.loss_labels(outputs, targets, num_boxes, indices=indices)
        loss_boxes = self.loss_boxes(outputs, targets, num_boxes, indices=indices)
        losses.update(loss_class)
        losses.update(loss_boxes)
        if include_acml and self.use_acml:
            losses.update(self.loss_acml(outputs, targets, indices=indices))
        return losses

    def forward(self, outputs, targets):
        """This performs the loss computation

        :param outputs: dict of tensors, see the output specification of the model for the format
        :param targets: list of dicts, such that len(targets) == batch_size
        :return: a dict containing losses
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            data=[num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        matching_outputs = {k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"}
        losses.update(self.calculate_loss(matching_outputs, targets, num_boxes, include_acml=True))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # get matching results for each image
                losses_aux = self.calculate_loss(aux_outputs, targets, num_boxes)
                losses.update({k + f"_{i}": v for k, v in losses_aux.items()})

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            if self.two_stage_binary_cls:
                for bt in bin_targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            losses_enc = self.calculate_loss(enc_outputs, bin_targets, num_boxes)
            losses.update({k + f"_enc": v for k, v in losses_enc.items()})

        return losses


class HybridSetCriterion(SetCriterion):
    def loss_labels(self, outputs, targets, num_boxes, indices, **kwargs):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou_score = torch.diag(
            box_ops.box_iou(
                box_ops._box_cxcywh_to_xyxy(src_boxes),
                box_ops._box_cxcywh_to_xyxy(target_boxes),
            )
        ).detach()  # add detach according to RT-DETR

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        # construct onehot targets, shape: (batch_size, num_queries, num_classes)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1]

        # construct iou_score, shape: (batch_size, num_queries)
        target_score = torch.zeros_like(target_classes, dtype=iou_score.dtype)
        target_score[idx] = iou_score

        loss_class = (
            vari_sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                target_score,
                num_boxes=num_boxes,
                alpha=self.alpha,
                gamma=self.gamma,
            ) * src_logits.shape[1]
        )
        losses = {"loss_class": loss_class}
        return losses
