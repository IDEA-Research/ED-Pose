import copy
import os
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops.boxes import nms
from torch import Tensor
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .utils import PoseProjector, sigmoid_focal_loss, MLP,OKSLoss

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, num_box_decoder_layers=2,num_body_points=17):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.vis = 0.1
        self.abs = 1
        self.num_body_points=num_body_points
        self.num_box_decoder_layers = num_box_decoder_layers
        self.oks=OKSLoss(linear=True,
                 num_keypoints=num_body_points,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0)
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        indices = indices[0]
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_keypoints(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the keypoints
        """
        indices = indices[0]
        idx = self._get_src_permutation_idx(indices)
        src_keypoints = outputs['pred_keypoints'][idx] # xyxyvv

        if len(src_keypoints) == 0:
            device = outputs["pred_logits"].device
            losses = {
                'loss_keypoints': torch.as_tensor(0., device=device)+src_keypoints.sum()*0,
                'loss_oks': torch.as_tensor(0., device=device)+src_keypoints.sum()*0,
            }
            return losses
        Z_pred = src_keypoints[:, 0:(self.num_body_points*2)]
        V_pred = src_keypoints[:, (self.num_body_points*2):]
        targets_keypoints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        targets_area = torch.cat([t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        Z_gt = targets_keypoints[:, 0:(self.num_body_points*2)]
        V_gt: torch.Tensor = targets_keypoints[:, (self.num_body_points*2):]
        oks_loss=self.oks(Z_pred,Z_gt,V_gt,targets_area,weight=None,avg_factor=None,reduction_override=None)
        pose_loss = F.l1_loss(Z_pred, Z_gt, reduction='none')
        pose_loss = pose_loss * V_gt.repeat_interleave(2, dim=1)
        losses = {}
        losses['loss_keypoints'] = pose_loss.sum() / num_boxes        
        losses['loss_oks'] = oks_loss.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        indices = indices[0]
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses


    def loss_dn_boxes(self, outputs, targets, indices, num_boxes):
        """
        Input:
            - src_boxes: bs, num_dn, 4
            - tgt_boxes: bs, num_dn, 4

        """
        indices = indices[0]
        if 'num_tgt' not in outputs:
            device = outputs["pred_logits"].device
            losses = {
                'dn_loss_bbox': torch.as_tensor(0., device=device),
                'dn_loss_giou': torch.as_tensor(0., device=device),
            }
            return losses

        num_tgt = outputs['num_tgt']
        src_boxes = outputs['dn_bbox_pred']
        tgt_boxes = outputs['dn_bbox_input']
        
        return self.tgt_loss_boxes(src_boxes, tgt_boxes, num_tgt)


    def loss_dn_labels(self, outputs, targets, indices, num_boxes):
        """
        Input:
            - src_logits: bs, num_dn, num_classes
            - tgt_labels: bs, num_dn

        """
        indices = indices[0]
        if 'num_tgt' not in outputs:
            device = outputs["pred_logits"].device
            losses = {
                'dn_loss_ce': torch.as_tensor(0., device=device),
            }
            return losses



        num_tgt = outputs['num_tgt']
        src_logits = outputs['dn_class_pred'] # bs, num_dn, text_len
        tgt_labels = outputs['dn_class_input']

        return self.tgt_loss_labels(src_logits, tgt_labels, num_tgt)

    @torch.no_grad()
    def loss_matching_cost(self, outputs, targets, indices, num_boxes):
        """
        Input:
            - src_logits: bs, num_dn, num_classes
            - tgt_labels: bs, num_dn

        """
        cost_mean_dict = indices[1]
        losses = {"set_{}".format(k):v for k,v in cost_mean_dict.items()}
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

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            "keypoints":self.loss_keypoints,
            'boxes': self.loss_boxes,
            "dn_label": self.loss_dn_labels,
            "dn_bbox": self.loss_dn_boxes,
            "matching": self.loss_matching_cost
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def prep_for_dn2(self,mask_dict):
        known_bboxs = mask_dict['known_bboxs']
        known_labels = mask_dict['known_labels']
        output_known_coord = mask_dict['output_known_coord']
        output_known_class = mask_dict['output_known_class']
        num_tgt = mask_dict['pad_size']

        return known_labels, known_bboxs,output_known_class,output_known_coord,num_tgt

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        device=next(iter(outputs.values())).device


        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # loss for final layer
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices0_copy = indices
            indices_list = []
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    if loss in ['keypoints'] and idx < self.num_box_decoder_layers:
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss in ['dn_bbox', 'dn_label', 'keypoints']:
                    continue
                kwargs = {}
                if loss == 'labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        # aux_init loss
        if 'query_expand' in outputs:
            interm_outputs = outputs['query_expand']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                if loss in ['dn_bbox', 'dn_label']:
                    continue
                kwargs = {}
                if loss == 'labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_query_expand': v for k, v in l_dict.items()}
                losses.update(l_dict)


        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def tgt_loss_boxes(self, src_boxes, tgt_boxes, num_tgt,):
        """
        Input:
            - src_boxes: bs, num_dn, 4
            - tgt_boxes: bs, num_dn, 4

        """
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')

        losses = {}
        losses['dn_loss_bbox'] = loss_bbox.sum() / num_tgt

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes.flatten(0, 1)),
            box_ops.box_cxcywh_to_xyxy(tgt_boxes.flatten(0, 1))))
        losses['dn_loss_giou'] = loss_giou.sum() / num_tgt
        return losses


    def tgt_loss_labels(self, src_logits: Tensor, tgt_labels: Tensor, num_tgt: int, log: bool=True):
        """
        Input:
            - src_logits: bs, num_dn, num_classes
            - tgt_labels: bs, num_dn

        """
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_tgt, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'dn_loss_ce': loss_ce}

        return losses