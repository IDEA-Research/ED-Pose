import copy
import os
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from util import box_ops
from util.keypoint_ops import keypoint_xyzxyz_to_xyxyzz
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .backbones import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .utils import PoseProjector, sigmoid_focal_loss, MLP
from .postprocesses import PostProcess
from .criterion import SetCriterion
from ..registry import MODULE_BUILD_FUNCS

class EDPose(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, 
                    aux_loss=False, iter_update=True,
                    query_dim=4, 
                    random_refpoints_xy=False,
                    fix_refpoints_hw=-1,
                    num_feature_levels=1,
                    nheads=8,
                    two_stage_type='no',
                    dec_pred_class_embed_share=False,
                    dec_pred_bbox_embed_share=False,
                    dec_pred_pose_embed_share=False,
                    two_stage_class_embed_share=True,
                    two_stage_bbox_embed_share=True,
                    dn_number = 100,
                    dn_box_noise_scale = 0.4,
                    dn_label_noise_ratio = 0.5,
                    dn_batch_gt_fuse=False,
                    dn_labelbook_size = 100,
                    dn_attn_mask_type_list = ['group2group'],
                    cls_no_bias = False,
                    num_group = 100,
                    num_body_points = 17,
                    num_box_decoder_layers = 2,
                    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.num_body_points = num_body_points
        self.num_box_decoder_layers = num_box_decoder_layers

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_batch_gt_fuse = dn_batch_gt_fuse
        self.dn_labelbook_size = dn_labelbook_size
        self.dn_attn_mask_type_list = dn_attn_mask_type_list
        assert all([i in ['match2dn', 'dn2dn', 'group2group'] for i in dn_attn_mask_type_list])
        assert not dn_batch_gt_fuse
        

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])


        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"


        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes, bias=(not cls_no_bias))
        if not cls_no_bias:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        _pose_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        _pose_hw_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(_pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_pose_embed.layers[-1].bias.data, 0)
        
        self.num_group = num_group
        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [copy.deepcopy(_class_embed) for i in range(transformer.num_decoder_layers)]

        if num_body_points==17:
            if dec_pred_pose_embed_share:
                pose_embed_layerlist = [_pose_embed for i in range(transformer.num_decoder_layers - num_box_decoder_layers+1)]
            else:
                pose_embed_layerlist = [copy.deepcopy(_pose_embed) for i in range(transformer.num_decoder_layers - num_box_decoder_layers+1)]
        else:
            if dec_pred_pose_embed_share:
                pose_embed_layerlist = [_pose_embed for i in range(transformer.num_decoder_layers - num_box_decoder_layers)]
            else:
                pose_embed_layerlist = [copy.deepcopy(_pose_embed) for i in range(transformer.num_decoder_layers - num_box_decoder_layers)]

        pose_hw_embed_layerlist = [_pose_hw_embed for i in range(transformer.num_decoder_layers - num_box_decoder_layers)]


        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.pose_embed=nn.ModuleList(pose_embed_layerlist)
        self.pose_hw_embed=nn.ModuleList(pose_hw_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.pose_embed = self.pose_embed
        self.transformer.decoder.pose_hw_embed = self.pose_hw_embed
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.num_box_decoder_layers = num_box_decoder_layers
        self.transformer.decoder.num_body_points = num_body_points
        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ['no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
    
            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed

            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
            self.refpoint_embed = None


        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def prepare_for_dn2(self, targets):
        if not self.training:

            device = targets[0]['boxes'].device
            bs = len(targets)
            attn_mask_infere = torch.zeros(bs, self.nheads, self.num_group*(self.num_body_points+1), self.num_group*(self.num_body_points+1),
                                    device=device, dtype=torch.bool)
            group_bbox_kpt = (self.num_body_points+1)
            group_nobbox_kpt = self.num_body_points
            kpt_index = [x for x in range(self.num_group * (self.num_body_points+1)) if x % (self.num_body_points+1) == 0]
            for matchj in range(self.num_group * (self.num_body_points+1)):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
                if sj > 0:
                    attn_mask_infere[:, :, matchj, :sj] = True
                if ej < self.num_group * (self.num_body_points+1):
                    attn_mask_infere[:, :, matchj, ej:] = True
            for match_x in range(self.num_group * (self.num_body_points+1)):
                if match_x % group_bbox_kpt==0:
                    attn_mask_infere[:,:,match_x,kpt_index]=False

            attn_mask_infere = attn_mask_infere.flatten(0, 1)
            return None, None, None, attn_mask_infere,None

        # targets, dn_scalar, noise_scale = dn_args
        device = targets[0]['boxes'].device
        bs = len(targets)
        dn_number = self.dn_number
        dn_box_noise_scale = self.dn_box_noise_scale
        dn_label_noise_ratio = self.dn_label_noise_ratio

        # gather gt boxes and labels
        gt_boxes = [t['boxes'] for t in targets]
        gt_labels = [t['labels'] for t in targets]
        gt_keypoints = [t['keypoints'] for t in targets]
        # repeat them
        def get_indices_for_repeat(now_num, target_num, device='cuda'):
            """
            Input:
                - now_num: int
                - target_num: int
            Output:
                - indices: tensor[target_num]
            """
            out_indice = []
            base_indice = torch.arange(now_num).to(device)
            multiplier = target_num // now_num
            out_indice.append(base_indice.repeat(multiplier))
            residue = target_num % now_num
            out_indice.append(base_indice[torch.randint(0, now_num, (residue,), device=device)])
            return torch.cat(out_indice)

        if self.dn_batch_gt_fuse:
            raise NotImplementedError
            gt_boxes_bsall = torch.cat(gt_boxes) # num_boxes, 4
            gt_labels_bsall = torch.cat(gt_labels)
            num_gt_bsall = gt_boxes_bsall.shape[0]
            if num_gt_bsall > 0:
                indices = get_indices_for_repeat(num_gt_bsall, dn_number, device)
                gt_boxes_expand = gt_boxes_bsall[indices][None].repeat(bs, 1, 1) # bs, num_dn, 4
                gt_labels_expand = gt_labels_bsall[indices][None].repeat(bs,  1) # bs, num_dn
            else:
                # all negative samples when no gt boxes
                gt_boxes_expand = torch.rand(bs, dn_number, 4, device=device)
                gt_labels_expand = torch.ones(bs, dn_number, dtype=torch.int64, device=device) * int(self.num_classes)
        else:
            gt_boxes_expand = []
            gt_labels_expand = []
            gt_keypoints_expand = []
            for idx, (gt_boxes_i, gt_labels_i, gt_keypoint_i) in enumerate(zip(gt_boxes, gt_labels, gt_keypoints)):
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i > 0:
                    indices = get_indices_for_repeat(num_gt_i, dn_number, device)
                    gt_boxes_expand_i = gt_boxes_i[indices] # num_dn, 4
                    gt_labels_expand_i = gt_labels_i[indices]
                    gt_keypoints_expand_i = gt_keypoint_i[indices]
                else:
                    # all negative samples when no gt boxes
                    gt_boxes_expand_i = torch.rand(dn_number, 4, device=device)
                    gt_labels_expand_i = torch.ones(dn_number, dtype=torch.int64, device=device) * int(self.num_classes)
                    gt_keypoints_expand_i = torch.rand(dn_number, self.num_body_points*3, device=device)
                gt_boxes_expand.append(gt_boxes_expand_i)
                gt_labels_expand.append(gt_labels_expand_i)
                gt_keypoints_expand.append(gt_keypoints_expand_i)
            gt_boxes_expand = torch.stack(gt_boxes_expand)
            gt_labels_expand = torch.stack(gt_labels_expand)
            gt_keypoints_expand = torch.stack(gt_keypoints_expand)
        knwon_boxes_expand = gt_boxes_expand.clone()
        knwon_labels_expand = gt_labels_expand.clone()


        # add noise
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0, self.dn_labelbook_size)  # randomly put a new one here
            knwon_labels_expand[chosen_indice] = new_label

        if dn_box_noise_scale > 0:
            diff = torch.zeros_like(knwon_boxes_expand)
            diff[..., :2] = knwon_boxes_expand[..., 2:] / 2
            diff[..., 2:] = knwon_boxes_expand[..., 2:]
            knwon_boxes_expand += torch.mul((torch.rand_like(knwon_boxes_expand) * 2 - 1.0), diff) * dn_box_noise_scale
            knwon_boxes_expand = knwon_boxes_expand.clamp(min=0.0, max=1.0)

        input_query_label = self.label_enc(knwon_labels_expand)
        input_query_bbox = inverse_sigmoid(knwon_boxes_expand)

        # prepare mask

        if 'group2group' in self.dn_attn_mask_type_list:
            attn_mask = torch.zeros(bs, self.nheads, dn_number + self.num_queries, dn_number + self.num_queries, device=device, dtype=torch.bool)
            attn_mask[:, :, dn_number:, :dn_number] = True
            for idx, (gt_boxes_i, gt_labels_i) in enumerate(zip(gt_boxes, gt_labels)):
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i == 0:
                    continue
                for matchi in range(dn_number):
                    si = (matchi // num_gt_i) * num_gt_i
                    ei = (matchi // num_gt_i + 1) * num_gt_i
                    if si > 0:
                        attn_mask[idx, :, matchi, :si] = True
                    if ei < dn_number:
                        attn_mask[idx, :, matchi, ei:dn_number] = True
            attn_mask = attn_mask.flatten(0, 1)




        if 'group2group' in self.dn_attn_mask_type_list:
            attn_mask2 = torch.zeros(bs, self.nheads, dn_number + self.num_group*(self.num_body_points+1), dn_number + self.num_group*(self.num_body_points+1),
                                    device=device, dtype=torch.bool)
            attn_mask2[:, :, dn_number:, :dn_number] = True
            group_bbox_kpt = (self.num_body_points+1)
            group_nobbox_kpt = self.num_body_points
            kpt_index = [x for x in range(self.num_group * (self.num_body_points+1)) if x % (self.num_body_points+1) == 0]
            for matchj in range(self.num_group * (self.num_body_points+1)):
                sj = (matchj // group_bbox_kpt) * group_bbox_kpt
                ej = (matchj // group_bbox_kpt + 1)*group_bbox_kpt
                if sj > 0:
                    attn_mask2[:, :, dn_number:, dn_number:][:, :, matchj, :sj] = True
                if ej < self.num_group * (self.num_body_points+1):
                    attn_mask2[:, :, dn_number:, dn_number:][:, :, matchj, ej:] = True

            for match_x in range(self.num_group * (self.num_body_points+1)):
                if match_x % group_bbox_kpt==0:
                    attn_mask2[:, :, dn_number:, dn_number:][:,:,match_x,kpt_index]=False

            for idx, (gt_boxes_i, gt_labels_i) in enumerate(zip(gt_boxes, gt_labels)):
                num_gt_i = gt_boxes_i.shape[0]
                if num_gt_i == 0:
                    continue
                for matchi in range(dn_number):
                    si = (matchi // num_gt_i) * num_gt_i
                    ei = (matchi // num_gt_i + 1) * num_gt_i
                    if si > 0:
                        attn_mask2[idx, :, matchi, :si] = True
                    if ei < dn_number:
                        attn_mask2[idx, :, matchi, ei:dn_number] = True
            attn_mask2 = attn_mask2.flatten(0, 1)





        mask_dict = {
            'pad_size': dn_number,
            'known_bboxs': gt_boxes_expand,
            'known_labels': gt_labels_expand,
            'known_keypoints': gt_keypoints_expand
        }

        return input_query_label, input_query_bbox, attn_mask,attn_mask2, mask_dict

    def dn_post_process2(self, outputs_class, outputs_coord, outputs_keypoints_list, mask_dict):
        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = [outputs_class_i[:, :mask_dict['pad_size'], :] for outputs_class_i in outputs_class]
            output_known_coord = [outputs_coord_i[:, :mask_dict['pad_size'], :] for outputs_coord_i in outputs_coord]

            outputs_class = [outputs_class_i[:, mask_dict['pad_size']:, :] for outputs_class_i in outputs_class]
            outputs_coord = [outputs_coord_i[:, mask_dict['pad_size']:, :] for outputs_coord_i in outputs_coord]
            outputs_keypoint = outputs_keypoints_list

            mask_dict.update({
                'output_known_coord': output_known_coord,
                'output_known_class': output_known_class
            })
        return outputs_class, outputs_coord, outputs_keypoint


    def forward(self, samples: NestedTensor, targets:List=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask,attn_mask2, mask_dict =\
                self.prepare_for_dn2(targets)

        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask =attn_mask1=attn_mask2= mask_dict = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(srcs, masks, input_query_bbox, poss, input_query_label, attn_mask,attn_mask2)

        # update human boxes
        effective_dn_number = self.dn_number if self.training else 0
        outputs_coord_list = []
        outputs_class=[]
        for dec_lid, (layer_ref_sig, layer_bbox_embed,layer_cls_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed,self.class_embed, hs)):
            if dec_lid < self.num_box_decoder_layers:
                layer_delta_unsig = layer_bbox_embed(layer_hs)
                layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                layer_cls=layer_cls_embed(layer_hs)
                outputs_coord_list.append(layer_outputs_unsig)
                outputs_class.append(layer_cls)
            else:
                layer_hs_bbox_dn = layer_hs[:,:effective_dn_number,:]
                layer_hs_bbox_norm = layer_hs[:,effective_dn_number:,:][:,0::(self.num_body_points+1),:]
                bs = layer_ref_sig.shape[0]
                reference_before_sigmoid_bbox_dn = layer_ref_sig[:,:effective_dn_number,:]
                reference_before_sigmoid_bbox_norm = layer_ref_sig[:,effective_dn_number:,:][:,0::(self.num_body_points+1),:]
                layer_delta_unsig_dn = layer_bbox_embed(layer_hs_bbox_dn)
                layer_delta_unsig_norm = layer_bbox_embed(layer_hs_bbox_norm)
                layer_outputs_unsig_dn = layer_delta_unsig_dn  + inverse_sigmoid(reference_before_sigmoid_bbox_dn)
                layer_outputs_unsig_dn = layer_outputs_unsig_dn.sigmoid()
                layer_outputs_unsig_norm = layer_delta_unsig_norm  + inverse_sigmoid(reference_before_sigmoid_bbox_norm)
                layer_outputs_unsig_norm = layer_outputs_unsig_norm.sigmoid()
                layer_outputs_unsig=torch.cat((layer_outputs_unsig_dn,layer_outputs_unsig_norm),dim=1)
                layer_cls_dn=layer_cls_embed(layer_hs_bbox_dn)
                layer_cls_norm=layer_cls_embed(layer_hs_bbox_norm)
                layer_cls=torch.cat((layer_cls_dn,layer_cls_norm),dim=1)
                outputs_class.append(layer_cls)
                outputs_coord_list.append(layer_outputs_unsig)

        # update keypoints boxes
        outputs_keypoints_list = []
        outputs_keypoints_hw = []
        kpt_index = [x for x in range(self.num_group * (self.num_body_points+1)) if x % (self.num_body_points+1) != 0]
        for dec_lid, (layer_ref_sig, layer_hs) in enumerate(zip(reference[:-1], hs)):
            if dec_lid < self.num_box_decoder_layers:
                assert isinstance(layer_hs, torch.Tensor)
                bs = layer_hs.shape[0]
                layer_res = layer_hs.new_zeros((bs, self.num_queries, self.num_body_points * 3))
                outputs_keypoints_list.append(layer_res)
            else:
                bs = layer_ref_sig.shape[0]
                layer_hs_kpt=layer_hs[:, effective_dn_number:, :].index_select(1,torch.tensor(kpt_index,device=layer_hs.device))
                delta_xy_unsig = self.pose_embed[dec_lid - self.num_box_decoder_layers](layer_hs_kpt)
                layer_ref_sig_kpt = layer_ref_sig[:, effective_dn_number:, :].index_select(1, torch.tensor(kpt_index,device=layer_hs.device))
                layer_outputs_unsig_keypoints = delta_xy_unsig + inverse_sigmoid(layer_ref_sig_kpt[...,:2])
                vis_xy_unsig = torch.ones_like(layer_outputs_unsig_keypoints,device=layer_outputs_unsig_keypoints.device)
                xyv = torch.cat((layer_outputs_unsig_keypoints, vis_xy_unsig[:,:,0].unsqueeze(-1)),dim=-1)
                xyv = xyv.sigmoid()
                layer_res = xyv.reshape((bs, self.num_group, self.num_body_points, 3)).flatten(2, 3)
                layer_hw= layer_ref_sig_kpt[...,2:].reshape(bs, self.num_group, self.num_body_points, 2).flatten(2, 3)
                layer_res = keypoint_xyzxyz_to_xyxyzz(layer_res)
                outputs_keypoints_list.append(layer_res)
                outputs_keypoints_hw.append(layer_hw)

        dn_mask_dict = mask_dict
        if self.dn_number > 0 and dn_mask_dict is not None:
            outputs_class, outputs_coord_list, outputs_keypoints_list = self.dn_post_process2(outputs_class, outputs_coord_list, outputs_keypoints_list, dn_mask_dict)
            dn_class_input = dn_mask_dict['known_labels']
            dn_bbox_input = dn_mask_dict['known_bboxs']
            dn_class_pred = dn_mask_dict['output_known_class']
            dn_bbox_pred = dn_mask_dict['output_known_coord']


        for idx, (_out_class, _out_bbox, _out_keypoint) in enumerate(zip(outputs_class, outputs_coord_list, outputs_keypoints_list)):
            assert  _out_class.shape[1] == _out_bbox.shape[1] == _out_keypoint.shape[1]


        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord_list[-1],'pred_keypoints': outputs_keypoints_list[-1]}
        if self.dn_number > 0 and dn_mask_dict is not None:
            out.update(
                    {
                        'dn_class_input': dn_class_input,
                        'dn_bbox_input': dn_bbox_input,
                        'dn_class_pred': dn_class_pred[-1],
                        'dn_bbox_pred': dn_bbox_pred[-1],
                        'num_tgt': dn_mask_dict['pad_size']
                    }
                )   

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list, outputs_keypoints_list)
            if self.dn_number > 0 and dn_mask_dict is not None:
                assert len(dn_class_pred[:-1]) == len(dn_bbox_pred[:-1]) == len(out["aux_outputs"])
                for aux_out, dn_class_pred_i, dn_bbox_pred_i in zip(out["aux_outputs"], dn_class_pred, dn_bbox_pred):
                    aux_out.update({
                        'dn_class_input': dn_class_input,
                        'dn_bbox_input': dn_bbox_input,
                        'dn_class_pred': dn_class_pred_i,
                        'dn_bbox_pred': dn_bbox_pred_i,
                        'num_tgt': dn_mask_dict['pad_size']
                    })
        # for encoder output
        if hs_enc is not None:
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            interm_pose = torch.zeros_like(outputs_keypoints_list[0])
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord,'pred_keypoints':interm_pose}

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord,outputs_keypoints):
        return [{'pred_logits': a, 'pred_boxes': b,'pred_keypoints': c}
                for a, b,c in zip(outputs_class[:-1], outputs_coord[:-1],outputs_keypoints[:-1])]



@MODULE_BUILD_FUNCS.registe_with_name(module_name='edpose')
def build_edpose(args):

    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_class_embed_share = args.dec_pred_class_embed_share
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share


    model = EDPose(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,

        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,

        dn_number = args.dn_number if args.use_dn else 0,
        dn_box_noise_scale = args.dn_box_noise_scale,
        dn_label_noise_ratio = args.dn_label_noise_ratio,
        dn_batch_gt_fuse=args.dn_batch_gt_fuse,
        dn_attn_mask_type_list=args.dn_attn_mask_type_list,
        dn_labelbook_size = dn_labelbook_size,

        cls_no_bias=args.cls_no_bias,
        num_group=args.num_group,
        num_body_points=args.num_body_points,
        num_box_decoder_layers=args.num_box_decoder_layers
    )
    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {
        'loss_ce': args.cls_loss_coef, 
        'loss_bbox': args.bbox_loss_coef,
        "loss_keypoints":args.keypoints_loss_coef,
        "loss_oks":args.oks_loss_coef
    }
    weight_dict['loss_giou'] = args.giou_loss_coef

    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)


    if args.use_dn:
        weight_dict.update({
            'dn_loss_ce': args.dn_label_coef,
            'dn_loss_bbox': args.bbox_loss_coef * args.dn_bbox_coef,
            'dn_loss_giou': args.giou_loss_coef * args.dn_bbox_coef,
        })

    clean_weight_dict = copy.deepcopy(weight_dict)

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            for k, v in clean_weight_dict.items():
                if i < args.num_box_decoder_layers and 'keypoints' in k:
                    continue
                aux_weight_dict.update({k + f'_{i}': v})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
            'loss_keypoints': 1.0 if not no_interm_box_loss else 0.0,
            'loss_oks': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items() if 'keypoints' not in k})
        weight_dict.update(interm_weight_dict)

        interm_weight_dict.update({k + f'_query_expand': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)


    losses = ['labels', 'boxes', "keypoints"]
    if args.dn_number > 0:
        losses += ["dn_label", "dn_bbox"]
    losses += ["matching"]

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses, num_box_decoder_layers=args.num_box_decoder_layers,num_body_points=args.num_body_points)
    criterion.to(device)
    postprocessors = {
        'bbox': PostProcess(num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold,num_body_points=args.num_body_points),
    }

    return model, criterion, postprocessors