_base_ = ['coco_transformer.py']
num_classes=2
lr = 0.0001
param_dict_type = 'default'
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0']
lr_linear_proj_names = ['reference_points', 'sampling_offsets']
lr_linear_proj_mult = 0.1
ddetr_lr_param = False
batch_size = 2
weight_decay = 0.0001
epochs = 50
lr_drop = 11
save_checkpoint_interval = 100
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [33, 45]


modelname = 'edpose'
frozen_weights = None
backbone = 'resnet50'
use_checkpoint = False

dilation = False
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
random_refpoints_xy = False
fix_refpoints_hw = -1
dec_layer_number = None
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
dln_xy_noise = 0.2
dln_hw_noise = 0.2
two_stage_type = 'standard'
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
rm_detach = None
num_select = 50
transformer_activation = 'relu'
batch_norm_type = 'FrozenBatchNorm2d'

masks = False
aux_loss = True
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
set_cost_keypoints = 10.0
set_cost_kpvis = 0.0
set_cost_oks=4.0
cls_loss_coef = 2.0
bbox_loss_coef = 5.0
keypoints_loss_coef = 10.0
oks_loss_coef=4.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
focal_alpha = 0.25
rm_self_attn_layers = None
indices_idx_list = [1,2,3,4,5,6,7]

decoder_sa_type = 'sa'
matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
nms_iou_threshold = -1

dec_pred_bbox_embed_share = False
dec_pred_class_embed_share = False
dec_pred_pose_embed_share = False

# for dn
use_dn = True
dn_number = 100
dn_box_noise_scale = 0.4
dn_label_noise_ratio = 0.5
embed_init_tgt = False
dn_label_coef = 0.3
dn_bbox_coef = 0.5
dn_batch_gt_fuse = False
dn_attn_mask_type_list = ['match2dn', 'dn2dn', 'group2group']
dn_labelbook_size = 100

match_unstable_error = False

# for ema
use_ema = True
ema_decay = 0.9997
ema_epoch = 0

cls_no_bias = False
num_body_points = 17 # for coco
num_group = 100
num_box_decoder_layers = 2
no_mmpose_keypoint_evaluator = True
strong_aug=False