import os
from collections import OrderedDict

from torch import nn

from models.backbones.resnet import ResNetBackbone
from models.bricks.misc import FrozenBatchNorm2d
from models.bricks.position_encoding import PositionEmbeddingSine
from models.bricks.post_process import PostProcess
from models.bricks.saga_transformer import (
    SalienceTransformer,
    SalienceTransformerDecoder,
    SalienceTransformerDecoderLayer,
    SalienceTransformerEncoder,
    SalienceTransformerEncoderLayer,
)
from models.bricks.set_criterion import HybridSetCriterion
from models.detectors.saga_detr import SalienceCriterion, SalienceDETR
from models.matcher.hungarian_matcher import HungarianMatcher
from models.necks.channel_mapper import ChannelMapper
from models.necks.repnet import RepVGGPluXNetwork


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def env_float(name, default):
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


dataset_file = "visdrone"

# mostly changed parameters
embed_dim = 256
num_classes = 10
num_queries = env_int("SAGA_NUM_QUERIES", 900)
num_feature_levels = 4
transformer_enc_layers = 6
transformer_dec_layers = 6
num_heads = 8
dim_feedforward = 2048

use_acml = env_bool("SAGA_USE_ACML", False)
acml_weight = env_float("SAGA_ACML_WEIGHT", 0.05)
acml_topk = env_int("SAGA_ACML_TOPK", 5)
acml_tau0 = env_float("SAGA_ACML_TAU0", 0.2)
acml_alpha = env_float("SAGA_ACML_ALPHA", 0.05)
acml_adaptive_margin = not env_bool("SAGA_ACML_FIXED_MARGIN", False)

use_gsda = env_bool("SAGA_USE_GSDA", False)
use_bbcr = env_bool("SAGA_USE_BBCR", False)
bbcr_layers = env_int("SAGA_BBCR_LAYERS", 2)
gsda_variant = os.getenv("SAGA_GSDA_VARIANT", "full")
gsda_scale_bias = gsda_variant != "no_scale_bias"
gsda_residual = gsda_variant != "no_residual"

# instantiate model components
position_embedding = PositionEmbeddingSine(embed_dim // 2, temperature=10000, normalize=True, offset=-0.5)

backbone_weights = OrderedDict() if env_bool("SAGA_NO_PRETRAINED_BACKBONE", False) else None
backbone = ResNetBackbone(
    "resnet50",
    weights=backbone_weights,
    norm_layer=FrozenBatchNorm2d,
    return_indices=(1, 2, 3),
    freeze_indices=(0,),
)

neck = ChannelMapper(
    in_channels=backbone.num_channels,
    out_channels=embed_dim,
    num_outs=num_feature_levels,
)

transformer = SalienceTransformer(
    encoder=SalienceTransformerEncoder(
        encoder_layer=SalienceTransformerEncoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
        ),
        num_layers=transformer_enc_layers,
    ),
    neck=RepVGGPluXNetwork(
        in_channels_list=neck.num_channels,
        out_channels_list=neck.num_channels,
        norm_layer=nn.BatchNorm2d,
        activation=nn.SiLU,
        groups=4,
    ),
    decoder=SalienceTransformerDecoder(
        decoder_layer=SalienceTransformerDecoderLayer(
            embed_dim=embed_dim,
            n_heads=num_heads,
            dropout=0.0,
            activation=nn.ReLU(inplace=True),
            n_levels=num_feature_levels,
            n_points=4,
            d_ffn=dim_feedforward,
            use_gsda=use_gsda,
            gsda_start_layer=1,
            gsda_scale_bias=gsda_scale_bias,
            gsda_residual=gsda_residual,
            use_bbcr=use_bbcr,
            bbcr_layers=bbcr_layers,
        ),
        num_layers=transformer_dec_layers,
        num_classes=num_classes,
    ),
    num_classes=num_classes,
    num_feature_levels=num_feature_levels,
    two_stage_num_proposals=num_queries,
    level_filter_ratio=(0.4, 0.8, 1.0, 1.0),
    layer_filter_ratio=(1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
)

matcher = HungarianMatcher(cost_class=2, cost_bbox=5, cost_giou=2)

weight_dict = {"loss_class": 1, "loss_bbox": 5, "loss_giou": 2}
if use_acml:
    weight_dict.update({"loss_acml": acml_weight})
weight_dict.update({"loss_class_dn": 1, "loss_bbox_dn": 5, "loss_giou_dn": 2})
weight_dict.update({
    k + f"_{i}": v
    for i in range(transformer_dec_layers - 1)
    for k, v in weight_dict.items()
    if k != "loss_acml"
})
weight_dict.update({"loss_class_enc": 1, "loss_bbox_enc": 5, "loss_giou_enc": 2})
weight_dict.update({"loss_salience": 2})

criterion = HybridSetCriterion(
    num_classes,
    matcher=matcher,
    weight_dict=weight_dict,
    alpha=0.25,
    gamma=2.0,
    use_acml=use_acml,
    acml_topk=acml_topk,
    acml_tau0=acml_tau0,
    acml_alpha=acml_alpha,
    acml_adaptive_margin=acml_adaptive_margin,
)
foreground_criterion = SalienceCriterion(noise_scale=0.0, alpha=0.25, gamma=2.0)
postprocessor = PostProcess(select_box_nums_for_evaluation=300)

model = SalienceDETR(
    backbone=backbone,
    neck=neck,
    position_embedding=position_embedding,
    transformer=transformer,
    criterion=criterion,
    focus_criterion=foreground_criterion,
    postprocessor=postprocessor,
    num_classes=num_classes,
    num_queries=num_queries,
    aux_loss=True,
    min_size=800,
    max_size=1333,
)
