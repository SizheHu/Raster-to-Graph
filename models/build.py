import torch
from models.position_encoding import PositionEmbeddingSine
from models.criterion import SetCriterion
from models.deformable_detr import DeformableDETR
from models.deformable_transformer import DeformableTransformer
from models.backbone import Joiner, Backbone
from models.matcher import HungarianMatcher
from models.postprocess import PostProcess


def build_backbone_with_position_embedding(args):
    position_embedding = PositionEmbeddingSine(args.hidden_dim // 2, normalize=True)
    backbone = Backbone(args.backbone)
    return Joiner(backbone, position_embedding)


def build_deforamble_transformer(args):
    return DeformableTransformer(
        hidden_dim=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        num_feature_levels=args.num_feature_maps,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points)


def build_model(args):
    backbone_with_position_embedding = build_backbone_with_position_embedding(args)
    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone_with_position_embedding,
        transformer,
        num_classes=args.num_classes,
        num_classes_edges=args.num_classes_edges,
        num_queries=args.num_queries,
        num_feature_maps=args.num_feature_maps
    )
    model.to(torch.device(args.device))
    return model


def build_criterion(args):
    matcher = HungarianMatcher(cost_point=args.set_cost_point,
                                  cost_edge=args.set_cost_edge,
                                  cost_last_edge=args.set_cost_last_edge,
                                  cost_this_edge=args.set_cost_this_edge,
                                  cost_semantic_left_up=args.set_cost_semantic,
                                  cost_semantic_right_up=args.set_cost_semantic,
                                  cost_semantic_right_down=args.set_cost_semantic,
                                  cost_semantic_left_down=args.set_cost_semantic)

    weight_dict = {'loss_point': args.point_loss_coef, 'loss_edge': args.edge_loss_coef,
                   'loss_last_edge': args.last_edge_loss_coef, 'loss_this_edge': args.this_edge_loss_coef,
                   'loss_semantic_left_up': args.semantic_loss_coef, 'loss_semantic_right_up': args.semantic_loss_coef,
                   'loss_semantic_right_down': args.semantic_loss_coef, 'loss_semantic_left_down': args.semantic_loss_coef}

    criterion = SetCriterion(args.num_classes, matcher, weight_dict)
    criterion.to(torch.device(args.device))

    return criterion


def build_postprocessor():
    return PostProcess()
