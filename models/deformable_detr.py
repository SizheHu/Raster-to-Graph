import torch
import torch.nn.functional as F
from torch import nn
import math
from util.misc import (NestedTensor, inverse_sigmoid)
from util.anumber_utils import anumber_func
from .mlp import MLP



class DeformableDETR(nn.Module):

    def __init__(self, backbone_with_position_embedding, transformer, num_classes, num_classes_edges, num_queries, num_feature_maps):
        super().__init__()
        self.backbone_with_position_embedding = backbone_with_position_embedding
        self.transformer = transformer

        self.edge_embed = nn.Linear(transformer.hidden_dim, 17)
        self.last_edge_embed = nn.Linear(transformer.hidden_dim, 17)

        self.this_edge_embed = MLP(transformer.hidden_dim, transformer.hidden_dim, 17, 2)

        self.point_embed = MLP(transformer.hidden_dim, transformer.hidden_dim, 2, 4)

        self.query_embed = nn.Embedding(num_queries, transformer.hidden_dim * 2)

        self.semantic_left_up_embed = nn.Linear(transformer.hidden_dim, 13)
        self.semantic_right_up_embed = nn.Linear(transformer.hidden_dim, 13)
        self.semantic_right_down_embed = nn.Linear(transformer.hidden_dim, 13)
        self.semantic_left_down_embed = nn.Linear(transformer.hidden_dim, 13)

        input_proj_list = []
        for _ in range(num_feature_maps):
            if _ == anumber_func(transformer.hidden_dim):
                input_proj_list.append(nn.Sequential(
                    nn.GroupNorm(32, transformer.hidden_dim),
                ))
            else:
                in_channels = backbone_with_position_embedding.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, transformer.hidden_dim, kernel_size=(1, 1)),
                    nn.GroupNorm(32, transformer.hidden_dim),
                ))
        self.input_proj = nn.ModuleList(input_proj_list)


    def forward(self, samples: NestedTensor):
        features, pos = self.backbone_with_position_embedding(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
        query_embeds = self.query_embed.weight


        outputs, reference_points = self.transformer(srcs, masks, pos, query_embeds)

        reference = inverse_sigmoid(reference_points)
        outputs_edge = self.edge_embed(outputs)

        outputs_last_edge = self.last_edge_embed(outputs)
        outputs_this_edge = self.this_edge_embed(outputs)

        outputs_semantic_left_up = self.semantic_left_up_embed(outputs)
        outputs_semantic_right_up = self.semantic_right_up_embed(outputs)
        outputs_semantic_right_down = self.semantic_right_down_embed(outputs)
        outputs_semantic_left_down = self.semantic_left_down_embed(outputs)


        tmp = self.point_embed(outputs)
        tmp[..., :2] += reference
        outputs_coord = tmp.sigmoid()
        out = {'pred_points': outputs_coord, 'pred_edges': outputs_edge,
               'pred_last_edges': outputs_last_edge, 'pred_this_edges': outputs_this_edge,
               'pred_semantic_left_up': outputs_semantic_left_up, 'pred_semantic_right_up': outputs_semantic_right_up,
               'pred_semantic_right_down': outputs_semantic_right_down, 'pred_semantic_left_down': outputs_semantic_left_down}

        return out
