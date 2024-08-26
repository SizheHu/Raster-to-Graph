# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
from torch import nn
from torch.nn.init import constant_, normal_, kaiming_uniform_
from models.deformable_decoder import DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from models.deformable_encoder import DeformableTransformerEncoderLayer, DeformableTransformerEncoder
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, hidden_dim=256, nhead=8,
                 num_encoder_layers=2, num_decoder_layers=6, dim_feedforward=1024, dropout=0,
                 activation="relu",
                 num_feature_levels=3, dec_n_points=32, enc_n_points=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        encoder_layer = DeformableTransformerEncoderLayer(hidden_dim, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, hidden_dim))
        self.query_embed_to_reference_points = nn.Linear(hidden_dim, 2)
        self._reset_parameters()

    # parameters initialization
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        kaiming_uniform_(self.query_embed_to_reference_points.weight.data)
        constant_(self.query_embed_to_reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = None
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        bs, _, c = memory.shape
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.query_embed_to_reference_points(query_embed).sigmoid()
        outputs = self.decoder(tgt, reference_points, memory,
                          spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        return outputs, reference_points
