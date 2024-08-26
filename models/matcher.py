# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    def __init__(self, cost_point, cost_edge, cost_last_edge, cost_this_edge,
                 cost_semantic_left_up, cost_semantic_right_up, cost_semantic_right_down, cost_semantic_left_down):
        super().__init__()
        self.cost_point = cost_point
        self.cost_edge = cost_edge
        self.cost_last_edge = cost_last_edge
        self.cost_this_edge = cost_this_edge
        self.cost_semantic_left_up = cost_semantic_left_up
        self.cost_semantic_right_up = cost_semantic_right_up
        self.cost_semantic_right_down = cost_semantic_right_down
        self.cost_semantic_left_down = cost_semantic_left_down

    def forward(self, outputs, targets):

        with torch.no_grad():
            bs, num_queries = outputs["pred_edges"].shape[:2]
            out_point = outputs["pred_points"].flatten(0, 1)
            out_edge = outputs["pred_edges"].flatten(0, 1).sigmoid()
            out_last_edge = outputs["pred_last_edges"].flatten(0, 1).sigmoid()
            out_this_edge = outputs["pred_this_edges"].flatten(0, 1).sigmoid()

            out_semantic_left_up = outputs["pred_semantic_left_up"].flatten(0, 1).sigmoid()
            out_semantic_right_up = outputs["pred_semantic_right_up"].flatten(0, 1).sigmoid()
            out_semantic_right_down = outputs["pred_semantic_right_down"].flatten(0, 1).sigmoid()
            out_semantic_left_down = outputs["pred_semantic_left_down"].flatten(0, 1).sigmoid()

            tgt_point = torch.cat([v["points"] for v in targets])
            tgt_edges = torch.cat([v["edges"] for v in targets])
            tgt_last_edges = torch.cat([v["last_edges"] for v in targets])
            tgt_this_edges = torch.cat([v["this_edges"] for v in targets])

            tgt_semantic_left_up = torch.cat([v["semantic_left_up"] for v in targets])
            tgt_semantic_right_up = torch.cat([v["semantic_right_up"] for v in targets])
            tgt_semantic_right_down = torch.cat([v["semantic_right_down"] for v in targets])
            tgt_semantic_left_down = torch.cat([v["semantic_left_down"] for v in targets])


            cost_point = torch.cdist(out_point, tgt_point, p=1)




            alpha2 = 0.25
            gamma2 = 2
            neg_cost_edge = (1 - alpha2) * (out_edge ** gamma2) * (-(1 - out_edge + 1e-8).log())
            pos_cost_edge = alpha2 * ((1 - out_edge) ** gamma2) * (-(out_edge + 1e-8).log())
            cost_edge = pos_cost_edge[:, tgt_edges] - neg_cost_edge[:, tgt_edges]


            alpha3 = 0.25
            gamma3 = 2
            neg_cost_last_edge = (1 - alpha3) * (out_last_edge ** gamma3) * (-(1 - out_last_edge + 1e-8).log())
            pos_cost_last_edge = alpha3 * ((1 - out_last_edge) ** gamma3) * (-(out_last_edge + 1e-8).log())
            cost_last_edge = pos_cost_last_edge[:, tgt_last_edges] - neg_cost_last_edge[:, tgt_last_edges]

            alpha4 = 0.25
            gamma4 = 2
            neg_cost_this_edge = (1 - alpha4) * (out_this_edge ** gamma4) * (-(1 - out_this_edge + 1e-8).log())
            pos_cost_this_edge = alpha4 * ((1 - out_this_edge) ** gamma4) * (-(out_this_edge + 1e-8).log())
            cost_this_edge = pos_cost_this_edge[:, tgt_this_edges] - neg_cost_this_edge[:, tgt_this_edges]


            alpha5 = 0.25
            gamma5 = 2
            neg_cost_semantic_left_up = (1 - alpha5) * (out_semantic_left_up ** gamma5) * (-(1 - out_semantic_left_up + 1e-8).log())
            pos_cost_semantic_left_up = alpha5 * ((1 - out_semantic_left_up) ** gamma5) * (-(out_semantic_left_up + 1e-8).log())
            cost_semantic_left_up = pos_cost_semantic_left_up[:, tgt_semantic_left_up] - neg_cost_semantic_left_up[:, tgt_semantic_left_up]

            alpha6 = 0.25
            gamma6 = 2
            neg_cost_semantic_right_up = (1 - alpha6) * (out_semantic_right_up ** gamma6) * (-(1 - out_semantic_right_up + 1e-8).log())
            pos_cost_semantic_right_up = alpha6 * ((1 - out_semantic_right_up) ** gamma6) * (-(out_semantic_right_up + 1e-8).log())
            cost_semantic_right_up = pos_cost_semantic_right_up[:, tgt_semantic_right_up] - neg_cost_semantic_right_up[:, tgt_semantic_right_up]

            alpha7 = 0.25
            gamma7 = 2
            neg_cost_semantic_right_down = (1 - alpha7) * (out_semantic_right_down ** gamma7) * (-(1 - out_semantic_right_down + 1e-8).log())
            pos_cost_semantic_right_down = alpha7 * ((1 - out_semantic_right_down) ** gamma7) * (-(out_semantic_right_down + 1e-8).log())
            cost_semantic_right_down = pos_cost_semantic_right_down[:, tgt_semantic_right_down] - neg_cost_semantic_right_down[:, tgt_semantic_right_down]

            alpha8 = 0.25
            gamma8 = 2
            neg_cost_semantic_left_down = (1 - alpha8) * (out_semantic_left_down ** gamma8) * (-(1 - out_semantic_left_down + 1e-8).log())
            pos_cost_semantic_left_down = alpha8 * ((1 - out_semantic_left_down) ** gamma8) * (-(out_semantic_left_down + 1e-8).log())
            cost_semantic_left_down = pos_cost_semantic_left_down[:, tgt_semantic_left_down] - neg_cost_semantic_left_down[:, tgt_semantic_left_down]



            C = self.cost_point * cost_point + self.cost_edge * cost_edge \
                + self.cost_last_edge * cost_last_edge + self.cost_this_edge * cost_this_edge + \
                self.cost_semantic_left_up * cost_semantic_left_up + self.cost_semantic_right_up * cost_semantic_right_up + \
                self.cost_semantic_right_down * cost_semantic_right_down + self.cost_semantic_left_down * cost_semantic_left_down
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["points"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

