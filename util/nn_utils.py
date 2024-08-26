import copy

import numpy as np
import torch
import math

from torch import nn
import torch.nn.functional as F

def sigmoid_focal_loss(inputs, targets, num_points, alpha, gamma):
    # print(inputs.shape) # bs, 500, 17 model outputs
    # print(targets.shape) # bs, 500, 17 padded gts
    # print(targets) # bs, 500, 17 padded gts
    # print(num_points) # in most cases, < 10

    prob = inputs.sigmoid()

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # print(ce_loss)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_points # divided by (500 * num_points)
    # return loss.mean(1).sum() / 1 # divided by (500)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
