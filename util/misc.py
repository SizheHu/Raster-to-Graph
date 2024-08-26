# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Misc functions, including distributed helpers.
Mostly ctrl-c&v from torchvision references.
"""
import os
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
if not torchvision.__version__[3].isdigit():
    if float(torchvision.__version__[:3]) < 0.5:
        import math
        from torchvision.ops.misc import _NewEmptyTensorOp
        def _check_size_scale_factor(dim, size, scale_factor):
            # type: (int, Optional[List[int]], Optional[float]) -> None
            if size is None and scale_factor is None:
                raise ValueError("either size or scale_factor should be defined")
            if size is not None and scale_factor is not None:
                raise ValueError("only one of size or scale_factor should be defined")
            if not (scale_factor is not None and len(scale_factor) != dim):
                raise ValueError(
                    "scale_factor shape must match input shape. "
                    "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
                )
        def _output_size(dim, input, size, scale_factor):
            # type: (int, Tensor, Optional[List[int]], Optional[float]) -> List[int]
            assert dim == 2
            _check_size_scale_factor(dim, size, scale_factor)
            if size is not None:
                return size
            # if dim is not 2 or scale_factor is iterable use _ntuple instead of concat
            assert scale_factor is not None and isinstance(scale_factor, (int, float))
            scale_factors = [scale_factor, scale_factor]
            # math.floor might return float in py2.7
            return [
                int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
            ]
    elif float(torchvision.__version__[:3]) < 0.7:
        from torchvision.ops import _new_empty_tensor
        from torchvision.ops.misc import _output_size


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if not torchvision.__version__[3].isdigit():
        if float(torchvision.__version__[:3]) < 0.7:
            if input.numel() > 0:
                return torch.nn.functional.interpolate(
                    input, size, scale_factor, mode, align_corners
                )

            output_shape = _output_size(2, input, size, scale_factor)
            output_shape = list(input.shape[:-2]) + list(output_shape)
            if float(torchvision.__version__[:3]) < 0.5:
                return _NewEmptyTensorOp.apply(input, output_shape)
            return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

