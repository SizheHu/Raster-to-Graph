import torch
import torch.nn.functional as F
import torchvision
from typing import List
from util.misc import NestedTensor
from util.normalization import FrozenBatchNorm2d
from collections import OrderedDict
from torch import nn
from typing import Dict


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module) -> None:
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
        del layers['avgpool']
        del layers['fc']
        super(IntermediateLayerGetter, self).__init__(layers)

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in {'layer1': '1', 'layer2': '2', 'layer3': '3', 'layer4': '4'}:
                out_name = {'layer1': '1', 'layer2': '2', 'layer3': '3', 'layer4': '4'}[name]
                out[out_name] = x
        return out

class Backbone(nn.Module):
    def __init__(self, name):
        backbone = getattr(torchvision.models, name)(pretrained=True, norm_layer=FrozenBatchNorm2d)
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if ('layer1' not in name) and ('layer2' not in name) and ('layer3' not in name) and ('layer4' not in name):
                parameter.requires_grad_(False)

        self.num_channels = [256, 512, 1024, 2048]
        self.body = IntermediateLayerGetter(backbone)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            mask = F.interpolate(tensor_list.mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        backbone = self[0]
        position_embedding = self[1]
        xs = backbone(tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)
        for x in out:
            pos.append(position_embedding(x).to(x.tensors.dtype))
        return out, pos
