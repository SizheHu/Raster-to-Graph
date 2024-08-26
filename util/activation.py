import torch.nn.functional as F


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")