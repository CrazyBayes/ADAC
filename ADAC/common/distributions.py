import torch
import numpy as np
from torch.distributions.kl import register_kl
from torch.distributions.relaxed_categorical import ExpRelaxedCategorical

@register_kl(ExpRelaxedCategorical, ExpRelaxedCategorical)
def kl_relaxedcat_relaxedcat(p, q):
    a0 = p.logits - torch.max(p.logits, dim=1, keepdim=True)[0]
    a1 = q.logits - torch.max(q.logits, dim=1, keepdim=True)[0]
    ea0 = torch.exp(a0)
    ea1 = torch.exp(a1)
    z0 = torch.sum(ea0, dim=1, keepdim=True)
    z1 = torch.sum(ea1, dim=1, keepdim=True)
    p0 = ea0 / z0
    return torch.sum(p0 * (a0 - torch.log(z0) - a1 + torch.log(z1)), dim=1)


def truncated_normal_(tensor, mean=0.0, std=1.0, threshold=2.0):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < threshold) & (tmp > -threshold)
    ind = valid.max(-1, keepdim=True)[1]
    # tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor = tmp.gather(-1, ind).squeeze(-1)
    tensor.data.mul_(std).add_(mean)
    return tensor


def truncated_normal(size, mean=0.0, std=1.0, threshold=2.0):
    tmp = np.random.randn(*(size + (4,)))
    valid = (tmp < threshold) & (tmp > -threshold)
    ind = np.argmax(valid, axis=-1)
    # https://stackoverflow.com/questions/32089973/numpy-index-3d-array-with-index-of-last-axis-stored-in-2d-array
    tensor = ind.T.choose(tmp.T).T
    return (tensor * std + mean).tolist()

