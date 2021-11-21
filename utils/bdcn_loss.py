import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss2d(inputs, targets, cuda=False, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
    #loss = nn.BCELoss(weights, size_average=False)(inputs, targets)
    loss = nn.BCELoss(weights, reduction='sum')(inputs, targets)
    return loss

def bdcn_loss(prediction, label, cuda=False, balance=1.1):
    total_loss = 0
    b, c, w, h = label.shape
    for j in range(c):
        p = prediction[:, j, :, :].unsqueeze(1)
        l = label[:, j, :, :].unsqueeze(1)
        loss = cross_entropy_loss2d(p, l, cuda, balance)
        total_loss = total_loss + loss

    total_loss = total_loss / b * 1.0
    return total_loss

def bdcn_loss_edge(prediction, label, cuda=False, balance=1.1):
    b, c, w, h = label.shape
    loss = cross_entropy_loss2d(prediction, label, cuda, balance)
    total_loss = loss / b * 1.0
    return total_loss

if __name__ == '__main__':
    N = 16
    H, W = 320, 320
    label = torch.randint(0, 2, size=(N, 1, H, W)).float()
    o_b = torch.rand(N, 1, H, W)

    loss=bdcn_loss_edge(o_b,label)
    print(loss)