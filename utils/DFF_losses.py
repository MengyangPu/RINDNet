##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Yuan Hu
# Email: huyuan@radi.ac.cn
# Copyright (c) 2019
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Calculate Multi-label Loss (Semantic Loss)"""
import torch
from torch.nn.modules.loss import _Loss

torch_ver = torch.__version__[:3]

__all__ = ['EdgeDetectionReweightedLosses', 'EdgeDetectionReweightedLosses_CPU']


class WeightedCrossEntropyWithLogits(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(WeightedCrossEntropyWithLogits, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs, targets):
        loss_total = 0
        for i in range(targets.size(0)): # iterate for batch size
            pred = inputs[i]
            target = targets[i]

            for j in range(pred.size(0)):
                p = pred[j]
                t = target[j]

                num_pos = torch.sum(t == 1).float()
                num_neg = torch.sum(t == 0).float()
                num_total = num_neg + num_pos  # true total number
                pos_weight = (num_neg / num_pos).clamp(min=1, max=num_total)  # compute a pos_weight for each image

                max_val = (-p).clamp(min=0)
                log_weight = 1 + (pos_weight - 1) * t
                loss = p - p * t + log_weight * (
                            max_val + ((-max_val).exp() + (-p - max_val).exp()).log())

                loss = torch.sum(loss)
                loss_total = loss_total + loss

        loss_total = loss_total / targets.size(0)
        return loss_total

class EdgeDetectionReweightedLosses(WeightedCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedLosses"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLosses, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        pre, target = tuple(inputs)
        side5, fuse = tuple(pre)

        loss_side5 = super(EdgeDetectionReweightedLosses, self).forward(side5, target)
        loss_fuse = super(EdgeDetectionReweightedLosses, self).forward(fuse, target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight

        return loss

class EdgeDetectionReweightedLosses_CPU(WeightedCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedLosses"""
    """CPU version used to dubug"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLosses_CPU, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        pred, target = tuple(inputs)

        loss_side5 = super(EdgeDetectionReweightedLosses_CPU, self).forward(pred[0], target)
        loss_fuse = super(EdgeDetectionReweightedLosses_CPU, self).forward(pred[1], target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight

        return loss


##########
class WeightedCrossEntropyWithLogitsSingle(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(WeightedCrossEntropyWithLogitsSingle, self).__init__(size_average, reduce, reduction)

    def forward(self, inputs, targets):
        loss_total = 0
        for i in range(targets.size(0)): # iterate for batch size
            pred = inputs[i]
            target = targets[i]
            #print(pred.shape, target.shape)

            num_pos = torch.sum(target == 1).float()
            num_neg = torch.sum(target == 0).float()
            num_total = num_neg + num_pos  # true total number
            pos_weight = (num_neg / num_pos).clamp(min=1, max=num_total) # compute a pos_weight for each image

            max_val = (-pred).clamp(min=0)
            log_weight = 1 + (pos_weight - 1) * target
            loss = pred - pred * target + log_weight * (max_val + ((-max_val).exp() + (-pred - max_val).exp()).log())

            loss = loss.sum()
            loss_total = loss_total + loss

        loss_total = loss_total / targets.size(0)
        return loss_total

class EdgeDetectionReweightedLossesSingle(WeightedCrossEntropyWithLogitsSingle):
    """docstring for EdgeDetectionReweightedLosses"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLossesSingle, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        pre, target = tuple(inputs)
        side5, fuse = tuple(pre)

        #print(side5.shape, fuse.shape, target.shape)

        loss_side5 = super(EdgeDetectionReweightedLossesSingle, self).forward(side5, target)
        loss_fuse = super(EdgeDetectionReweightedLossesSingle, self).forward(fuse, target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight

        return loss

class EdgeDetectionReweightedLossesSingle_CPU(WeightedCrossEntropyWithLogits):
    """docstring for EdgeDetectionReweightedLosses"""
    """CPU version used to dubug"""
    def __init__(self, weight=None, side5_weight=1, fuse_weight=1):
        super(EdgeDetectionReweightedLossesSingle_CPU, self).__init__(weight=weight)
        self.side5_weight = side5_weight
        self.fuse_weight = fuse_weight

    def forward(self, *inputs):
        pred, target = tuple(inputs)

        loss_side5 = super(EdgeDetectionReweightedLossesSingle_CPU, self).forward(pred[0], target)
        loss_fuse = super(EdgeDetectionReweightedLossesSingle_CPU, self).forward(pred[1], target)
        loss = loss_side5 * self.side5_weight + loss_fuse * self.fuse_weight

        return loss



if __name__ == '__main__':
    N = 16
    H, W = 320, 320
    label = torch.randint(0, 2, size=(N, 1, H, W)).float()
    output = tuple([torch.rand(N, 1, H, W),torch.rand(N, 1, H, W)])

    crientation = EdgeDetectionReweightedLossesSingle()
    total_loss = crientation(output, label)
    print(total_loss)