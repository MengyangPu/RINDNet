import torch
import torch.nn as nn


def weighted_cross_entropy_loss(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    total_loss = 0
    batch, channel_num, imh, imw = edges.shape
    for b_i in range(batch):
        p = preds[b_i, :, :, :].unsqueeze(1)
        t = edges[b_i, :, :, :].unsqueeze(1)
        mask = (t > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        weight = torch.zeros_like(mask)
        weight[t > 0.5] = num_neg / (num_pos + num_neg)
        weight[t <= 0.5] = num_pos / (num_pos + num_neg)
        # Calculate loss.
        loss = torch.nn.functional.binary_cross_entropy(p.float(), t.float(), weight=weight, reduction='none')
        loss = torch.sum(loss)
        total_loss = total_loss + loss
    return total_loss

def weighted_cross_entropy_loss2(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    num_pos = torch.sum(edges == 1).float()
    num_neg = torch.sum(edges == 0).float()
    weight = torch.zeros_like(edges)
    mask = (edges > 0.5).float()
    weight[edges > 0.5] = num_neg / (num_pos + num_neg)
    weight[edges <= 0.5] = num_pos / (num_pos + num_neg)
    # Calculate loss.
    loss = torch.nn.functional.binary_cross_entropy(preds, edges, weight=weight, reduction='none')
    loss = torch.sum(loss)
    return loss

def weighted_cross_entropy_loss3(preds, edges):
    """ Calculate sum of weighted cross entropy loss. """
    # Reference:
    #   hed/src/caffe/layers/sigmoid_cross_entropy_loss_layer.cpp
    #   https://github.com/s9xie/hed/issues/7
    total_loss = 0
    batch, channel_num, imh, imw = edges.shape
    for b_i in range(batch):
        p = preds[b_i, :, :, :].unsqueeze(0)
        t = edges[b_i, :, :, :].unsqueeze(0)
        mask = (t > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        weight = torch.zeros_like(mask)
        weight[t > 0.5] = num_neg / (num_pos + num_neg)
        weight[t <= 0.5] = num_pos / (num_pos + num_neg)
        # Calculate loss.
        loss = torch.nn.functional.binary_cross_entropy(p.float(), t.float(), weight=weight, reduction='none')
        loss = torch.sum(loss)
        total_loss = total_loss + loss
    return total_loss


class HED_Loss(nn.Module):
    def __init__(self):
        super(HED_Loss, self).__init__()

    def forward(self,output,label):#,depth_gad):
        """
        output: [N,4,H,W]
        label: [N,4,H,W]
        """
        total_loss = 0
        b,c,w,h = label.shape
        for j in range(c):
            p = output[:, j, :, :].unsqueeze(1)
            t = label[:, j, :, :].unsqueeze(1)
            loss = weighted_cross_entropy_loss(p, t)
            total_loss = total_loss + loss

        total_loss=total_loss/b*1.0
        return total_loss

class HED_Loss2(nn.Module):
    def __init__(self):
        super(HED_Loss2, self).__init__()

    def forward(self,output,label):#,depth_gad):
        """
        output: [N,4,H,W]
        label: [N,4,H,W]
        """
        b, c, w, h = label.shape
        loss = weighted_cross_entropy_loss2(output, label)
        total_loss = loss / b * 1.0
        return total_loss

class HED_Loss3(nn.Module):
    def __init__(self):
        super(HED_Loss3, self).__init__()

    def forward(self,output,label):#,depth_gad):
        """
        output: [N,4,H,W]
        label: [N,4,H,W]
        """
        b, c, w, h = label.shape
        loss = weighted_cross_entropy_loss3(output, label)
        total_loss = loss / b * 1.0
        return total_loss

if __name__ == '__main__':
    N = 8
    H, W = 320, 320
    label = torch.randint(0, 2, size=(N, 4, H, W)).float()
    o_b = [torch.rand(N, 4, H, W), torch.rand(N, 4, H, W), torch.rand(N, 4, H, W), torch.rand(N, 4, H, W)]
    criterion = HED_Loss()
    total_loss = sum([criterion(o, label) for o in o_b])
    print(total_loss)

    criterion = HED_Loss3()
    total_loss = sum([criterion(o, label) for o in o_b])
    print(total_loss)