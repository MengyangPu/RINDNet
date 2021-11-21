from typing import Tuple
from torch import nn, Tensor
import torch
from torch import nn
import torch.nn.functional as F

def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[:,1,:,:].unsqueeze(1))
    y_pred_neg = torch.cat([y_pred_neg, zeros], axis=1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], axis=1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss

class LabelCircleLossModel(nn.Module):
    def __init__(self, num_classes, m=0.35, gamma=30, feature_dim=192):
        super(LabelCircleLossModel, self).__init__()
        self.margin = m
        self.gamma = gamma
        self.weight = torch.nn.Parameter(torch.randn(feature_dim, num_classes, requires_grad=True))
        self.labels = torch.tensor([x for x in range(num_classes)])
        self.classes = num_classes
        self.init_weights()
        self.O_p = 1 + self.margin
        self.O_n = -self.margin
        self.Delta_p = 1 - self.margin
        self.Delta_n = self.margin
        self.loss = nn.CrossEntropyLoss()
    def init_weights(self, pretrained=None):
        self.weight.data.normal_()

    def _forward_train(self, feat, label):
        normed_feat = torch.nn.functional.normalize(feat)
        normed_weight = torch.nn.functional.normalize(self.weight,dim=0)

        bs = label.size(0)
        mask = label.expand(self.classes, bs).t().eq(self.labels.expand(bs,self.classes)).float()
        y_true = torch.zeros((bs,self.classes)).scatter_(1,label.view(-1,1),1)
        y_pred = torch.mm(normed_feat,normed_weight)
        y_pred = y_pred.clamp(-1,1)
        sp = y_pred[mask == 1]
        sn = y_pred[mask == 0]

        alpha_p = (self.O_p - y_pred.detach()).clamp(min=0)
        alpha_n = (y_pred.detach() - self.O_n).clamp(min=0)

        y_pred = (y_true * (alpha_p * (y_pred - self.Delta_p)) +
                    (1-y_true) * (alpha_n * (y_pred - self.Delta_n))) * self.gamma
        loss = self.loss(y_pred,label)

        return loss, sp, sn

    def forward(self, input, label,  mode='train'):
            if mode == 'train':
                return self._forward_train(input, label)
            elif mode == 'val':
                raise KeyError

class CircleLoss2(nn.Module):
    def __init__(self, scale=32, margin=0.25, similarity='cos', **kwargs):
        super(CircleLoss2, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

        m = labels.size(0)
        mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)
        if self.similarity == 'dot':
            sim_mat = torch.matmul(feats, torch.t(feats))
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            sim_mat = feats.mm(feats.t())
        else:
            raise ValueError('This similarity is not implemented.')

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]

        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss

def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


if __name__ == "__main__":
    batch_size = 1
    nclass = 2
    w = 3
    h = 3
    label = torch.randint(0, 2, size=(batch_size, nclass, w, h)).float()
    y_pred = torch.rand(batch_size, nclass, w, h)
    loss=multilabel_categorical_crossentropy(label,y_pred)
    print(loss)

    circleloss = LabelCircleLossModel(4)
    print(circleloss(y_pred, label))

    batch_size = 10
    feats = torch.rand(batch_size, 4, 1028)
    labels = torch.randint(high=10, dtype=torch.long, size=(batch_size,4))
    circleloss = CircleLoss2(similarity='cos')
    print(circleloss(feats, labels))

    feat = nn.functional.normalize(torch.rand(4,5,10,10, requires_grad=True))
    lbl = torch.randint(high=5, size=(4,5,10,10))

    inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)

    criterion = CircleLoss(m=0.25, gamma=10)
    circle_loss = criterion(inp_sp, inp_sn)

    print(circle_loss)