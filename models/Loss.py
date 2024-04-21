
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.functional import cross_entropy, one_hot, softmax

def CalculateMean(features, labels, class_num):
    N = features.size(0)
    C = class_num
    A = features.size(1)

    avg_CxA = torch.zeros(C, A).cuda()
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
    onehot = torch.zeros(N, C).cuda()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1.0

    for c in range(class_num):
        c_temp = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        c_temp = torch.sum(c_temp, dim=0)
        avg_CxA[c] = c_temp / Amount_CxA[c]
    return avg_CxA.detach()

def Calculate_CV(features, labels, ave_CxA, class_num):
    N = features.size(0)
    C = class_num
    A = features.size(1)

    var_temp = torch.zeros(C, A, A).cuda()
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

    onehot = torch.zeros(N, C).cuda()
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1
    Amount_CxAxA = Amount_CxA.view(C, A, 1).expand(C, A, A)
    avg_NxCxA = ave_CxA.expand(N, C, A)
    for c in range(C):
        features_by_sort_c = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        avg_by_sort_c = avg_NxCxA[:, c, :].mul(NxCxA_onehot[:, c, :])
        var_temp_c = features_by_sort_c - avg_by_sort_c
        var_temp[c] = torch.mm(var_temp_c.permute(1,0), var_temp_c).div(Amount_CxAxA[c])
    return var_temp.detach()

class cls_Loss(torch.nn.Module):
    def __init__(self, num_classes=3,
                 p=0.8, q=2.0, eps=1e-2):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.q = q
        self.eps = eps
        self.register_buffer('accumulated',
                             torch.zeros(self.num_classes, dtype=torch.float))

    def forward(self, outputs, targets):
        for unique in targets.unique():
            self.accumulated[unique] += (targets == unique.item()).sum()

        onehot_targets = one_hot(targets, self.num_classes)
        seesaw_weights = outputs.new_ones(onehot_targets.size())
        if self.p > 0:
            matrix = self.accumulated[None, :].clamp(min=1) / self.accumulated[:, None].clamp(min=1)
            index = (matrix < 1.0).float()
            sample_weights = matrix.pow(self.p) * index + (1 - index)
            mitigation_factor = sample_weights[targets.long(), :]
            seesaw_weights = seesaw_weights * mitigation_factor

        if self.q > 0:
            scores = softmax(outputs.detach(), dim=1)
            self_scores = scores[torch.arange(0, len(scores)).to(scores.device).long(), targets.long()]
            score_matrix = scores / self_scores[:, None].clamp(min=self.eps)
            index = (score_matrix > 1.0).float()
            compensation_factor = score_matrix.pow(self.q) * index + (1 - index)
            seesaw_weights = seesaw_weights * compensation_factor

        outputs = outputs + (seesaw_weights.log() * (1 - onehot_targets))

        return cross_entropy(outputs, targets)

class RSCM_Loss(nn.Module):
    def __init__(self, class_num):
        super(RSCM_Loss, self).__init__()
        self.class_num = class_num
        self.cross_entropy = nn.CrossEntropyLoss()
        self.class_unequal = cls_Loss(num_classes=class_num)

    def aug(self, s_mean_matrix, t_mean_matrix, fc, features, y_s, labels_s, t_cv_matrix, Lambda):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]
        NxW_ij = weight_m.expand(N, C, A)
        NxW_kj = torch.gather(NxW_ij, 1, labels_s.view(N, 1, 1).expand(N, C, A))

        t_CV_temp = t_cv_matrix[labels_s]

        sigma2 = Lambda * torch.bmm(torch.bmm(NxW_ij - NxW_kj, t_CV_temp), (NxW_ij - NxW_kj).permute(0, 2, 1))
        sigma2 = sigma2.mul(torch.eye(C).cuda().expand(N, C, C)).sum(2).view(N, C)

        sourceMean_NxA = s_mean_matrix[labels_s]
        targetMean_NxA = t_mean_matrix[labels_s]
        dataMean_NxA = (targetMean_NxA - sourceMean_NxA)
        dataMean_NxAx1 = dataMean_NxA.expand(1, N, A).permute(1, 2, 0)
        dataW_NxCxA = NxW_ij - NxW_kj
        
        dataW_x_detaMean_NxCx1 = torch.bmm(dataW_NxCxA, dataMean_NxAx1)
        datW_x_detaMean_NxC = dataW_x_detaMean_NxCx1.view(N, C)
        aug_result = y_s + 0.5 * sigma2 + Lambda * datW_x_detaMean_NxC
        return aug_result

    def forward(self, fc, features_source: torch.Tensor, y_s, labels_source, Lambda, mean_source, mean_target, covariance_target):
        aug_y = self.aug(mean_source, mean_target, fc, features_source, y_s, labels_source, covariance_target, Lambda)
        loss = self.class_unequal(aug_y, labels_source)
        return loss

class CFAMLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(CFAMLoss, self).__init__()

        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda else torch.device('cpu'))
        features_mean = features.mean(dim=1, keepdim=True)
        features = features - features_mean

        features = F.normalize(features, p=2, dim=1)

        class_centers = []
        for c in torch.unique(labels):
            class_features = features[labels == c]
            class_center = torch.mean(class_features, dim=0)
            class_centers.append(class_center)
        class_centers = torch.stack(class_centers)
        dot_prod_centers = torch.matmul(class_centers, class_centers.t())
        eye = torch.eye(dot_prod_centers.shape[0], dot_prod_centers.shape[1]).bool().to(device)
        mask_neg = (~eye).float()
        neg_pairs_mean1 = (mask_neg * dot_prod_centers).sum() / (mask_neg.sum() + 1e-6)
        labels = labels[:, None] 
        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device) 

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / \
            (mask_neg.sum() + 1e-6)  
        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean
        loss = 0.1 * neg_pairs_mean1  
        return loss

def entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-10
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def adv(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(features.device)
    return torch.nn.BCELoss()(ad_out, dc_target)

def adv_local(features, ad_net, is_source=False, weights=None):
    ad_out = ad_net(features).squeeze(3)
    batch_size = ad_out.size(0)
    num_heads = ad_out.size(1)
    seq_len = ad_out.size(2)
    
    if is_source:
        label = torch.from_numpy(np.array([[[1]*seq_len]*num_heads] * batch_size)).float().to(features.device)
    else:
        label = torch.from_numpy(np.array([[[0]*seq_len]*num_heads] * batch_size)).float().to(features.device)

    return ad_out, torch.nn.BCELoss()(ad_out, label)


