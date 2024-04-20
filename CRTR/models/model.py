# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
from os.path import join as pjoin
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from models.Loss import *
logger = logging.getLogger(__name__)

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(
            config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(
            config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(
            config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, posi_emb=None, ad_net=None, is_source=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        if posi_emb is not None:
            eps = 1e-10
            batch_size = key_layer.size(0)  

            patch = key_layer
            ad_out, loss_ad = adv_local(
                patch[:, :, 1:], ad_net, is_source)
            entropy = - ad_out * \
                torch.log2(ad_out + eps) - (1.0 - ad_out) * \
                torch.log2(1.0 - ad_out + eps)
            entropy = torch.cat((torch.ones(batch_size, self.num_attention_heads, 1).to(
                hidden_states.device).float(), entropy), 2)
            trans_ability = entropy if self.vis else None   
            entropy = entropy.view(
                batch_size, self.num_attention_heads, 1, -1)  
            attention_probs = torch.cat((attention_probs[:, :, 0, :].unsqueeze(
                2) * entropy, attention_probs[:, :, 1:, :]), 2)

        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        if posi_emb is not None:
            return attention_output, loss_ad, weights, trans_ability
        else:
            return attention_output, weights
# MSFFM
class Attention2(nn.Module):
    def __init__(self, config, vis):
        super(Attention2, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(
            config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(
            config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(
            config.transformer["attention_dropout_rate"])
        self.conv1 = nn.Conv2d(config.hidden_size,config.hidden_size,kernel_size=1,stride=1,padding=0,groups = config.hidden_size,bias=False)
        self.conv2 = nn.Conv2d(config.hidden_size,config.hidden_size,kernel_size=3,stride=1,padding=0,groups = config.hidden_size,bias=False)
        self.conv3 = nn.Conv2d(config.hidden_size,config.hidden_size,kernel_size=5,stride=1,padding=0,groups = config.hidden_size,bias=False)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        cls_token, x = torch.split(hidden_states,[1,256],dim=1)
        x = x.transpose(1,2).view(x.shape[0],768,16,16)
        x1,x2,x3 = self.conv1(x).flatten(2).transpose(1,2),self.conv2(x).flatten(2).transpose(1,2),self.conv3(x).flatten(2).transpose(1,2)
        x1,x2,x3 = torch.cat([cls_token,x1],dim=1),torch.cat([cls_token,x2],dim=1),torch.cat([cls_token,x3],dim=1)

        mixed_key_layer1,mixed_key_layer2,mixed_key_layer3 = self.key(x1),self.key(x2),self.key(x3)
        mixed_value_layer1,mixed_value_layer2,mixed_value_layer3 = self.value(x1),self.value(x2),self.value(x3)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer1,key_layer2,key_layer3 = self.transpose_for_scores(mixed_key_layer1),self.transpose_for_scores(mixed_key_layer2),self.transpose_for_scores(mixed_key_layer3)
        value_layer1,value_layer2,value_layer3 = self.transpose_for_scores(mixed_value_layer1),self.transpose_for_scores(mixed_value_layer2),\
            self.transpose_for_scores(mixed_value_layer3)
        attention_scores1,attention_scores2,attention_scores3 = torch.matmul(query_layer, key_layer1.transpose(-1, -2)),\
            torch.matmul(query_layer, key_layer2.transpose(-1, -2)),torch.matmul(query_layer, key_layer3.transpose(-1, -2))
        attention_scores1,attention_scores2,attention_scores3 = attention_scores1 / math.sqrt(self.attention_head_size),attention_scores2 / math.sqrt(self.attention_head_size),\
            attention_scores3 / math.sqrt(self.attention_head_size)

        attention_probs1,attention_probs2,attention_probs3 = self.softmax(attention_scores1),self.softmax(attention_scores2),self.softmax(attention_scores3)
        weights = attention_probs1 if self.vis else None
        attention_probs1,attention_probs2,attention_probs3 = self.attn_dropout(attention_probs1),self.attn_dropout(attention_probs2),self.attn_dropout(attention_probs3)
        context_layer1,context_layer2,context_layer3 = torch.matmul(attention_probs1, value_layer1),torch.matmul(attention_probs2, value_layer2),\
            torch.matmul(attention_probs3, value_layer3)
        context_layer = (context_layer1+context_layer2+context_layer3)/3
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0],
                          img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * \
                (img_size[1] // patch_size[1])
            self.hybrid = False

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, self.position_embeddings

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)



class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs',
                 wo_dp_conv=False, dp_first=False):

        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3
        layers = []
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)])
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config, vis)
        self.conv = LocalityFeedForward(768, 768, 1, 4, 'hs', wo_dp_conv=False, dp_first=False)

    def forward(self, x, posi_emb=None, ad_net=None, is_source=False):
        h = x
        x = self.attention_norm(x)
        if posi_emb is not None:
            x, loss_ad, weights, tran_weights = self.attn(
                x, posi_emb, ad_net, is_source)
        else:
            x, weights = self.attn(x)
        x = x + h
        cls_token, x = torch.split(x, [1, 256], dim=1)
        x = x.transpose(1, 2).view(x.shape[0], 768, 16, 16)
        x = self.conv(x).flatten(2).transpose(1, 2)                              
        x = torch.cat([cls_token, x], dim=1)
        if posi_emb is not None:
            return x, loss_ad, weights, tran_weights
        else:
            return x, weights
class Block2(nn.Module):
    def __init__(self, config, vis):
        super(Block2, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention2(config, vis)
        self.conv = LocalityFeedForward(768, 768, 1, 4, 'hs', wo_dp_conv=False, dp_first=False)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        cls_token, x = torch.split(x, [1, 256], dim=1)
        x = x.transpose(1, 2).view(x.shape[0], 768, 16, 16)
        x = self.conv(x).flatten(2).transpose(1, 2)                              
        x = torch.cat([cls_token, x], dim=1)
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis, msa_layer):
        super(Encoder, self).__init__()
        self.vis = vis
        self.msa_layer = msa_layer
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            if i == 0 or i==2 :
                layer = Block2(config, vis)
            else:
                layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, posi_emb, ad_net, is_source=False):
        attn_weights = []
        for i, layer_block in enumerate(self.layer):
            if i == (self.msa_layer-1):
                hidden_states, loss_ad, weights, tran_weights = layer_block(
                    hidden_states, posi_emb, ad_net, is_source)
            else:
                hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, loss_ad, attn_weights, tran_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, msa_layer):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, msa_layer)

    def forward(self, input_ids, ad_net, is_source=False):
        embedding_output, posi_emb = self.embeddings(input_ids)
        encoded, loss_ad, attn_weights, tran_weights = self.encoder(
            embedding_output, posi_emb, ad_net, is_source)
        return encoded, loss_ad, attn_weights, tran_weights

class LocalViT(nn.Module):
    def __init__(self, config, img_size=224, num_classes=3, zero_head=False, vis=False, msa_layer=12):
        super(LocalViT, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.criterion = nn.MSELoss()
        self.transformer = Transformer(config, img_size, vis, msa_layer)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x_s, x_t=None, ad_net=None):
        x_s, loss_ad_s, attn_s, tran_s = self.transformer(
            x_s, ad_net, is_source=True)
        feature_s = x_s[:, 0]
        logits_s = self.head(x_s[:, 0])
        if x_t is not None:
            x_t, loss_ad_t, _, _ = self.transformer(x_t, ad_net)
            feature_t = x_t[:, 0]
            logits_t = self.head(x_t[:, 0])
            return feature_s, feature_t, logits_s, logits_t, (loss_ad_s + loss_ad_t) / 2.0, x_s, x_t
        else:
            return feature_s, logits_s, attn_s, tran_s



def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high,
                           self.low, self.alpha, self.max_iter)
        x = x * 1.0
        if self.training and x.requires_grad:
            x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]
