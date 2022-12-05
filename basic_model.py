import torch
import torch.nn as nn
import pdb
import torch.nn.functional as f
from torchvision import models

from torch.nn import TransformerEncoder, TransformerEncoderLayer

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        init.xavier_uniform_(m.bias)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.01)
        init.constant_(m.bias, 0)

class ProbabilisticEncoder_alpha(nn.Module):
    def __init__(self, config, feature_dim, num_learnable_token):
        super(ProbabilisticEncoder_alpha, self).__init__()
        # inducing tokens
        self.learnable_tokens = nn.Parameter(
            torch.empty((num_learnable_token, 1, feature_dim), dtype=torch.float32).normal_(0.,0.1),requires_grad=True)

        emsize = feature_dim
        ninp = emsize
        nhead = int(emsize/64)
        # nhead = 4
        nhid = emsize
        dropout = 0.2
        task_encoder_layer = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='gelu', batch_first=True)
        nlayers = 1
        self.transformer_encoder = TransformerEncoder(task_encoder_layer, nlayers)

        sizes = [emsize, emsize, emsize, emsize]
        self.mu_infer = nn.Linear(sizes[-2], sizes[-1])
        self.sigma_infer = nn.Linear(sizes[-2], sizes[-1])

    def forward(self, input):
        context = input
        query = self.learnable_tokens

        # transformer, interactions between tokens
        src = torch.cat([context, query], 1)
        updated_src = self.transformer_encoder(src)
        n_all, n_context = src.shape[1], context.shape[1]
        updated_learnable_tokens = updated_src[:, n_context:, :]  # num_task * D

        # amortization inference
        x = updated_learnable_tokens.squeeze(1)
        mu = self.mu_infer(x)

        sigma = f.softplus(self.sigma_infer(x), beta=1, threshold=20)  # !!! very important
        return mu, sigma

class ProbabilisticEncoder_phi(nn.Module):
    def __init__(self, config, feature_dim, num_learnable_token):
        super(ProbabilisticEncoder_phi, self).__init__()
        self.way_number = config["way_number"]
        self.d_feature = config["d_feature"]

        # inducing tokens
        self.learnable_tokens = nn.Parameter(
            torch.empty((num_learnable_token, 1, feature_dim), dtype=torch.float32).normal_(0., 0.1),
            requires_grad=True)
        emsize = feature_dim

        # traditional transformer ---------------
        ninp = emsize
        nhead = int(emsize / 64)
        # nhead = 4
        nhid = emsize
        dropout = 0.2
        task_encoder_layer = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='gelu', batch_first=True)
        nlayers = 1
        self.transformer_encoder = TransformerEncoder(task_encoder_layer, nlayers)

        sizes = [emsize, emsize, emsize, emsize]
        self.mu_infer = nn.Linear(sizes[-2], sizes[-1])
        self.sigma_infer = nn.Linear(sizes[-2], sizes[-1])

    def forward(self, input, embedding):
        context = input.transpose(0, 1).reshape(self.way_number, -1, self.d_feature)
        query = self.learnable_tokens

        # transformer, interactions between tokens
        src = torch.cat([context, query], 1)
        updated_src = self.transformer_encoder(src)
        n_all, n_context = src.shape[1], context.shape[1]
        updated_learnable_tokens = updated_src[:, n_context:, :]  # num_task * D

        # amortization inference
        x = updated_learnable_tokens + embedding
        # 5 * 1 * 4096 + 10 * 4096 = 5 * 10 * 4096

        mu = self.mu_infer(x)
        sigma = f.softplus(self.sigma_infer(x), beta=1, threshold=20)
        return mu, sigma
