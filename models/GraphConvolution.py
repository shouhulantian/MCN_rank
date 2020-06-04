import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class  GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def normalize(self,mx):
        r_mat_inv = torch.Tensor(mx.shape[0],mx.shape[1],mx.shape[1])
        rowsum = mx.sum(2)
        r_inv = torch.pow(rowsum, -1)
        r_inv[torch.isinf(r_inv)] = 0.
        for i in range(mx.shape[0]):
            r_mat_inv[i] = torch.diag(r_inv[i])
        mx = torch.matmul(r_mat_inv,(mx))
        return mx

    def forward(self, input, adj):
        #input = self.normalize(input,adj)
        # adj = self.normalize(adj)

        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        denom = adj.sum(2).unsqueeze(2)
        denom[torch.eq(denom,0)] = 1.
        output = output/denom  + input
        output[torch.isinf(output)] = 0.
        output[torch.isnan(output)] = 0.
        if self.bias is not None:
            output = output + self.bias
        return F.relu(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
