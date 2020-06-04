import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class PairwiseRankingLoss(nn.Module):
    def __init__(self, nr, pos_margin=-1, neg_margin=-2, gamma=1,device= 'cpu'):
        super(PairwiseRankingLoss, self).__init__()
        self.mPos = pos_margin
        self.mNeg = neg_margin
        self.gamma = gamma
        self.device = device
        self.nr = nr

    def findPairs(self, logit, target, mask, use_Type = True):
        logit, indices = torch.sort(logit, dim= 0,descending=True)
        target= target[indices]
        mask = mask[indices]

        label = target == 1
        neg_label = target == 0
        ins = int(torch.sum(label).item())
        print(ins)

        pos_logit = logit*mask.float()*label.float()
        pos_logit[pos_logit == 0] = 1
        #pos_logit = pos_logit[torch.nonzero(pos_logit)].flatten()
        pos_val, _ = torch.topk(pos_logit,ins,largest=False)
        #print(torch.max(pos_val))
        #pos_val[pos_val > 9999] = 1

        neg_logit = logit*mask.float()* neg_label.float()
        #neg_logit[neg_logit == 0] = -10000
        neg_val, _ = torch.topk(neg_logit,ins)
        #neg_val[neg_val < -9999] = 0


        return pos_val, neg_val

    def NA_loss(self, logit, target, theta=0.7):

        logit = F.softmax(logit,dim=1)
        #pro = logit
        pro =  (logit>theta )
        logit = logit[:,1:]*pro[:,1:].float()
        # neg_val, ind = torch.max(logit, dim=1)
        # logit[ind == 0, 0] = 1

        BCE = nn.BCELoss(reduction='none')
        loss = torch.sum(BCE(logit, target[:,1:]))
        return loss

    def forward(self, logit, target, mask, theta = 0):
        batch_size = logit.shape[0]
        logit = logit.flatten()
        target = target.flatten()
        #triple_num = torch.sum(mask)
        mask = mask.unsqueeze(dim=-1).repeat(1,1,self.nr-1).flatten()

        # noneOtherInd = target[:, 0] != 1  # not Other index
        # OtherInd = target[:, 0] == 1

        pos, neg = self.findPairs(logit, target, mask)

        part1 = torch.sum( torch.log(1 + torch.exp(self.gamma * (self.mPos - pos))) + torch.log(
            1 + torch.exp(self.gamma * (-100 + pos))))  # positive loss
        print(part1)
        part2 = torch.sum(torch.log(1 + torch.exp(self.gamma * (self.mNeg + neg))) + torch.log(
            1 + torch.exp(self.gamma * (-100 - neg)))) # negative loss
        print(part2)
        #loss = part1 + part2
        # noneOtherInd[noneOtherInd == 0] = theta
        # na_loss = self.NA_loss(logit[OtherInd,], target[OtherInd,])
        loss = torch.sum(part1 + part2) # exclusive other loss
        # loss = torch.sum(loss* mask.float())
        #print(loss)
        # print(na_loss)
        # loss = loss +na_loss
        return loss/len(pos)



