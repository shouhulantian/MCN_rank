import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self,embedding_dim, relation_size):
        super(Attention,self).__init__()
        self.embedding_dim = embedding_dim
        self.relation_size = relation_size

        self.relation_matrix = nn.Embedding(self.relation_size, self.embedding_dim)
        self.bias = nn.Parameter(torch.Tensor(self.relation_size))

        nn.init.xavier_uniform_(self.relation_matrix.weight.data)
        nn.init.normal_(self.bias)

    def key_request(self,label):
        return self.relation_matrix(label)

    def entity_mention_select(self,node,edge,relation_label,is_training = True):
        men_list = (edge == 1).nonzero().squeeze(-1)
        n = torch.index_select(node,0,men_list)
        #relation_label[torch.eq(relation_label, -1)] = 0
        if is_training:
            q_result = self.relation_matrix(relation_label).unsqueeze(0)
            att_weight = F.softmax(torch.matmul(n,torch.t(q_result)),dim=0)
            att_hidden = torch.matmul(torch.transpose(att_weight,0,1),n)
        else:
            att_weight = F.softmax(torch.matmul(n,torch.t(self.relation_matrix.weight)), dim =0)
            att_hidden = torch.matmul(torch.transpose(att_weight,0,1),n)
        return att_hidden.unsqueeze(0)


    def forward(self, bert_hidden,gcn_hidden,relation_label,is_train):
        batch_size = bert_hidden.shape[0]
        hidden_size = bert_hidden.shape[-1]
        bert_hidden = bert_hidden.view(-1, hidden_size)
        gcn_hidden = gcn_hidden.view(-1, hidden_size)
        relation_label =relation_label.view(batch_size*relation_label.shape[1])
        bag_hidden = torch.cat([torch.cat([a.unsqueeze(0),b.unsqueeze(0)],dim=0).unsqueeze(0)for a,b in zip(bert_hidden,gcn_hidden)])

        if is_train:
            q_result = self.relation_matrix(relation_label).unsqueeze(1)
            att_weight = F.softmax(torch.bmm(bag_hidden,torch.transpose(q_result,1,2)),dim=1)
            att_hidden = torch.bmm(torch.transpose(att_weight,1,2),bag_hidden).view(batch_size,-1,hidden_size)
            predict_re = torch.matmul(att_hidden,torch.t(self.relation_matrix.weight))
        else:
            att_weight = F.softmax(torch.matmul(bag_hidden,torch.t(self.relation_matrix.weight)), dim =1)
            att_hidden = torch.bmm(torch.transpose(att_weight,1,2),bag_hidden).view(batch_size,-1,self.relation_size,hidden_size)
            predict_re = torch.matmul(att_hidden, torch.t(self.relation_matrix.weight))
            predict_re = torch.diagonal(predict_re, dim1=2, dim2=3)
        return predict_re



    # def forward(self, hidden,length,label,is_train = True):
    #     if is_train:
    #         q_result = self.relation_matrix(label)
    #         q_request = q_result.repeat(1,length[0],1).reshape(hidden.size())
    #         att_weight = F.softmax(torch.sum(hidden*q_request,dim=2,keepdim=True),dim=1)
    #         att_hidden = torch.bmm(torch.transpose(hidden,1,2), att_weight).squeeze(dim = 2)
    #         att_score = torch.matmul(att_hidden,torch.t(self.relation_matrix.weight)) + self.bias
    #         score = F.softmax(att_score,dim=1)
    #         return score
    #     else:
    #         q_request = torch.t(self.relation_matrix.weight).repeat(len(label),1,1)
    #         att_weight = F.softmax(torch.bmm(hidden,q_request), dim =1)
    #         att_hidden = torch.bmm(torch.transpose(att_weight,1,2),hidden)
    #         att_score = torch.matmul(att_hidden,torch.t(self.relation_matrix.weight))+self.bias
    #         score = F.softmax(torch.diagonal(att_score, dim1=1,dim2=2),dim=1)
    #         return score