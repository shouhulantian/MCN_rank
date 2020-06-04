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

    # def entity_mention_attention(self, p,is_train):
    #     node_feature, index, edge_weight, mention_count,relation_label = p
    #     #mention_select = torch.eq(edge_weight[index,:mention_count] ,1)
    #     # print('node_feature')
    #     # print(node_feature)
    #     # print('index')
    #     # print(index)
    #     # print('edge_weight')
    #     # print(edge_weight)
    #     # print('mention_count')
    #     # print(mention_count)
    #     # print('relation_label')
    #     # print(relation_label)
    #     if mention_count > 0:
    #         node = torch.cat([self.entity_mention_select(node_feature,edge_weight[j,:mention_count].squeeze(0),relation_label[i],is_train) for i,j in enumerate(index)])
    #         return node.unsqueeze(0)
    #     else:
    #         return torch.index_select(node_feature,0,index).unsqueeze(0)

    # def entity_pair_construct(self,index, node_feature,edge_weight,mention_count):
    #     #node = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(node_feature, index)])
    #     # node = torch.Tensor(node_feature.shape[0],index.shape[1],node_feature.shape[2]).to(self.config.device)
    #     node = torch.cat([self.entity_mention_attention(p) for p in zip(node_feature, index,edge_weight,mention_count)])
    #     # for i in range(node_feature.shape[0]):
    #     	# print('index')
    #     	# print(index[i])
    #     	# print('node')
    #     	# print(node_feature[i])
    #     	#node = torch.index_select(node_feature[i],0,index[i])
    #     return node

    def forward(self, node_feature, edge_weight, index, mention_count,relation_label,is_train):
        node = torch.cat([self.entity_mention_attention(p,is_train) for p in zip(node_feature, index, edge_weight, mention_count,relation_label)])

        return node



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