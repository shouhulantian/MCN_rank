# coding: utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import datetime
import json
import sys
import sklearn.metrics
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import torch.nn.functional as F
from models import Ranking_Loss as pa

IGNORE_INDEX = 0
is_transformer = False

class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


class RankConfig(object):
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()
        self.data_path = './prepro_data'
        self.use_bag = False
        self.use_gpu = True
        self.is_training = True
        self.max_length = 512
        self.pos_num = 2 * self.max_length
        self.entity_num = self.max_length
        self.relation_num = 97

        self.coref_size = 20
        self.entity_type_size = 20
        self.node_type_size = 20
        self.max_epoch = 20
        self.opt_method = 'Adam'
        self.optimizer = None

        self.checkpoint_dir = './checkpoint'
        self.fig_result_dir = './fig_result'
        self.test_epoch = 1
        #self.pretrain_model = 'checkpoint/checkpoint_GCN_bert_coref_dis_2layer'
        self.pretrain_model = 'checkpoint/checkpoint_GCN_bert_ematt'
        self.word_size = 100
        self.epoch_range = None
        self.cnn_drop_prob = 0.5  # for cnn
        self.keep_prob = 0.8  # for lstm

        self.period = 40

        self.batch_size = 12
        self.h_t_limit = 1800
        self.lr = 2e-6

        self.test_batch_size = 20
        self.test_relation_limit = 1800
        self.char_limit = 16
        self.sent_limit = 25
        self.dis2idx = np.zeros((512), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 10

        self.train_prefix = args.train_prefix
        self.test_prefix = args.test_prefix


        if not os.path.exists("log"):
            os.mkdir("log")

    def set_data_path(self, data_path):
        self.data_path = data_path

    def set_max_length(self, max_length):
        self.max_length = max_length
        self.pos_num = 2 * self.max_length

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes

    def set_window_size(self, window_size):
        self.window_size = window_size

    def set_word_size(self, word_size):
        self.word_size = word_size

    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_drop_prob(self, drop_prob):
        self.drop_prob = drop_prob

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def set_test_epoch(self, test_epoch):
        self.test_epoch = test_epoch

    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model

    def set_is_training(self, is_training):
        self.is_training = is_training

    def set_use_bag(self, use_bag):
        self.use_bag = use_bag

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_epoch_range(self, epoch_range):
        self.epoch_range = epoch_range

    def load_train_data(self):
        print("Reading training data...")
        prefix = self.train_prefix

        print('train', prefix)
        self.data_train_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))
        self.data_train_pos = np.load(os.path.join(self.data_path, prefix + '_pos.npy'))
        self.data_train_ner = np.load(os.path.join(self.data_path, prefix + '_ner.npy'))
        self.data_train_char = np.load(os.path.join(self.data_path, prefix + '_char.npy'))
        self.train_file = json.load(open(os.path.join(self.data_path, prefix + '.json')))
        self.data_train_len = np.load(os.path.join(self.data_path, prefix +'_len.npy'))

        self.data_train_bert_word = np.load(os.path.join(self.data_path, prefix+'_bert_word.npy'))
        self.data_train_bert_mask = np.load(os.path.join(self.data_path, prefix+'_bert_mask.npy'))
        self.data_train_bert_starts = np.load(os.path.join(self.data_path, prefix+'_bert_starts.npy'))

        print("Finish reading")

        self.train_len = ins_num = self.data_train_word.shape[0]
        assert (self.train_len == len(self.train_file))

        self.train_order = list(range(ins_num))
        self.train_batches = ins_num // self.batch_size
        if ins_num % self.batch_size != 0:
            self.train_batches += 1

    def load_test_data(self):
        print("Reading testing data...")
        self.data_word_vec = np.load(os.path.join(self.data_path, 'vec.npy'))
        self.data_char_vec = np.load(os.path.join(self.data_path, 'char_vec.npy'))
        self.rel2id = json.load(open(os.path.join(self.data_path, 'rel2id.json')))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        prefix = self.test_prefix
        print(prefix)
        self.is_test = ('dev_test' == prefix)
        self.data_test_word = np.load(os.path.join(self.data_path, prefix + '_word.npy'))
        self.data_test_pos = np.load(os.path.join(self.data_path, prefix + '_pos.npy'))
        self.data_test_ner = np.load(os.path.join(self.data_path, prefix + '_ner.npy'))
        self.data_test_char = np.load(os.path.join(self.data_path, prefix + '_char.npy'))
        self.data_test_len = np.load(os.path.join(self.data_path, prefix + '_len.npy'))
        self.test_file = json.load(open(os.path.join(self.data_path, prefix + '.json')))

        self.data_test_bert_word = np.load(os.path.join(self.data_path, prefix+'_bert_word.npy'))
        self.data_test_bert_mask = np.load(os.path.join(self.data_path, prefix+'_bert_mask.npy'))
        self.data_test_bert_starts = np.load(os.path.join(self.data_path, prefix+'_bert_starts.npy'))

        self.test_len = self.data_test_word.shape[0]
        assert (self.test_len == len(self.test_file))

        print("Finish reading")

        self.test_batches = self.data_test_word.shape[0] // self.test_batch_size
        if self.data_test_word.shape[0] % self.test_batch_size != 0:
            self.test_batches += 1

        self.test_order = list(range(self.test_len))
        self.test_order.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)

    def get_train_batch(self):
        random.shuffle(self.train_order)

        context_idxs = torch.LongTensor(self.batch_size, self.max_length).to(self.device)
        context_pos = torch.LongTensor(self.batch_size, self.max_length).to(self.device)
        context_len = torch.LongTensor(self.batch_size, self.max_length).to(self.device)
        node_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length).to(self.device)
        node_type = torch.LongTensor(self.batch_size, self.h_t_limit).to(self.device)
        edge_dis = torch.Tensor(self.batch_size,self.max_length,self.max_length).to(self.device)
        #entity_count = torch.zeros(self.batch_size,dtype= torch.int64).to(self.device)
        relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num).to(self.device)
        relation_mask = torch.Tensor(self.batch_size, self.h_t_limit).to(self.device)
        pos_idx = torch.LongTensor(self.batch_size, self.max_length).to(self.device)
        context_ner = torch.LongTensor(self.batch_size, self.max_length).to(self.device)
        #context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit).to(self.device)
        relation_label = torch.LongTensor(self.batch_size, self.h_t_limit).to(self.device)
        head_index = torch.LongTensor(self.batch_size, self.h_t_limit).to(self.device)
        tail_index = torch.LongTensor(self.batch_size, self.h_t_limit).to(self.device)
        context_masks = torch.LongTensor(self.batch_size, self.max_length).to(self.device)
        context_starts = torch.LongTensor(self.batch_size, self.max_length).to(self.device)
        ht_pair_pos = torch.LongTensor(self.batch_size, self.h_t_limit).to(self.device)
        mention_count = torch.LongTensor(self.batch_size).to(self.device)

        for b in range(self.train_batches):
            start_id = b * self.batch_size
            cur_bsz = min(self.batch_size, self.train_len - start_id)
            cur_batch = list(self.train_order[start_id: start_id + cur_bsz])
            cur_batch.sort(key=lambda x: np.sum(self.data_train_word[x] > 0), reverse=True)


            for mapping in [node_mapping,edge_dis,head_index,tail_index,node_type,mention_count]:
                mapping.zero_()

            for mapping in [relation_multi_label, relation_mask, pos_idx,ht_pair_pos]:
                mapping.zero_()

            relation_label.fill_(IGNORE_INDEX)

            max_h_t_cnt = 1
            max_e_cnt = 1

            for i, index in enumerate(cur_batch):
                context_idxs[i].copy_(torch.from_numpy(self.data_train_bert_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_train_pos[index, :]))
                context_len[i].copy_(torch.from_numpy(self.data_train_len[index, :]))
                #context_char_idxs[i].copy_(torch.from_numpy(self.data_train_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_train_ner[index, :]))
                context_masks[i].copy_(torch.from_numpy(self.data_train_bert_mask[index, :]))
                context_starts[i].copy_(torch.from_numpy(self.data_train_bert_starts[index, :]))

                m_mapping = np.zeros((self.h_t_limit, self.max_length),dtype=np.float32)
                e_mapping = np.zeros((self.h_t_limit, self.max_length),dtype=np.float32)
                s_mapping = np.zeros((self.h_t_limit, self.max_length),dtype=np.float32)

                for j in range(self.max_length):
                    if self.data_train_word[index, j] == 0:
                        break
                    pos_idx[i, j] = j + 1

                sen_count = 0
                for j in range(self.max_length-1):
                    if context_len[i][j+1] == 0 and j > 0:
                        break
                    s_mapping[sen_count,context_len[i][j]:context_len[i][j+1]] =1.0/(context_len[i][j+1]-context_len[i][j]).cpu().numpy()
                    #print(s_mapping[sen_count])
                    sen_count= sen_count +1

                ins = self.train_file[index]
                labels = ins['labels']
                L = len(ins['vertexSet'])
                #entity_count[i]=L
                max_e_cnt = max(max_e_cnt, L)
                men_count = 0
                mention_loc = []
                for j in range(L):
                    en = ins['vertexSet'][j]
                    edge_dis[i, j, j] = 1
                    for k in range(len(en)):
                        mention = en[k]
                        start_pos = mention['pos'][0]
                        mention_loc.append(start_pos)
                        end_pos = mention['pos'][1]
                        sent_id = mention['sent_id']

                        m_mapping[men_count,start_pos:end_pos] =1.0/(end_pos-start_pos)
                        #print(m_mapping[men_count])
                        e_mapping[j,start_pos:end_pos] = 1.0/len(en)/(end_pos-start_pos)

                        edge_dis[i,j, L+men_count] = 1    #ME
                        edge_dis[i,L+men_count,j] = 1
                        men_count = men_count + 1
                    #print(e_mapping[j])

                for j in range(L):
                    en = ins['vertexSet'][j]
                    count = 0
                    edge_dis[i,j,L+men_count+sent_id] = 1.0 #ES
                    edge_dis[i, L + men_count + sent_id,j] = 1.0

                    for n in range(len(en)):
                        mention = en[n]
                        sent_id = mention['sent_id']
                        edge_dis[i,L+count,L+men_count+sent_id] = 1.0  #MS
                        edge_dis[i,L+men_count+sent_id,L+count]=1.0
                        count = count + 1

                for m in range(men_count):
                    for n in range(men_count):
                        if m==n:
                            edge_dis[i,L+m,L+n] = 1.0  #MM
                            edge_dis[i,L+m,L+n] = 1.0
                        if mention_loc[n]-mention_loc[m] == 0:
                            edge_dis[i, L + m, L + n] = 1.0
                            edge_dis[i, L + n, L + m] = 1.0
                        else:
                            edge_dis[i,L+m,L+n] = 1.0/self.dis2idx[abs(mention_loc[n]-mention_loc[m])]
                            edge_dis[i, L + n, L + m] = 1.0 / self.dis2idx[abs(mention_loc[m] - mention_loc[n])]

                for m in range(sen_count):  #SS
                    for n in range(sen_count):
                        if m != n:
                            edge_dis[i,L+men_count+m,L+men_count+n] = 1.0/(n-m)
                            edge_dis[i,L+men_count+n,L+men_count+n] = 1.0/(m-n)
                        else:
                            edge_dis[i,L+men_count+m,L+men_count+m] =1.0

                for j in range(L):
                    edge_dis[i, j, j] = 1

                node_count = L +men_count+sen_count
                mention_count[i] = men_count + L
                node_type[i, :L] = 0
                node_type[i, L:L + men_count] = 1
                node_type[i, L + men_count:node_count] = 2
                max_e_cnt = max(max_e_cnt, node_count)
                node_mapping[i,:node_count].copy_(torch.from_numpy(np.concatenate((e_mapping[:L,],m_mapping[:men_count,],s_mapping[:sen_count,]),axis = 0)))

                en_pair = 0
                for m in range(L):
                    for n in range(L):
                        if m != n:
                            # head_index[i,en_pair] = m
                            # tail_index[i,en_pair] = n
                            en_pair = en_pair + 1

                idx2label = defaultdict(list)

                for label in labels:
                    idx2label[(label['h'], label['t'])].append(label['r'])

                train_tripe = list(idx2label.keys())

                entity_pairs = []
                for m in range(L):
                    for n in range(L):
                        if m!= n:
                            entity_pairs.append((m,n))
                entity_pairs2id = {}
                for m,n in enumerate(entity_pairs):
                    entity_pairs2id[n] = m

                for j, (h_idx, t_idx) in enumerate(train_tripe):
                    label = idx2label[(h_idx, t_idx)]
                    pair_id = entity_pairs2id[(h_idx,t_idx)]
                    head_index[i,j] = h_idx
                    tail_index[i,j] = t_idx
                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]
                    for r in label:
                        relation_multi_label[i, j, int(r)] = 1
                    delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]

                    if delta_dis < 0:
                        ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                    else:
                        ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                    relation_mask[i, j] = 1
                    rt = np.random.randint(len(label))
                    relation_label[i,j] = int(label[rt])

                lower_bound = len(ins['na_triple'])
                # random.shuffle(ins['na_triple'])
                # lower_bound = max(20, len(train_tripe)*3)

                for j, (h_idx, t_idx) in enumerate(ins['na_triple'][:lower_bound], len(train_tripe)):
                    pair_id = entity_pairs2id[(h_idx, t_idx)]
                    head_index[i,j] = h_idx
                    tail_index[i,j] = t_idx
                    relation_multi_label[i,j, 0] = 1
                    relation_label[i, j] = 0
                    relation_mask[i, j] = 1
                    hlist = ins['vertexSet'][h_idx]
                    tlist = ins['vertexSet'][t_idx]
                    delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]

                    if delta_dis < 0:
                        ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                    else:
                        ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   #'context_len': context_len[:cur_bsz, :max_c_len].contiguous(),
                   'relation_label': relation_label[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'input_lengths': input_lengths.contiguous(),
                   'pos_idx': pos_idx[:cur_bsz, :max_c_len].contiguous(),
                   'relation_multi_label': relation_multi_label[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   #'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'node_mapping': node_mapping[:cur_bsz,:max_e_cnt,:max_c_len].contiguous(),
                   'edge_weight': edge_dis[:cur_bsz,:max_e_cnt,:max_e_cnt].contiguous(),
                   'head_index': head_index[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'tail_index': tail_index[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'node_type': node_type[:cur_bsz,:max_e_cnt].contiguous(),
                   'context_masks': context_masks[:cur_bsz, :max_c_len].contiguous(),
                   'context_starts': context_starts[:cur_bsz, :max_c_len].contiguous(),
                   #'entity_count': entity_count[:cur_bsz].contiguous(),
                   'ht_pair_pos': ht_pair_pos[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'mention_count': mention_count[:cur_bsz].contiguous(),
                   }

    def get_test_batch(self):
        context_idxs = torch.LongTensor(self.test_batch_size, self.max_length).to(self.device)
        context_pos = torch.LongTensor(self.test_batch_size, self.max_length).to(self.device)
        context_len = torch.LongTensor(self.test_batch_size, self.max_length).to(self.device)
        context_ner = torch.LongTensor(self.test_batch_size, self.max_length).to(self.device)
        #context_char_idxs = torch.LongTensor(self.test_batch_size, self.max_length, self.char_limit).to(self.device)
        relation_mask = torch.Tensor(self.test_batch_size, self.h_t_limit).to(self.device)
        node_mapping = torch.Tensor(self.test_batch_size, self.h_t_limit, self.max_length).to(self.device)
        node_type = torch.LongTensor(self.test_batch_size, self.h_t_limit).to(self.device)
        edge_dis = torch.Tensor(self.test_batch_size,self.max_length,self.max_length).to(self.device)
        head_index = torch.LongTensor(self.test_batch_size, self.h_t_limit).to(self.device)
        tail_index = torch.LongTensor(self.test_batch_size, self.h_t_limit).to(self.device)
        context_masks = torch.LongTensor(self.test_batch_size, self.max_length).to(self.device)
        context_starts = torch.LongTensor(self.test_batch_size, self.max_length).to(self.device)
        ht_pair_pos = torch.LongTensor(self.test_batch_size, self.h_t_limit).to(self.device)
        mention_count = torch.LongTensor(self.test_batch_size).to(self.device)
        relation_label = torch.LongTensor(self.test_batch_size, self.h_t_limit).to(self.device)

        for b in range(self.test_batches):
            start_id = b * self.test_batch_size
            cur_bsz = min(self.test_batch_size, self.test_len - start_id)
            cur_batch = list(self.test_order[start_id: start_id + cur_bsz])

            for mapping in [relation_mask,node_mapping,edge_dis,head_index,tail_index,node_type,ht_pair_pos,mention_count,relation_label]:
                mapping.zero_()

            max_h_t_cnt = 1
            max_e_cnt = 1

            cur_batch.sort(key=lambda x: np.sum(self.data_test_word[x] > 0), reverse=True)

            labels = []

            L_vertex = []
            titles = []
            indexes = []
            for i, index in enumerate(cur_batch):
                context_idxs[i].copy_(torch.from_numpy(self.data_test_bert_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_test_pos[index, :]))
                context_len[i].copy_(torch.from_numpy(self.data_test_len[index, :]))
                #context_char_idxs[i].copy_(torch.from_numpy(self.data_test_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_test_ner[index, :]))
                context_masks[i].copy_(torch.from_numpy(self.data_test_bert_mask[index, :]))
                context_starts[i].copy_(torch.from_numpy(self.data_test_bert_starts[index, :]))
                m_mapping = np.zeros((self.h_t_limit, self.max_length),dtype=np.float32)
                e_mapping = np.zeros((self.h_t_limit, self.max_length),dtype=np.float32)
                s_mapping = np.zeros((self.h_t_limit, self.max_length),dtype=np.float32)

                idx2label = defaultdict(list)
                ins = self.test_file[index]

                sen_count = 0
                for j in range(self.max_length-1):
                    if context_len[i][j+1] == 0 and j > 0:
                        break
                    s_mapping[sen_count,context_len[i][j]:context_len[i][j+1]] =1.0/(context_len[i][j+1]-context_len[i][j]).cpu().numpy()
                    #print(s_mapping[sen_count])
                    sen_count= sen_count +1

                for label in ins['labels']:
                    idx2label[(label['h'], label['t'])].append(label['r'])

                L = len(ins['vertexSet'])
                titles.append(ins['title'])

                max_e_cnt = max(max_e_cnt, L)
                men_count = 0
                mention_loc = []
                for j in range(L):
                    en = ins['vertexSet'][j]
                    edge_dis[i, j, j] = 1
                    for k in range(len(en)):
                        mention = en[k]
                        start_pos = mention['pos'][0]
                        mention_loc.append(start_pos)
                        end_pos = mention['pos'][1]
                        sent_id = mention['sent_id']

                        m_mapping[men_count,start_pos:end_pos] =1.0/(end_pos-start_pos)
                        #print(m_mapping[men_count])
                        e_mapping[j,start_pos:end_pos] = 1.0/len(en)/(end_pos-start_pos)
                        edge_dis[i,j, L+men_count] = 1   #ME
                        edge_dis[i,L+men_count,j] = 1
                        men_count = men_count + 1
                    #print(e_mapping[j])

                for j in range(L):
                    en = ins['vertexSet'][j]
                    count = 0
                    edge_dis[i,j,L+men_count+sent_id] = 1.0 #ES
                    edge_dis[i, L + men_count + sent_id,j] = 1.0
                    for n in range(len(en)):
                        mention = en[n]
                        sent_id = mention['sent_id']
                        edge_dis[i,L+count,L+men_count+sent_id] = 1.0  #MS
                        edge_dis[i,L+men_count+sent_id,L+count]=1.0
                        count = count + 1

                for m in range(men_count):
                    for n in range(men_count):
                        if m==n:
                            edge_dis[i,L+m,L+n] = 1.0
                            edge_dis[i,L+m,L+n] = 1.0
                        if mention_loc[n]-mention_loc[m] == 0:
                            edge_dis[i, L + m, L + n] = 1.0
                            edge_dis[i, L + n, L + m] = 1.0
                        else:
                            edge_dis[i,L+m,L+n] = 1.0/self.dis2idx[abs(mention_loc[n]-mention_loc[m])]
                            edge_dis[i, L + n, L + m] = 1.0 / self.dis2idx[abs(mention_loc[m] - mention_loc[n])]

                for m in range(sen_count):
                    for n in range(sen_count):
                        if m != n:
                            edge_dis[i,L+men_count+m,L+men_count+n] = 1.0/(n-m)
                            edge_dis[i,L+men_count+n,L+men_count+m] = 1.0/(m-n)
                        else:
                            edge_dis[i,L+men_count+m,L+men_count+m] =1.0

                node_count = L +men_count+sen_count
                max_e_cnt = max(max_e_cnt, node_count)
                mention_count[i] = L+men_count
                node_type[i,:L] = 0
                node_type[i,L:L+men_count] = 1
                node_type[i,L+men_count:node_count] = 2
                node_mapping[i,:node_count].copy_(torch.from_numpy(np.concatenate((e_mapping[:L,],m_mapping[:men_count,],s_mapping[:sen_count,]),axis = 0)))

                en_pair = 0
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            head_index[i,en_pair] = h_idx
                            tail_index[i,en_pair] = t_idx

                            relation_mask[i, en_pair] = 1

                            hlist = ins['vertexSet'][h_idx]
                            tlist = ins['vertexSet'][t_idx]
                            delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
                            if delta_dis < 0:
                                ht_pair_pos[i, en_pair] = -int(self.dis2idx[-delta_dis])
                            else:
                                ht_pair_pos[i, en_pair] = int(self.dis2idx[delta_dis])
                            en_pair = en_pair + 1
                max_h_t_cnt = max(max_h_t_cnt, en_pair)
                label_set = {}
                for label in ins['labels']:
                    label_set[(label['h'], label['t'], label['r'])] = label['in' + self.train_prefix]

                labels.append(label_set)

                L_vertex.append(L)
                indexes.append(index)

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            yield {'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'context_pos': context_pos[:cur_bsz, :max_c_len].contiguous(),
                   #'context_len': context_len[:cur_bsz, :max_c_len].contiguous(),
                   'labels': labels,
                   'L_vertex': L_vertex,
                   'input_lengths': input_lengths.contiguous(),
                   'context_ner': context_ner[:cur_bsz, :max_c_len].contiguous(),
                   #'context_char_idxs': context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                   'relation_mask': relation_mask[:cur_bsz, :max_h_t_cnt],
                   'titles': titles,
                   'indexes': indexes,
                   'node_mapping': node_mapping[:cur_bsz, :max_e_cnt, :max_c_len].contiguous(),
                   'edge_weight': edge_dis[:cur_bsz, :max_e_cnt, :max_e_cnt].contiguous(),
                   'head_index': head_index[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'tail_index': tail_index[:cur_bsz, :max_h_t_cnt].contiguous(),
                   'node_type': node_type[:cur_bsz,:max_e_cnt].contiguous(),
                   'context_masks': context_masks[:cur_bsz, :max_c_len].contiguous(),
                   'context_starts': context_starts[:cur_bsz, :max_c_len].contiguous(),
                   'ht_pair_pos': ht_pair_pos[:cur_bsz,:max_h_t_cnt].contiguous(),
                   'mention_count':mention_count[:cur_bsz].contiguous(),
                   'relation_label':relation_label[:cur_bsz,:max_h_t_cnt].contiguous(),
                   }

    def train(self, model_pattern, model_name):

        ori_model = model_pattern(config=self)

        if self.pretrain_model != None:
            ori_model.load_state_dict(torch.load(self.pretrain_model,map_location=self.device), strict=False)
        ori_model.to(self.device)

        # for i in ori_model.parameters():
        #     i.requires_grad = False
        # for i in ori_model.bili.parameters():
        #     i.requires_grad = True
        # for i in ori_model.dis_embed.parameters():
        #     i.requires_grad = True

        model = nn.DataParallel(ori_model)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr= self.lr)
        # nll_average = nn.CrossEntropyLoss(size_average=True, ignore_index=IGNORE_INDEX)
        BCE = nn.BCEWithLogitsLoss(reduction='none')
        loss_func = pa.PairwiseRankingLoss(self.relation_num)

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        best_auc = 0.0
        best_f1 = 0.0
        best_epoch = 0
        # model.eval()
        # f1, auc, pr_x, pr_y = self.test(model, model_name)
        model.train()

        global_step = 0
        total_loss = 0
        start_time = time.time()

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.3, 1.0)
        plt.xlim(0.0, 0.4)
        plt.title('Precision-Recall')
        plt.grid(True)

        for epoch in range(self.max_epoch):
            self.acc_NA.clear()
            self.acc_not_NA.clear()
            self.acc_total.clear()

            for data in self.get_train_batch():

                context_idxs = data['context_idxs']
                context_pos = data['context_pos']
                #context_len = data['context_len']
                relation_label = data['relation_label']
                input_lengths = data['input_lengths']
                relation_multi_label = data['relation_multi_label']
                relation_mask = data['relation_mask']
                context_ner = data['context_ner']
                #context_char_idxs = data['context_char_idxs']
                node_mapping = data['node_mapping']
                edge_weight = data['edge_weight']
                head_index = data['head_index']
                tail_index = data['tail_index']
                node_type = data['node_type']
                context_masks = data['context_masks']
                context_starts = data['context_starts']
                ht_pair_pos = data['ht_pair_pos']
                mention_count = data['mention_count']
                #entity_count = data['entity_count']

                dis_h_2_t = ht_pair_pos + 10
                dis_t_2_h = -ht_pair_pos + 10

                predict_re = model(context_idxs, context_pos, context_ner, input_lengths,
                                    relation_mask,node_mapping,edge_weight,head_index, tail_index,node_type,
                                   context_masks,context_starts, dis_h_2_t, dis_t_2_h,mention_count,relation_label,True)
                # loss = torch.sum(BCE(predict_re, relation_multi_label) * relation_mask.unsqueeze(2)) / (
                #             self.relation_num * torch.sum(relation_mask))
                predict_re = torch.sigmoid(predict_re)
                loss = loss_func(predict_re[:,:,1:], relation_multi_label[:, :,1:], relation_mask)

                output = torch.argmax(predict_re, dim=-1)
                output = output.data.cpu().numpy()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                relation_label = relation_label.data.cpu().numpy()

                for i in range(output.shape[0]):
                    for j in range(output.shape[1]):
                        label = relation_label[i][j]
                        if label < 0:
                            break

                        if label == 0:
                            self.acc_NA.add(output[i][j] == label)
                        else:
                            self.acc_not_NA.add(output[i][j] == label)

                        self.acc_total.add(output[i][j] == label)

                global_step += 1
                total_loss += loss.item()

                if global_step % self.period == 0:
                    cur_loss = total_loss / self.period
                    elapsed = time.time() - start_time
                    logging(
                        '| epoch {:2d} | step {:4d} |  ms/b {:5.2f} | train loss {:5.3f} | NA acc: {:4.2f} | not NA acc: {:4.2f}  | tot acc: {:4.2f} '.format(
                            epoch, global_step, elapsed * 1000 / self.period, cur_loss, self.acc_NA.get(),
                            self.acc_not_NA.get(), self.acc_total.get()))
                    total_loss = 0
                    start_time = time.time()

            if epoch % self.test_epoch == 0:
                logging('-' * 89)
                eval_start_time = time.time()
                model.eval()
                f1, auc, pr_x, pr_y = self.test(model, model_name)
                model.train()
                logging('| epoch {:3d} | time: {:5.2f}s'.format(epoch, time.time() - eval_start_time))
                logging('-' * 89)

                if f1 > best_f1:
                    best_f1 = f1
                    best_auc = auc
                    best_epoch = epoch
                    path = os.path.join(self.checkpoint_dir, model_name)
                    torch.save(ori_model.state_dict(), path)

                    plt.plot(pr_x, pr_y, lw=2, label=str(epoch))
                    plt.legend(loc="upper right")
                    plt.savefig(os.path.join("fig_result", model_name))

        print("Finish training")
        print("Best epoch = %d | auc = %f" % (best_epoch, best_auc))
        print("Storing best result...")
        print("Finish storing")

    def test(self, model, model_name, output=False, input_theta=-1):
        data_idx = 0
        eval_start_time = time.time()
        # test_result_ignore = []
        total_recall_ignore = 0

        test_result = []
        total_recall = 0
        top1_acc = have_label = 0

        def logging(s, print_=True, log_=True):
            if print_:
                print(s)
            if log_:
                with open(os.path.join(os.path.join("log", model_name)), 'a+') as f_log:
                    f_log.write(s + '\n')

        for data in self.get_test_batch():
            with torch.no_grad():
                context_idxs = data['context_idxs']
                context_pos = data['context_pos']
                #context_len = data['context_len']
                labels = data['labels']
                L_vertex = data['L_vertex']
                input_lengths = data['input_lengths']
                context_ner = data['context_ner']
                #context_char_idxs = data['context_char_idxs']
                relation_mask = data['relation_mask']
                titles = data['titles']
                indexes = data['indexes']
                node_mapping = data['node_mapping']
                edge_weight = data['edge_weight']
                head_index = data['head_index']
                tail_index = data['tail_index']
                node_type = data['node_type']
                context_masks = data['context_masks']
                context_starts = data['context_starts']
                ht_pair_pos = data['ht_pair_pos']
                #entity_count = data['entity_count']
                mention_count = data['mention_count']
                relation_label = data['relation_label']

                dis_h_2_t = ht_pair_pos + 10
                dis_t_2_h = -ht_pair_pos + 10

                predict_re = model(context_idxs, context_pos, context_ner, input_lengths,
                                    relation_mask,node_mapping,edge_weight,head_index, tail_index,
                                   node_type,context_masks,context_starts,dis_h_2_t,dis_t_2_h,mention_count,relation_label,False)

                predict_re = torch.sigmoid(predict_re)

            predict_re = predict_re.data.cpu().numpy()

            for i in range(len(labels)):
                label = labels[i]
                index = indexes[i]

                total_recall += len(label)
                for l in label.values():
                    if not l:
                        total_recall_ignore += 1

                L = L_vertex[i]
                j = 0
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            r = np.argmax(predict_re[i, j])
                            if (h_idx, t_idx, str(r)) in label:
                                top1_acc += 1

                            flag = False

                            for r in range(1, self.relation_num):
                                intrain = False

                                if (h_idx, t_idx, r) in label:
                                    flag = True
                                    if label[(h_idx, t_idx, r)] == True:
                                        intrain = True

                                # if not intrain:
                                # 	test_result_ignore.append( ((h_idx, t_idx, r) in label, float(predict_re[i,j,r]),  titles[i], self.id2rel[r], index, h_idx, t_idx, r) )

                                test_result.append((
                                                   (h_idx, t_idx, str(r)) in label, float(predict_re[i, j, r]), intrain,
                                                   titles[i], self.id2rel[str(r)], index, h_idx, t_idx, r))

                            if flag:
                                have_label += 1

                            j += 1

            data_idx += 1

            if data_idx % self.period == 0:
                print('| step {:3d} | time: {:5.2f}'.format(data_idx // self.period, (time.time() - eval_start_time)))
                eval_start_time = time.time()

        # test_result_ignore.sort(key=lambda x: x[1], reverse=True)
        test_result.sort(key=lambda x: x[1], reverse=True)

        print('total_recall', total_recall)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.2, 1.0)
        plt.xlim(0.0, 0.6)
        plt.title('Precision-Recall')
        plt.grid(True)

        pr_x = []
        pr_y = []
        correct = 0
        w = 0

        if total_recall == 0:
            total_recall = 1  # for test

        for i, item in enumerate(test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        np.save('f1_pos.npy', f1_pos)
        theta = test_result[f1_pos][1]

        if input_theta == -1:
            w = f1_pos
            input_theta = theta

        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)
        if not self.is_test:
            logging('ALL  : Theta {:3.4f} | F1 {:3.4f} | AUC {:3.4f}'.format(theta, f1, auc))
        else:
            logging('ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta,
                                                                                                      f1_arr[w], auc))

        if output:
            # output = [x[-4:] for x in test_result[:w+1]]
            output = [{'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6],'predict_re':x[-8],'acc':x[0]} for x
                      in test_result[:w + 1]]
            json.dump(output, open(self.test_prefix + "_index.json", "w"))

            correct_output = []
            for rank in range(len(test_result)):
                if test_result[rank][0]:
                    x = test_result[rank]
                    ins = {'index': x[-4], 'h_idx': x[-3], 't_idx': x[-2], 'r_idx': x[-1], 'r': x[-5], 'title': x[-6],'predict_re':x[-8]}
                    ins['rank'] = rank
                    correct_output.append(ins)
            json.dump(correct_output, open(self.test_prefix + "_correct.json", "w"))

        plt.plot(pr_x, pr_y, lw=2, label=model_name)
        plt.legend(loc="upper right")
        if not os.path.exists(self.fig_result_dir):
            os.mkdir(self.fig_result_dir)
        plt.savefig(os.path.join(self.fig_result_dir, model_name))

        pr_x = []
        pr_y = []
        correct = correct_in_train = 0
        w = 0
        for i, item in enumerate(test_result):
            correct += item[0]
            if item[0] & item[2]:
                correct_in_train += 1
            if correct_in_train == correct:
                p = 0
            else:
                p = float(correct - correct_in_train) / (i + 1 - correct_in_train)
            pr_y.append(p)
            pr_x.append(float(correct) / total_recall)
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        f1 = f1_arr.max()

        auc = sklearn.metrics.auc(x=pr_x, y=pr_y)

        logging(
            'Ignore ma_f1 {:3.4f} | input_theta {:3.4f} test_result F1 {:3.4f} | AUC {:3.4f}'.format(f1, input_theta,
                                                                                                     f1_arr[w], auc))

        return f1, auc, pr_x, pr_y

    def testall(self, model_pattern, model_name, input_theta,save_name):  # , ignore_input_theta):
        model = model_pattern(config=self)
        #model = nn.DataParallel(model)

        model.load_state_dict(torch.load(save_name,map_location=self.device))
        model.to(self.device)
        model.eval()
        f1, auc, pr_x, pr_y = self.test(model, model_name, True, input_theta)
        np.save('f1.npy', f1)
        np.save('auc.npy', auc)
        np.save('pr_x.npy',pr_x)
        np.save('pr_y.npy',pr_y)

