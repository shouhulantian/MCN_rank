# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 20:56:16 2020

@author: pjx
"""
#%%
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
#%%
prec = []
for i in range(1000):
    label = dev[i]['labels']
    if len(label)< 5:
        prec.append(2)
    else:
        acc = [x['acc'] for x in output if x['index'] == i]
        correct = acc.count(1)
        prec.append(float(correct/len(label)))
        
#%%
def list_count(l):
    dict = {}
    for key in l:
        dict[key] = dict.get(key, 0) + 1
    return dict
#%%
def choose_rel(r, output):
    ins = [x for x in output if x['r_idx'] == r]
    #ins = output[ins]
    return ins
#%%
def show_ins_info(ins, flag):
    index = int(ins['index'])
    if flag == 0:
        vertex = test[index]['vertexSet']
    else:
        vertex = dev[index]['vertexSet']
    head = vertex[ins['h_idx']]
    tail = vertex[ins['t_idx']]
    title = ins['title']
    r = ins['r']

    print(index)
    print(title)
    print(head)
    print(tail)
    print(r)
    if flag == 1:
        acc = ins['acc']
        print(acc)
        return index, head, tail, title, r, acc
    else:
        return index, head, tail, title, r
#%%    
def output_rel_info(ins,flag,  rel):
    output = []
    for i in ins:
        o = show_ins_info(i, flag)
        output.append(o)
    f = open('rel_info_output_'+ str(rel) +'.json','w')
    json.dump(output, f, indent = 2)
#%%
def assert_rel(rel):
    aa = choose_rel(rel, best)
    index = [x['index'] for x in aa]
    print (Counter(index))

assert_rel(3)
#%%
def assert_type(rel):
    aa = choose_rel(rel, best)
    for ins in aa:
        index = int(ins['index'])
        vertex = test[index]['vertexSet']
        head = vertex[ins['h_idx']]
        tail = vertex[ins['t_idx']]
        h_type = head[0]['type']
        t_type = tail[0]['type']
        flag = 0
#        if h_type != 'MISC' or (t_type != 'PER' and t_type != 'ORG'):
        if h_type != 'PER':
            flag = 1
        if t_type != 'LOC':
            flag = 1
        if flag == 1:
            print(index)
            print(ins['title'])

assert_type(2)
#%%
def assert_unique(rel):
    aa = choose_rel(rel, best)
    index = [x['index'] for x in aa]
    index_set = list(set(index))
    for i in index_set:
        en_list = []
        ins = [x for x in aa if x['index'] == i]
        for j in ins:
            en_list.append(j['h_idx'])
        en_counts = Counter(en_list)
        for k in en_counts.items():
            if k[1] > 1:
                print(i)
                print(k[1])
                print(test[i]['vertexSet'][k[0]])
assert_unique(3)
#%%
def assert_stopword(rel):
    aa = choose_rel(rel, best)
    for ins in aa:
        index = int(ins['index'])
        vertex = test[index]['vertexSet']
        head = vertex[ins['h_idx']]
        tail = vertex[ins['t_idx']]
        flag = 0
        for h in head:
            if h == 'South' or h == 'East' or h == 'West' or h == 'North':
                flag = 1
        for t in tail:
            if t == 'South' or t == 'East' or t == 'West' or t == 'North':
                flag = 1
        if flag == 1:
            print(index)
            print(ins['title'])

assert_stopword(2)
#%%
process_best = []
for i in best:
    ins ={}
    ins['index'] = i['index']
    ins['h_idx'] = i['h_idx']
    ins['t_idx'] = i['t_idx']
    ins['title'] = i['title']
    ins['r'] = i['r']
    process_best.append(ins)
    
process_rank = []
for i in rank:
    ins ={}
    ins['index'] = i['index']
    ins['h_idx'] = i['h_idx']
    ins['t_idx'] = i['t_idx']
    ins['title'] = i['title']
    ins['r'] = i['r']
    process_rank.append(ins)
#%%
def find_ins(index, r):
    candidate = [x for x in best if x['index'] == index 
                 and x['r_idx'] == r]
    print(candidate)
#%%
error_output = json.load(open('error_output.json','r'))
rel2id = json.load(open('prepro_data/rel2id.json', "r"))
del_list = []
#%%
error = []
for i in error_output:
    index = i[0]
    head_en = i[1]
    tail_en = i[2]
    title = i[3]
    rel = i[4]
    
    vertexSet = test[index]['vertexSet']
    h_id = vertexSet.index(head_en)
    t_id = vertexSet.index(tail_en)
    r_id = rel2id[rel]
    
    ins = {}
    ins['index'] = index
    ins['h_idx'] = h_id
    ins['t_idx'] = t_id
    ins['title'] = title
    ins['r'] = rel
    error.append(ins)
    #del_list.append(process_best.index(ins))
#    if ins in process_best: 
#        process_best.remove(ins)
#%%
rank_notin_best = []
for i in range(len(process_rank)):
    if (process_rank[i] not in process_best) and (process_rank[i] not in error):
        rank_notin_best.append(i)
#%%
process_rank = json.load(open('rank_notin.json','r'))
process_rank_best = process_best[:]
for i in range(100):
    index = process_rank[i]['rank']
    process_rank_best.append(rank[index])
json.dump(process_rank_best, open('result.json','w'))

#%%
rank_notin_ins = []
for i in rank_notin_best:
    ins = {}
    index, head, tail, title, r = show_ins_info(rank[i],0)
    ins['rank'] = i
    ins['index'] = index
    ins['head'] = head
    ins['tail'] = tail
    ins['title'] = title
    ins['r'] = r
    rank_notin_ins.append(ins)
#json.dump(rank_notin_ins, open('rank_notin.json','w'), indent = 2)

#%%
label = [x['labels'] for x in dev]
for i in range(len(label)):
    for j in label[i]:
        if len(j['evidence']) == 0:
            print(i)
            print(j)

#%%
best = json.load(open('result_best.json','r'))
output = json.load(open('dev_dev_index.json','r'))
correct = json.load(open('dev_dev_correct.json','r'))
dev = json.load(open('dev.json','r'))
test = json.load(open('test.json','r'))
rank = json.load(open('result_rank.json','r'))

best_r = [x['r_idx'] for x in best]
correct_r = [x['r_idx'] for x in correct]

correct_r_num = list_count(correct_r)
best_r_num = list_count(best_r)
#%%
r = 1
aa = choose_rel(r, output)
output_rel_info(aa,1, r)
