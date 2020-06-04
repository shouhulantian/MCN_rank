import config
import models
import numpy as np
import os
import time
import datetime
import json
from sklearn.metrics import average_precision_score
import sys
import os
import argparse
# import IPython
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
# sys.excepthook = IPython.core.ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, default = 'GCN_ATT', help = 'name of the model')
parser.add_argument('--save_name', type = str,default='checkpoint_GCN_bert_ematt')

parser.add_argument('--train_prefix', type = str, default = 'dev_train')
parser.add_argument('--test_prefix', type = str, default = 'dev_dev')
parser.add_argument('--input_theta', type = float, default = 0.6100)
# parser.add_argument('--ignore_input_theta', type = float, default = -1)


args = parser.parse_args()
model = {
	'CNN3': models.CNN3,
	'LSTM': models.LSTM,
	'BiLSTM': models.BiLSTM,
	'ContextAware': models.ContextAware,
	'GCN': models.GCN,
	'GCN_ATT':models.GCN_ATT,
	# 'LSTM_SP': models.LSTM_SP
}

con = config.RankConfig(args)
#con.load_train_data()
con.load_test_data()
# con.set_train_model()
save_name = './checkpoint/checkpoint_GCN_bert_ematt'
con.testall(model[args.model_name], args.save_name, args.input_theta,save_name)#, args.ignore_input_theta)