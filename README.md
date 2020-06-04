# MCN_rank
dataset can be downloaded from [this link](https://github.com/thunlp/DocRED/tree/master/code)

## preprocessing data

```
python gen_data.py --in_path ../data --out_path prepro_data
```

## train model
We recommend first train train_GCN_att.py for BCE loss, and then fine-tune the checkpoint using train_RANK.
```
python train_RANK.py
```

## test model

```
python test_RANK.py
```
