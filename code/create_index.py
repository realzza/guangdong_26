import os
import numpy as np
import pandas as pf
import argparse
from tqdm import tqdm

def parse_args():
    desc="create index"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data', type=str, required=True, help="dir to the training dir")
    parser.add_argument('--index', type=str, required=True, help="dir to index")
    parser.add_argument('--name', type=str, required=True, help="name of the index subdir")
    return parser.parse_args()

args = parse_args()
data_dir = args.data
index_dir = args.index
name = args.name
all_birds = os.listdir(data_dir+'train_h5')
u2w_train_segs = []
u2w_val_segs = []
u2l_train_segs = []
u2l_val_segs = []

for bird in all_birds:
    u2w_train_segs += [bird+'_'+x[:-7]+" "+data_dir+'train_h5/'+bird+'/'+x for x in os.listdir(data_dir+'train_h5/'+bird)]
    u2w_val_segs += [bird+'_'+x[:-7]+" "+data_dir+'val_h5/'+bird+'/'+x for x in os.listdir(data_dir+'val_h5/'+bird)]
    u2l_train_segs += [bird+'_'+x[:-7]+" "+bird for x in os.listdir(data_dir+'train_h5/'+bird)]
    u2l_val_segs += [bird+'_'+x[:-7]+" "+bird for x in os.listdir(data_dir+'val_h5/'+bird)]

train_u2w_text = "\n".join(u2w_train_segs)
val_u2w_text = "\n".join(u2w_val_segs)
train_u2l_text = "\n".join(u2l_train_segs)
val_u2l_text = "\n".join(u2l_val_segs)

# create index branch dir
if not os.path.isdir(index_dir+name):
    os.mkdir(index_dir+name)
with open(index_dir+name+'/train_utt2wav','w') as f:
    f.write(train_u2w_text)
with open(index_dir+name+'/val_utt2wav','w') as f:
    f.write(val_u2w_text)
with open(index_dir+name+'/train_utt2label','w') as f:
    f.write(train_u2l_text)
with open(index_dir+name+'/val_utt2label','w') as f:
    f.write(val_u2l_text)