import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from shutil import copyfile

def parse_args():
    desc="sort three kinds of DA methods"
    parser=argparse.ArgumentParser(description=desc)
    parser.add_argument('--aug', type=str, required=True,help="/path/to/aug/dir")
    parser.add_argument('--out', type=str, required=True, help="/path/to/all/aug/dir")
    return parser.parse_args()

args = parse_args()
aug_dir = args.aug
out_dir = args.out
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
if not os.path.isdir(out_dir+'train/'):
    os.mkdir(out_dir+'train/')
if not os.path.isdir(out_dir+'val/'):
    os.mkdir(out_dir+'val/')
policies = ['1.8/','mix2/','op1/']

for pol in tqdm(policies, desc="policies"):
    data_dir = aug_dir+pol
    all_birds = os.listdir(data_dir+'train')
    if len(all_birds) == 0:
        continue
    for bird in tqdm(all_birds,desc="25 birds"):
        if not os.path.isdir(out_dir+'train/'+bird):
            os.mkdir(out_dir+'train/'+bird)
        all_train = [data_dir+'train/'+bird+'/'+x for x in os.listdir(data_dir+'train/'+bird)]
        for song in all_train:
            copyfile(song, out_dir+'train/'+bird+'/'+pol[:-1]+'_'+song.split('/')[-1])
            
        if len(os.listdir(data_dir+'val'))==0:
            continue
        if not os.path.isdir(out_dir+'val/'+bird):
            os.mkdir(out_dir+'val/'+bird)
        all_val = [data_dir+'val/'+bird+'/'+x for x in os.listdir(data_dir+'val/'+bird)]
        for song in all_val:
            copyfile(song, out_dir+'val/'+bird+'/'+pol[:-1]+'_'+song.split('/')[-1])