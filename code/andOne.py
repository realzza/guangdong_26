import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from shutil import copyfile
from random import shuffle

def parse_args():
    desc="make other class"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data', type=str, required=True, help="from where select the class")
    parser.add_argument('--outDir', type=str, required=True, help="/path/to/other/class")
    parser.add_argument('--num', type=int, required=True)
    return parser.parse_args()

args = parse_args()
data_dir = args.data
other_dir = args.outDir+'other/'
all_birds = os.listdir(args.outDir)
num = args.num
if not os.path.isdir(other_dir):
    os.mkdir(other_dir)
    
# avoid duplicate
birds_pot = os.listdir(data_dir)
uncoverd = [bird for bird in birds_pot if bird not in all_birds]
num = int(num/len(uncoverd))
print('... each contribute %d segs ...'%num)
for bird in tqdm(uncoverd,desc="uncovered birds"):
    bird_segs = [data_dir+bird+'/'+x for x in os.listdir(data_dir+bird)]
    shuffle(bird_segs)
    selected = bird_segs[:num]
    for song in selected:
        copyfile(song, other_dir+song.split('/')[-1])