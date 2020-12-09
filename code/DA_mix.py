import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import random
import audiofile as af
from scipy.io.wavfile import write

def parse_args():
    desc="parse args for mix audios"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data',type=str, required=True, help="/path/to/data/to train or val")
    parser.add_argument('--outDir', type=str, required=True, help="/path/to/output/")
    parser.add_argument('--mixNum', type=int, default=2, help="mix n segs and gain new")
    parser.add_argument('--cmn', type=bool, default=True)
    return parser.parse_args()

def mix_songs(song_list, cmn=True):
    n = len(song_list)
    mixed_signal = np.zeros(220500)
    wts = np.random.dirichlet(np.ones(n),size=1)[0]
    for i in range(n):
        song, sr = af.read(song_list[i])
        mixed_signal += song*wts[i]
    if cmn:
        mixed_signal -= np.mean(mixed_signal,axis=0, keepdims=True)
    return mixed_signal

args = parse_args()
dataset_dir = args.data
output_dir = args.outDir
mix_n = args.mixNum
cmn = args.cmn
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

target_more = {'SpottedDove': 501,
 'anthus_richardi': 614,
 'PlainPrinia': 630,
 'Sooty-headedBulbul': 898,
 'Scaly-breastedMunia': 1009,
 'AfricanStonechat': 1285,
 'ChestnutBulbul': 1772,
 'Black-collaredStarling': 1863,
 'MaskedLaughingthrush': 1955,
 'CrestedMyna': 2002,
 'White-rumpedMunia': 2028,
 'Red-billedStarling': 2259,
 'ChinesePondHeron': 2441}

all_birds = list(target_more.keys())
for bird in all_birds:
    if target_more[bird] > 1500:
        mix_n=3
    if not os.path.isdir(output_dir+bird):
        os.mkdir(output_dir+bird)
    bird_dir = dataset_dir + bird +'/'
    bird_segs = [bird_dir+x for x in os.listdir(bird_dir)]
    for i in tqdm(range(target_more[bird]), desc='%s mix %d'%(bird,mix_n)):
        random.shuffle(bird_segs)
        mix_list = bird_segs[:mix_n]
        mixed_signal = mix_songs(mix_list, cmn=cmn)
        write(output_dir+bird+'/'+'mix_%s.wav'%i, 44100, mixed_signal)
    print('... %s DONE ...'%bird)