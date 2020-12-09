import os
import numpy as np
import pandas as pd
import sox
import argparse
from tqdm import tqdm

def parse_args():
    desc="parse args for DA methods: tempo"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--inDir', type=str, required=True, help="/path/to/dataset/")
    parser.add_argument('--outDir', type=str, required=True, help="/path/to/output/")
    parser.add_argument('--tempoFactor', type=float, default=1.2, help="factor to do sox tempo")
    return parser.parse_args()

args = parse_args()
dataset_dir = args.inDir
output_dir = args.outDir
tempo_factor = args.tempoFactor
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

zeroEight = []
oneThree = []
oneFive = []
allSegs = []
# all_birds = os.listdir(dataset_dir)
seg_dir = '/DATA1/ziang/data/guangdong26/splited/train/'
all_birds = os.listdir(seg_dir)
data_stats = {bird:len(os.listdir(seg_dir+bird)) for bird in all_birds}
data_stats = {k: v for k, v in sorted(data_stats.items(), key=lambda item: item[1], reverse=True)}

# create dir for output
for bird in all_birds:
    if data_stats[bird] < 1500:
        allSegs += [dataset_dir + bird +'/' + x for x in os.listdir(dataset_dir+bird)]
#     if 500 < data_stats[bird] < 1000:
#         oneThree += [dataset_dir + bird +'/' + x for x in os.listdir(dataset_dir+bird)]
#     if 1000 < data_stats[bird] < 2500:
#         oneFive += [dataset_dir + bird +'/' + x for x in os.listdir(dataset_dir+bird)]
    if not os.path.isdir(output_dir+bird):
        os.mkdir(output_dir+bird)
    

# define a transformer
tfm = sox.Transformer()
tfm.tempo(factor=tempo_factor, audio_type='l')

# build tempoed segments
print('... start tempo with factor %.2f ...'%tempo_factor)
for song in tqdm(allSegs, desc='allSegs'):
    name = song.split('/')[-1]
    species = song.split('/')[-2]
    tfm.build_file(song, output_dir+species+'/'+name)

# tfm.tempo(factor=1.5, audio_type='l')
# for song in tqdm(oneThree, desc='oneThree'):
#     name = song.split('/')[-1]
#     species = song.split('/')[-2]
#     tfm.build_file(song, output_dir+species+'/'+name)

# tfm.tempo(factor=1.5, audio_type='l')
# for song in tqdm(oneFive, desc='oneFive'):
#     name = song.split('/')[-1]
#     species = song.split('/')[-2]
#     tfm.build_file(song, output_dir+species+'/'+name)
print('... tempoing finished ...')