import os
import audiofile as af
import numpy as np
import h5py
import argparse
from tqdm import tqdm
from python_speech_features import logfbank, fbank, mfcc

# parse args
def parse_args():
    desc="extract features to h5 struct"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default=None, help="path to the input/output dir")
    parser.add_argument('--output', type=str, default=None, help="path to the output dir")
    return parser.parse_args()

args = parse_args()
trainDataDir = args.dataset + 'train/'
valDataDir = args.dataset + 'val/'
train_h5Dir = args.output + 'train_h5/'
val_h5Dir = args.output + 'val_h5/'
if not os.path.isdir(train_h5Dir):#init output dir
    os.mkdir(train_h5Dir)
if not os.path.isdir(val_h5Dir):
    os.mkdir(val_h5Dir)
trainAllSegs = []
valAllSegs = []
allBirds = os.listdir(trainDataDir)
for bird in allBirds:# init all segs and output subdirs
    trainAllSegs += [trainDataDir+bird+'/'+x for x in os.listdir(trainDataDir+bird)]
    valAllSegs += [valDataDir+bird+'/'+x for x in os.listdir(valDataDir+bird)]
    if not os.path.isdir(train_h5Dir+bird):
        os.mkdir(train_h5Dir+bird)
    if not os.path.isdir(val_h5Dir+bird):
        os.mkdir(val_h5Dir+bird)

alreadyH5 = []
for bird in allBirds:
    alreadyH5 += os.listdir(train_h5Dir+bird)

kwargs = {
    "winlen": 0.025,
    "winstep": 0.01,
    "nfilt": 256,
    "nfft": 2048,
    "lowfreq": 50,
    "highfreq": 11000,
    "preemph": 0.97
}

def featExtractWriter(wavPath, cmn=True):
    y, sr = af.read(wavPath)
    featMfcc = mfcc(y, sr, winfunc=np.hamming, **kwargs)
    featLogfbank = logfbank(y, sr, **kwargs)
    featFbank = fbank(y, sr, winfunc=np.hamming, **kwargs)[0]
    if cmn:
        featMfcc -= np.mean(featMfcc, axis=0, keepdims=True)
        featLogfbank -= np.mean(featLogfbank, axis=0, keepdims=True)
        featFbank -= np.mean(featFbank, axis=0, keepdims=True)        
    return (featMfcc,featLogfbank,featFbank)

print('Start sorting training h5 files (this may take a while...)')
for song in tqdm(trainAllSegs, desc="Train featExtract"):
    songName = song.split('/')[-1]
    birdName = song.split('/')[-2]
    if songName+'.h5' in alreadyH5:
        continue
    h5Out = train_h5Dir + birdName + '/' + songName + '.h5'
    featMfcc, featLogfbank, featFbank = featExtractWriter(song)
    
    hf = h5py.File(h5Out, 'w')
    hf.create_dataset('mfcc', data=featMfcc)
    hf.create_dataset('logfbank', data=featLogfbank)
    hf.create_dataset('fbank', data=featFbank)
    
print('Start sorting valdidating h5 files (this may take a while...)')
for song in tqdm(valAllSegs, desc="Val featExtract"):
    songName = song.split('/')[-1]
    birdName = song.split('/')[-2]
    h5Out = val_h5Dir + birdName + '/' + songName + '.h5'
    featMfcc, featLogfbank, featFbank = featExtractWriter(song)
    
    hf = h5py.File(h5Out, 'w')
    hf.create_dataset('mfcc', data=featMfcc)
    hf.create_dataset('logfbank', data=featLogfbank)
    hf.create_dataset('fbank', data=featFbank)

print('finished sorting h5 files!')