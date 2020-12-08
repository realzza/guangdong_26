import os
import numpy as np
import pandas as pd
import argparse
import sys
import sox
import random
from scipy.io import wavfile
import soundfile as sf
import audiofile as af
import h5py
import python_speech_features
from python_speech_features import logfbank, fbank, mfcc
from tqdm import tqdm
from shutil import copyfile

def parse_args():
    desc="parse args"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gpuid', type=str, default="0")
    parser.add_argument('--data', type=str, required=True, help="/path/to/dataset/")
    parser.add_argument('--out', type=str, required=True, help="/path/to/output/")
    parser.add_argument('--segLen', type=int, default=5)
    parser.add_argument('--winShift', type=int, default=1)
    parser.add_argument('--noiseThres', type=float, default=0.5)
    return parser.parse_args()

args = parse_args()
gpuid = args.gpuid
data_dir = args.data
output_dir = args.output
segLen = args.segLen
win_shift = args.winShift
noise_thres = args.noise_thres
if not os.path.isdir(output_dir+'train'):
    os.mkdir(output_dir+'train')
if not os.path.isdir(output_dir+'val'):
    os.mkdir(output_dir+'val')
# define h5 saving dir
train_h5 = output_dir + 'train_h5/'
val_h5 = output_dir + 'val_h5/'
if not os.path.isdir(train_h5):
    os.mkdir(train_h5)
if not os.path.isdir(val_h5):
    os.mkdir(val_h5)

# first load the model
import keras
from keras.layers import Conv2D, Dropout, MaxPooling2D, Dense, GlobalAveragePooling2D, Flatten, BatchNormalization, AveragePooling2D
from keras.models import Sequential, load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error
from keras.regularizers import l2

print('... loading VAD model ...')
os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
model = Sequential()

# convolution layers
model.add(Conv2D(16, (3, 3), padding='valid', input_shape=(700, 80, 1), ))  # low: try different kernel_initializer
model.add(BatchNormalization())  # explore order of Batchnorm and activation
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3, 3)))  # experiment with using smaller pooling along frequency axis
model.add(Conv2D(16, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding='valid'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3, 1)))
model.add(Conv2D(16, (3, 3), padding='valid', kernel_regularizer=l2(0.01)))  # drfault 0.01. Try 0.001 and 0.001
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(MaxPooling2D(pool_size=(3, 1)))

# dense layers
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))
model.add(Dropout(0.5))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=.001))  # leaky relu value is very small experiment with bigger ones
model.add(Dropout(0.5))  # experiment with removing this dropout
model.add(Dense(1, activation='sigmoid'))

# load trained model
modelDir = '/Netdata/2020/ziang/code/ukybirddet/trained_model/baseline/model97.h5'
modelTest = load_model(modelDir)
print('... model loading OK ...')

# util funcs
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def wav2spec(audioDir, sampleRate):
    sr, y = wavfile.read(audioDir)
    logFeat = python_speech_features.base.logfbank(y, samplerate=44100, winlen=0.046, winstep=0.014, nfilt=80, nfft=2048, lowfreq=50, highfreq=12000, preemph=0.97)
    logFeat = np.resize(logFeat, (700,80))
    logFeat = NormalizeData(logFeat)
    return logFeat

# fix all segments need to be transformed and trimmed
train_dir = data_dir+'train/' # needed to gain bird name list
birds_25 = os.listdir(train_dir)
storage_dir = data_dir+'storage/' # define a tmp storage dir
if not os.path.isdir(storage_dir):
    os.mkdir(storage_dir)

# define a trimmer
def trimmer(clipIn, sampleRate, segLen, clipOut, win_shift):
    audioLen = sox.file_info.duration(clipIn)
    label = 0
    frameStart = 0
    frameWin = win_shift
    frameLen = segLen
    frameEnd = frameLen
    frames = []
    while frameEnd <= audioLen:
        nameTmp = '_'.join(clipIn.split('/')[-2:])[:-4] + '_seg_%d.wav'%label
        os.system('sox %s %s trim %d %d'%(clipIn, clipOut+nameTmp, frameStart, segLen))
        label += 1
        frameStart += frameWin
        frameEnd = frameStart + frameLen    

for bird in tqdm(birds_25, desc='processing'):
    sess = ['train/','val/']
    for s in sess:
        bird_dir = data_dir+sess+bird+'/'
        bird_clips = [bird_dir+x for x in os.listdir(bird_dir)]
        tfm = sox.Transformer()
        tfm.set_output_format(file_type='wav', rate=44100, channels=1)
        # transform data to dealable
        for clip in bird_clips:
            clip_name = clip.split('/')[-1]
            if not os.path.isdir(storage_dir+bird):
                os.mkdir(storage_dir+bird)
            clip_n = storage_dir + bird + '/' + clip_name[:-4] + '_new.wav'
            tfm.build_file(clip, clip_n)
            # update audio file
            audio = clip_n
            
            # check whether segment or not
            audio_length = sox.file_info.duration(audio)
            if audio_length > segLen:
                audio_out = output_dir + sess + bird + '/'
                if not os.path.isdir(audio_out):
                    os.mkdir(audio_out)
                trimmer(audio, 44100, segLen, audio_out, win_shift)
            else: 
                continue
            # trimming done
            tmp_all_segs = [audio_out + x for x in os.listdir(audio_out)]
            for seg in tmp_all_segs:
                segSpec = wav2spec(seg, sampleRate=44100)
                segSpec = segSpec.reshape(1, segSpec.shape[0], segSpec.shape[1], 1)
                predProb = modelTest.predict(segSpec)[0][0] 
                if predProb < noise_thres:
                    os.remove(seg)
            all_segs = [audio_out + x for x in os.listdir(audio_out)]
            # finished audio trimming
            # start extracting features
            kwargs = {
                "winlen": 0.025,
                "winstep": 0.01,
                "nfilt": 256,
                "nfft": 2048,
                "lowfreq": 50,
                "highfreq": 11000,
                "preemph": 0.97
            }
            # !!to-do take tool functions out of loop
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
            
            for song in tmp_all_segs:
                song_name = song.split('/')[-1]
                save_dir = train_h5+bird+'/'
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
                h5_out_path = save_dir+song_name+'.h5'
                featMfcc, featLogfbank, featFbank = featExtractWriter(song)
                hf = h5py.File(h5_out_path,'w')
                hf.create_dataset('logfbank', data=featLogfbank)
            print('... %s extraction finished ...'%bird)