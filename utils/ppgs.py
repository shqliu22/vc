import json

import librosa
import requests
import numpy as np
import os
import math
import scipy.io.wavfile

import shutil
from pydub import AudioSegment

basepath = '../datasets/S0252/S0252_mic'


def get_int16(input_sound):
    return np.int16(input_sound / np.max(np.abs(input_sound)) * 32767)


def write_wav(wave_name, sigs, rate=16000):
    scipy.io.wavfile.write(wave_name, rate, get_int16(sigs))


def read_wav(filename):
    rate, data = scipy.io.wavfile.read(filename)
    # only use the 1st channel if stereo
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    data = data / 32768  # convert PCM int16 to float
    return data, rate


def wav_segmentation(in_sig, framesamp=320, hopsamp=160):
    sigLength = in_sig.shape[0]
    increment = framesamp / hopsamp
    M = int(math.floor(sigLength / hopsamp))
    a = np.zeros((M, framesamp), dtype=np.float32)
    for m in range(M):
        if m < increment - 1:
            seg = in_sig[0:(m + 1) * hopsamp]
            print(seg.shape)
            seg = seg * scipy.hamming(seg.shape[0])
            a[m, -len(seg):] = seg
        else:
            startpoint = (m + 1 - increment) * hopsamp;
            seg = in_sig[startpoint:startpoint + framesamp]
            # print seg.shape
            seg = seg * scipy.hamming(seg.shape[0])

            a[m, :] = seg
    return a


def get_ppgs():
    for filename in os.listdir(basepath):
        name, ext = os.path.splitext(filename)
        if ext == '.wav' and not os.path.exists(basepath + "/" + name + ".npy"):
            multipart_form_data = {
                'wave': ('wav.wav', open(basepath + "/" + filename, 'rb'))
            }
            try:
                response = requests.post('http://202.207.12.156:9000/asr', {'ali': 'true'}, files=multipart_form_data)
                content = json.loads(response.text)
                print(response.text)
                print(content['txt'])
                ppgs = np.array(json.loads(content['ali']))
                # print(ppgs)
                np.save(basepath + "/" + name + ".npy", ppgs)
            except:
                # wav=AudioSegment.from_wav(basepath + "/" + filename)
                print(filename, "失败")


def check_ppgs():
    min = 999999999999
    for filename in os.listdir(basepath):
        name, ext = os.path.splitext(filename)
        if ext == ".npy":
            ppgs = np.load(basepath + "/" + filename)
            wav = librosa.load(basepath + "/" + name + ".wav", sr=None)
            if len(wav[0]) < min:
                min = len(wav[0])
            print((len(wav[0]) + 80) // 480 == len(ppgs), len(wav[0]) / 480, (len(wav[0]) + 80) // 480, len(ppgs))
            # 到达400点算一个ppg
    print(min)


if __name__ == '__main__':
    check_ppgs()
