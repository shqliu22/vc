# -*- coding: utf-8 -*-
# /usr/bin/python2

import glob
import random

import librosa
import numpy as np
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow import PrefetchData
from utils.audio import preemphasis, amp2db
from hparams.hparam import hparam as hp
import os


def normalize_0_1(values, max, min):
    normalized = np.clip((values - min) / (max - min), 0, 1)
    return normalized


def denormalize_0_1(normalized, max, min):
    values = np.clip(normalized, 0, 1) * (max - min) + min
    return values


class DataFlow(RNGDataFlow):

    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        # self.wav_files = glob.glob(data_path)
        self.npy_files = glob.glob(data_path)

    def __call__(self, n_prefetch=1000, n_thread=1):
        df = self
        df = BatchData(df, self.batch_size)
        df = PrefetchData(df, n_prefetch, n_thread)
        return df


class Net2DataFlow(DataFlow):

    def get_data(self):
        while True:
            npy_file = random.choice(self.npy_files)
            length = len(np.load(npy_file))
            while length < ((hp.default.duration * hp.default.sr) // hp.default.hop_length) + 1:
                npy_file = random.choice(self.npy_files)
                length = len(np.load(npy_file))

            yield get_mfccs_and_spectrogram(npy_file)


def wav_random_crop(wav, sr, duration):
    assert (wav.ndim <= 2)

    target_len = sr * duration
    wav_len = wav.shape[-1]
    start = np.random.choice(range(np.maximum(1, wav_len - target_len)), 1)[0]
    end = start + target_len
    if wav.ndim == 1:
        wav = wav[start:end]
    else:
        wav = wav[:, start:end]
    return wav


def get_mfccs_and_spectrogram(npy_file, trim=True, random_crop=False):
    '''This is applied in `train2`, `test2` or `convert` phase.
    '''
    basepath, filename = os.path.split(npy_file)
    filename, _ = os.path.splitext(filename)

    # Load
    wav, _ = librosa.load(basepath + "/" + filename + ".wav", sr=hp.default.sr)

    # Trim
    if trim:
        wav, _ = librosa.effects.trim(wav, frame_length=hp.default.win_length, hop_length=hp.default.hop_length)

    if random_crop:
        wav = wav_random_crop(wav, hp.default.sr, hp.default.duration)

    # Padding or crop
    length = hp.default.sr * hp.default.duration
    wav = librosa.util.fix_length(wav, length)

    return _get_mfcc_and_spec(npy_file, wav, hp.default.preemphasis, hp.default.n_fft, hp.default.win_length,
                              hp.default.hop_length)


def make_one_hot(data):
    return (np.arange(9999) == data[:, None]).astype(np.int32)


# TODO refactoring
def _get_mfcc_and_spec(npy_file, wav, preemphasis_coeff, n_fft, win_length, hop_length):
    # Pre-emphasis
    y_preem = preemphasis(wav, coeff=preemphasis_coeff)

    # Get spectrogram
    D = librosa.stft(y=y_preem, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(D)

    # Get mel-spectrogram
    mel_basis = librosa.filters.mel(hp.default.sr, hp.default.n_fft, hp.default.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) # mel spectrogram

    # Get mfccs, amp to db
    mag_db = amp2db(mag)
    mel_db = amp2db(mel)
    mfccs = np.dot(librosa.filters.dct(hp.default.n_mfcc, mel_db.shape[0]), mel_db)

    # Normalization (0 ~ 1)
    mag_db = normalize_0_1(mag_db, hp.default.max_db, hp.default.min_db)
    mel_db = normalize_0_1(mel_db, hp.default.max_db, hp.default.min_db)

    ppgs = np.load(npy_file)
    ppgs = librosa.util.fix_length(ppgs, ((hp.default.duration * hp.default.sr) // hp.default.hop_length + 1))
    return make_one_hot(ppgs), mfccs.T, mag_db.T, mel_db.T  # (t, n_mfccs), (t, 1+n_fft/2), (t, n_mels)
