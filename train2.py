# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

import argparse
import glob
import os

import tensorflow as tf
from tensorpack.callbacks.saver import ModelSaver
from tensorpack.input_source.input_source import QueueInput
from tensorpack.tfutils.sessinit import ChainInit
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.interface import TrainConfig
from tensorpack.train.interface import launch_train_with_config
from tensorpack.train.trainers import SyncMultiGPUTrainerReplicated
from tensorpack.utils import logger

from models.data_load import Net2DataFlow
from hparams.hparam import hparam as hp
from models.models import Net2


def remove_all_files(path):
    files = glob.glob('{}/*'.format(path))
    for f in files:
        os.remove(f)


def train(args, logdir2):
    # models
    model = Net2()

    # dataflow
    df = Net2DataFlow(hp.train2.data_path, hp.train2.batch_size)

    # set logger for event and models saver
    logger.set_logger_dir(logdir2)

    # session_conf = tf.ConfigProto(
    #     gpu_options=tf.GPUOptions(
    #         allow_growth=True,
    #         per_process_gpu_memory_fraction=0.6,
    #     ),
    # )

    session_inits = []
    ckpt2 = '{}/{}'.format(logdir2, args.ckpt) if args.ckpt else tf.train.latest_checkpoint(logdir2)
    if ckpt2:
        session_inits.append(SaverRestore(ckpt2))
    # ckpt1 = tf.train.latest_checkpoint(logdir1)
    # if ckpt1:
    #     session_inits.append(SaverRestore(ckpt1, ignore=['global_step']))
    train_conf = TrainConfig(
        model=model,
        data=QueueInput(df(n_prefetch=256, n_thread=2)),
        callbacks=[
            # TODO save on prefix net2
            ModelSaver(checkpoint_dir=logdir2),
            # ConvertCallback(logdir2, hp.train2.test_per_epoch),
        ],
        max_epoch=hp.train2.num_epochs,
        steps_per_epoch=hp.train2.steps_per_epoch,
        session_init=ChainInit(session_inits)
    )
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        train_conf.nr_tower = len(args.gpu.split(','))

    trainer = SyncMultiGPUTrainerReplicated(hp.train2.num_gpu)

    launch_train_with_config(train_conf, trainer=trainer)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case2', type=str, help='experiment case name of train2')
    parser.add_argument('-ckpt', help='checkpoint to load models.')
    parser.add_argument('-gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-r', action='store_true', help='start training from the beginning.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    hp.set_hparam_yaml(args.case2)
    logdir_train2 = '{}/{}/train2'.format(hp.logdir_path, args.case2)

    if args.r:
        remove_all_files(logdir_train2)

    print('case2: {}, logdir2: {}'.format(args.case2, logdir_train2))

    train(args, logdir2=logdir_train2)

    print("Done")
