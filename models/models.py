# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.tfutils import (
    get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import re
from tensorpack.utils import logger
from tensorpack.tfutils.gradproc import GradientProcessor

from hparams.hparam import hparam as hp
from models.modules import prenet, cbhg


class FilterGradientVariables(GradientProcessor):
    """
    Skip the update of certain variables and print a warning.
    """

    def __init__(self, var_regex='.*', verbose=True):
        """
        Args:
            var_regex (string): regular expression to match variable to update.
            verbose (bool): whether to print warning about None gradients.
        """
        super(FilterGradientVariables, self).__init__()
        self._regex = var_regex
        self._verbose = verbose

    def _process(self, grads):
        g = []
        to_print = []
        for grad, var in grads:
            if re.match(self._regex, var.op.name):
                g.append((grad, var))
            else:
                to_print.append(var.op.name)
        if self._verbose and len(to_print):
            message = ', '.join(to_print)
            logger.warn("No gradient w.r.t these trainable variables: {}".format(message))
        return g


class Net2(ModelDesc):

    def _get_inputs(self):
        n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length + 1

        return [InputDesc(tf.float32, (None, n_timesteps, 9999), 'x_ppgs'),
                InputDesc(tf.float32, (None, n_timesteps, hp.default.n_mfcc), 'x_mfccs'),
                InputDesc(tf.float32, (None, n_timesteps, hp.default.n_fft // 2 + 1), 'y_spec'),
                InputDesc(tf.float32, (None, n_timesteps, hp.default.n_mels), 'y_mel'), ]

    def _build_graph(self, inputs):
        self.ppgs, self.x_mfcc, self.y_spec, self.y_mel = inputs

        is_training = get_current_tower_context().is_training

        # build net1
        # self.net1 = Net1()
        # with tf.variable_scope('net1'):
        #     self.ppgs, _, _ = self.net1.network(self.x_mfcc, is_training)
        self.ppgs = tf.identity(self.ppgs, name='ppgs')

        # build net2
        with tf.variable_scope('net2'):
            self.pred_spec, self.pred_mel = self.network(self.ppgs, is_training)
        self.pred_spec = tf.identity(self.pred_spec, name='pred_spec')

        self.cost = self.loss()

        # summaries
        tf.summary.scalar('net2/train/loss', self.cost)

        if not is_training:
            tf.summary.scalar('net2/eval/summ_loss', self.cost)

    def _get_optimizer(self):
        gradprocs = [
            FilterGradientVariables('.*net2.*', verbose=False),
            gradproc.MapGradient(
                lambda grad: tf.clip_by_value(grad, hp.train2.clip_value_min, hp.train2.clip_value_max)),
            gradproc.GlobalNormClip(hp.train2.clip_norm),
            # gradproc.PrintGradient(),
            # gradproc.CheckGradient(),
        ]
        lr = tf.get_variable('learning_rate', initializer=hp.train2.lr, trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        return optimizer.apply_grad_processors(opt, gradprocs)

    @auto_reuse_variable_scope
    def network(self, ppgs, is_training):
        # Pre-net
        prenet_out = prenet(ppgs,
                            num_units=[hp.train2.hidden_units, hp.train2.hidden_units // 2],
                            dropout_rate=hp.train2.dropout_rate,
                            is_training=is_training)  # (N, T, E/2)

        # CBHG1: mel-scale
        pred_mel = cbhg(prenet_out, hp.train2.num_banks, hp.train2.hidden_units // 2,
                        hp.train2.num_highway_blocks, hp.train2.norm_type, is_training,
                        scope="cbhg_mel")
        pred_mel = tf.layers.dense(pred_mel, self.y_mel.shape[-1], name='pred_mel')  # (N, T, n_mels)

        # CBHG2: linear-scale
        pred_spec = tf.layers.dense(pred_mel, hp.train2.hidden_units // 2)  # (N, T, n_mels)
        pred_spec = cbhg(pred_spec, hp.train2.num_banks, hp.train2.hidden_units // 2,
                         hp.train2.num_highway_blocks, hp.train2.norm_type, is_training, scope="cbhg_linear")
        pred_spec = tf.layers.dense(pred_spec, self.y_spec.shape[-1],
                                    name='pred_spec')  # log magnitude: (N, T, 1+n_fft//2)

        return pred_spec, pred_mel

    def loss(self):
        loss_spec = tf.reduce_mean(tf.squared_difference(self.pred_spec, self.y_spec))
        loss_mel = tf.reduce_mean(tf.squared_difference(self.pred_mel, self.y_mel))
        loss = loss_spec + loss_mel
        return loss
