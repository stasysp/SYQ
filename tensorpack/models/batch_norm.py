#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: batch_norm.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages
from copy import copy
import re

from ..tfutils.common import get_tf_version
from ..tfutils.tower import get_current_tower_context
from ..utils import logger
from ._common import layer_register

__all__ = ['BatchNorm', 'BatchNormV2']

# def BatchNorm(inputs, axis=None, training=None, momentum=0.9, epsilon=1e-5,
#               center=True, scale=True,
#               beta_initializer=tf.zeros_initializer(),
#               gamma_initializer=tf.ones_initializer(),
#               virtual_batch_size=None,
#               data_format='channels_last',
#               internal_update=False,
#               sync_statistics=None):

#     # parse shapes
#     data_format = get_data_format(data_format, keras_mode=False)
#     shape = inputs.get_shape().as_list()
#     ndims = len(shape)
#     assert ndims in [2, 4], ndims
#     if sync_statistics is not None:
#         sync_statistics = sync_statistics.lower()
#     assert sync_statistics in [None, 'nccl', 'horovod'], sync_statistics

#     if axis is None:
#         if ndims == 2:
#             axis = 1
#         else:
#             axis = 1 if data_format == 'NCHW' else 3
#     assert axis in [1, 3], axis
#     num_chan = shape[axis]

#     # parse training/ctx
#     ctx = get_current_tower_context()
#     if training is None:
#         training = ctx.is_training
#     training = bool(training)
#     TF_version = get_tf_version_tuple()
#     freeze_bn_backward = not training and ctx.is_training
#     if freeze_bn_backward:
#         assert TF_version >= (1, 4), \
#             "Fine tuning a BatchNorm model with fixed statistics needs TF>=1.4!"
#         if ctx.is_main_training_tower:  # only warn in first tower
#             logger.warn("[BatchNorm] Using moving_mean/moving_variance in training.")
#         # Using moving_mean/moving_variance in training, which means we
#         # loaded a pre-trained BN and only fine-tuning the affine part.

#     if sync_statistics is None or not (training and ctx.is_training):
#         coll_bk = backup_collection([tf.GraphKeys.UPDATE_OPS])
#         with rename_get_variable(
#                 {'moving_mean': 'mean/EMA',
#                     'moving_variance': 'variance/EMA'}):
#             tf_args = dict(
#                 axis=axis,
#                 momentum=momentum, epsilon=epsilon,
#                 center=center, scale=scale,
#                 beta_initializer=beta_initializer,
#                 gamma_initializer=gamma_initializer,
#                 # https://github.com/tensorflow/tensorflow/issues/10857#issuecomment-410185429
#                 fused=(ndims == 4 and axis in [1, 3] and not freeze_bn_backward),
#                 _reuse=tf.get_variable_scope().reuse)
#             if TF_version >= (1, 5):
#                 tf_args['virtual_batch_size'] = virtual_batch_size
#             else:
#                 assert virtual_batch_size is None, "Feature not supported in this version of TF!"
#             use_fp16 = inputs.dtype == tf.float16
#             if use_fp16:
#                 # non-fused does not support fp16; fused does not support all layouts.
#                 # we made our best guess here
#                 tf_args['fused'] = True
#             layer = tf.layers.BatchNormalization(**tf_args)
#             xn = layer.apply(inputs, training=training, scope=tf.get_variable_scope())

#         # maintain EMA only on one GPU is OK, even in replicated mode.
#         # because during training, EMA isn't used
#         if ctx.is_main_training_tower:
#             for v in layer.non_trainable_variables:
#                 if isinstance(v, tf.Variable):
#                     tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)
#         if not ctx.is_main_training_tower or internal_update:
#             restore_collection(coll_bk)

#         if training and internal_update:
#             assert layer.updates
#             with tf.control_dependencies(layer.updates):
#                 ret = tf.identity(xn, name='output')
#         else:
#             ret = tf.identity(xn, name='output')

#         vh = ret.variables = VariableHolder(
#             moving_mean=layer.moving_mean,
#             mean=layer.moving_mean,  # for backward-compatibility
#             moving_variance=layer.moving_variance,
#             variance=layer.moving_variance)  # for backward-compatibility
#         if scale:
#             vh.gamma = layer.gamma
#         if center:
#             vh.beta = layer.beta
#     else:
#         red_axis = [0] if ndims == 2 else ([0, 2, 3] if axis == 1 else [0, 1, 2])

#         new_shape = None  # don't need to reshape unless ...
#         if ndims == 4 and axis == 1:
#             new_shape = [1, num_chan, 1, 1]

#         batch_mean = tf.reduce_mean(inputs, axis=red_axis)
#         batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axis)

#         if sync_statistics == 'nccl':
#             num_dev = ctx.total
#             if num_dev == 1:
#                 logger.warn("BatchNorm(sync_statistics='nccl') is used with only one tower!")
#             else:
#                 assert six.PY2 or TF_version >= (1, 10), \
#                     "Cross-GPU BatchNorm is only supported in TF>=1.10 ." \
#                     "Upgrade TF or apply this patch manually: https://github.com/tensorflow/tensorflow/pull/20360"

#                 if TF_version <= (1, 12):
#                     try:
#                         from tensorflow.contrib.nccl.python.ops.nccl_ops import _validate_and_load_nccl_so
#                     except Exception:
#                         pass
#                     else:
#                         _validate_and_load_nccl_so()
#                     from tensorflow.contrib.nccl.ops import gen_nccl_ops
#                 else:
#                     from tensorflow.python.ops import gen_nccl_ops
#                 shared_name = re.sub('tower[0-9]+/', '', tf.get_variable_scope().name)
#                 batch_mean = gen_nccl_ops.nccl_all_reduce(
#                     input=batch_mean,
#                     reduction='sum',
#                     num_devices=num_dev,
#                     shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
#                 batch_mean_square = gen_nccl_ops.nccl_all_reduce(
#                     input=batch_mean_square,
#                     reduction='sum',
#                     num_devices=num_dev,
#                     shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)
#         elif sync_statistics == 'horovod':
#             # Require https://github.com/uber/horovod/pull/331
#             import horovod.tensorflow as hvd
#             if hvd.size() == 1:
#                 logger.warn("BatchNorm(sync_statistics='horovod') is used with only one process!")
#             else:
#                 import horovod
#                 hvd_version = tuple(map(int, horovod.__version__.split('.')))
#                 assert hvd_version >= (0, 13, 6), "sync_statistics=horovod needs horovod>=0.13.6 !"

#                 batch_mean = hvd.allreduce(batch_mean, average=True)
#                 batch_mean_square = hvd.allreduce(batch_mean_square, average=True)
#         batch_var = batch_mean_square - tf.square(batch_mean)
#         batch_mean_vec = batch_mean
#         batch_var_vec = batch_var

#         beta, gamma, moving_mean, moving_var = get_bn_variables(
#             num_chan, scale, center, beta_initializer, gamma_initializer)
#         if new_shape is not None:
#             batch_mean = tf.reshape(batch_mean, new_shape)
#             batch_var = tf.reshape(batch_var, new_shape)
#             # Using fused_batch_norm(is_training=False) is actually slightly faster,
#             # but hopefully this call will be JITed in the future.
#             xn = tf.nn.batch_normalization(
#                 inputs, batch_mean, batch_var,
#                 tf.reshape(beta, new_shape),
#                 tf.reshape(gamma, new_shape), epsilon)
#         else:
#             xn = tf.nn.batch_normalization(
#                 inputs, batch_mean, batch_var,
#                 beta, gamma, epsilon)

#         if ctx.is_main_training_tower:
#             ret = update_bn_ema(
#                 xn, batch_mean_vec, batch_var_vec, moving_mean, moving_var, momentum)
#         else:
#             ret = tf.identity(xn, name='output')

#         vh = ret.variables = VariableHolder(
#             moving_mean=moving_mean,
#             mean=moving_mean,  # for backward-compatibility
#             moving_variance=moving_var,
#             variance=moving_var)  # for backward-compatibility
#         if scale:
#             vh.gamma = gamma
#         if center:
#             vh.beta = beta
#     return ret

# # decay: being too close to 1 leads to slow start-up. torch use 0.9.
# # eps: torch: 1e-5. Lasagne: 1e-4
# @layer_register(log_shape=False)
# def BatchNormV1(x, use_local_stat=None, decay=0.9, epsilon=1e-5):
#     """
#     Batch normalization layer as described in:

#     `Batch Normalization: Accelerating Deep Network Training by
#     Reducing Internal Covariance Shift <http://arxiv.org/abs/1502.03167>`_.

#     :param input: a NHWC or NC tensor
#     :param use_local_stat: bool. whether to use mean/var of this batch or the moving average.
#         Default to True in training and False in inference.
#     :param decay: decay rate. default to 0.9.
#     :param epsilon: default to 1e-5.

#     Note that only the first training tower maintains a moving average.
#     """

#     shape = x.get_shape().as_list()
#     assert len(shape) in [2, 4]

#     n_out = shape[-1]  # channel
#     assert n_out is not None
#     beta = tf.get_variable('beta', [n_out],
#             initializer=tf.constant_initializer())
#     gamma = tf.get_variable('gamma', [n_out],
#             initializer=tf.constant_initializer(1.0))

#     if len(shape) == 2:
#         batch_mean, batch_var = tf.nn.moments(x, [0], keep_dims=False)
#     else:
#         batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=False)
#     # just to make a clear name.
#     batch_mean = tf.identity(batch_mean, 'mean')
#     batch_var = tf.identity(batch_var, 'variance')

#     emaname = 'EMA'
#     ctx = get_current_tower_context()
#     if use_local_stat is None:
#         use_local_stat = ctx.is_training
#     if use_local_stat != ctx.is_training:
#         logger.warn("[BatchNorm] use_local_stat != is_training")

#     if use_local_stat:
#         # training tower
#         if ctx.is_training:
#             #reuse = tf.get_variable_scope().reuse
#             with tf.variable_scope(tf.get_variable_scope(), reuse=False):
#                 # BatchNorm in reuse scope can be tricky! Moving mean/variance are not reused
#                 with tf.name_scope(None): # https://github.com/tensorflow/tensorflow/issues/2740
#                     # TODO if reuse=True, try to find and use the existing statistics
#                     # how to use multiple tensors to update one EMA? seems impossbile
#                     ema = tf.train.ExponentialMovingAverage(decay=decay, name=emaname)
#                     ema_apply_op = ema.apply([batch_mean, batch_var])
#                     ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
#                     if ctx.is_main_training_tower:
#                         # inside main training tower
#                         add_model_variable(ema_mean)
#                         add_model_variable(ema_var)
#     else:
#         # no apply() is called here, no magic vars will get created,
#         # no reuse issue will happen
#         assert not ctx.is_training
#         with tf.name_scope(None):
#             ema = tf.train.ExponentialMovingAverage(decay=decay, name=emaname)
#             mean_var_name = ema.average_name(batch_mean)
#             var_var_name = ema.average_name(batch_var)
#             sc = tf.get_variable_scope()
#             if ctx.is_main_tower:
#                 # main tower, but needs to use global stat. global stat must be from outside
#                 # TODO when reuse=True, the desired variable name could
#                 # actually be different, because a different var is created
#                 # for different reuse tower
#                 ema_mean = tf.get_variable('mean/' + emaname, [n_out])
#                 ema_var = tf.get_variable('variance/' + emaname, [n_out])
#             else:
#                 ## use statistics in another tower
#                 G = tf.get_default_graph()
#                 ema_mean = ctx.find_tensor_in_main_tower(G, mean_var_name + ':0')
#                 ema_var = ctx.find_tensor_in_main_tower(G, var_var_name + ':0')

#     if use_local_stat:
#         batch = tf.cast(tf.shape(x)[0], tf.float32)
#         mul = tf.where(tf.equal(batch, 1.0), 1.0, batch / (batch - 1))
#         batch_var = batch_var * mul  # use unbiased variance estimator in training

#         with tf.control_dependencies([ema_apply_op] if ctx.is_training else []):
#             # only apply EMA op if is_training
#             return tf.nn.batch_normalization(
#                 x, batch_mean, batch_var, beta, gamma, epsilon, 'output')
#     else:
#         return tf.nn.batch_normalization(
#             x, ema_mean, ema_var, beta, gamma, epsilon, 'output')

@layer_register(log_shape=False)
def BatchNormV2(x, use_local_stat=True, decay=0.9, epsilon=1e-5, post_scale=True):
    """
    Batch normalization layer as described in:

    `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariance Shift <http://arxiv.org/abs/1502.03167>`_.

    :param input: a NHWC or NC tensor
    :param use_local_stat: bool. whether to use mean/var of this batch or the moving average.
        Default to True in training and False in inference.
    :param decay: decay rate. default to 0.9.
    :param epsilon: default to 1e-5.

    Note that only the first training tower maintains a moving average.
    """
    shape = x.get_shape().as_list()
    assert len(shape) in [2, 4]
    n_out = shape[-1]  # channel
    assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"
    if len(shape) == 2:
        x = tf.reshape(x, [-1, 1, 1, n_out])
    
    #with tf.variable_scope('bn' + str(shape[3]), reuse=tf.AUTO_REUSE):
    beta = tf.get_variable('beta', [n_out],
            initializer=tf.constant_initializer())
    gamma = tf.get_variable('gamma', [n_out],
            initializer=tf.constant_initializer(1.0))
    # x * gamma + beta
    if not post_scale:
        # Disable the post-scaling factors - attempting to do it in a way that doesn't totally remove these values from the DF graph.
        beta = 0.*beta
        gamma = 0.*gamma + tf.ones_like(gamma)

    ctx = get_current_tower_context()
#     if use_local_stat is None:
#         use_local_stat = ctx.is_training
    print(use_local_stat)
#     if use_local_stat != ctx.is_training:
#         logger.warn("[BatchNorm] use_local_stat != is_training")

    #with tf.variable_scope('bn' + str(shape[3]), reuse=tf.AUTO_REUSE):
    moving_mean = tf.get_variable('mean/EMA', [n_out],
            initializer=tf.constant_initializer(), trainable=False)
    moving_var = tf.get_variable('variance/EMA', [n_out],
            initializer=tf.constant_initializer(), trainable=False)

    if use_local_stat:
        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta,
                epsilon=epsilon, is_training=True)

        # maintain EMA only in the main training tower
        if ctx.is_main_training_tower:
            update_op1 = moving_averages.assign_moving_average(
                    moving_mean, batch_mean, decay, zero_debias=False,
                    name='mean_ema_op')
            update_op2 = moving_averages.assign_moving_average(
                    moving_var, batch_var, decay, zero_debias=False,
                    name='var_ema_op')
            add_model_variable(moving_mean)
            add_model_variable(moving_var)
    else:
        assert not ctx.is_training, "In training, local statistics has to be used!"
        # TODO do I need to add_model_variable.
        # consider some fixed-param tasks, such as load model and fine tune one layer

        # fused seems slower in inference
        #xn, _, _ = tf.nn.fused_batch_norm(x, gamma, beta,
                #moving_mean, moving_var,
                #epsilon=epsilon, is_training=False, name='output')
        xn = tf.nn.batch_normalization(
            x, moving_mean, moving_var, beta, gamma, epsilon)

    # TODO for other towers, maybe can make it depend some op later
    if ctx.is_main_training_tower:
        with tf.control_dependencies([update_op1, update_op2]):
            return tf.identity(xn, name='output')
    else:
        return tf.identity(xn, name='output')

BatchNorm = BatchNormV2
