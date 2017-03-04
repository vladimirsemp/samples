from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

_log2round_module = tf.load_op_library(
    os.path.join('', 'log2round_op.so'))
log2round = _log2round_module.log2round