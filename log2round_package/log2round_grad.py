from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops


@ops.RegisterGradient("Log2round")
def _log2round_grad(_, grad):
  """The gradients for `log2round`.
  Args:
    op: The `log2round` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `log2round` op.
  Returns:
    Gradients with respect to the input of `log2round`.
  """
  return [grad]  # List of one Tensor, since we have one input