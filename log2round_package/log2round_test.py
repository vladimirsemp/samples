from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf
import log2round_grad
from log2round_module import log2round


class MyTest(tf.test.TestCase):

  def test(self):
    with self.test_session():
      result = log2round([0, 1, -1, -2, 0.125, 7, -0.1, 95])
      self.assertAllEqual(result.eval(), [0, 1, -1, -2, 0.125, 8, -0.125, 64])

  def testLoadTwice(self):
    load_again = tf.load_op_library(
        os.path.join('', 'log2round_op.so'))
    self.assertEqual(load_again.log2round, log2round)
  
  def test_grad(self):
    x = tf.Variable(float)
    sess = tf.Session()
    g1 = sess.run(tf.gradients(x, x), feed_dict={x:1000.0})
    g2 = sess.run(tf.gradients(log2round(x), x), feed_dict={x:1000.0}) 
    self.assertEqual(g1, g2)

if __name__ == '__main__':
  tf.test.main()
