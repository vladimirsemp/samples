## User operation for Tensorflow: Log2round.

# Discription
This is code for a custom activation function Log2round for [Tensorflow](https://github.com/tensorflow/tensorflow). It takes tensor as input and outputs a tensor of the same size. Log2round doesn't change gradient during the backpropagation step. Log2round makes it easy to implement an approach analogous to and inspired by the one discussed in [https://arxiv.org/pdf/1602.02830v3.pdf](https://arxiv.org/pdf/1602.02830v3.pdf).
```
log2round(x) = pow(2, n) * sign(x),
where integer n minimizes | |x| - pow(2,n) |; n can be of any sign;
n is bounded to be between -14 and +10.

d(log2round(x))/dx = dx/dx = 1

```
It's expected that n only takes values from a relatively narrow subset of integers when computed for all weights and activations of a neural network.


# Usage

To use Log2round one needs Tensorflow installed and import these modules:
```
import log2round_grad
from log2round_module import log2round
```
See log2round_test.py for an example.

# Ownership
Relu and tensorflow math operations were used as a prototype for this code. That's why some portion of this code is written by the tensorflow contributors and I don't own it. 
