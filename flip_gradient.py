from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

class FlipGradientBuilder(tf.Module):
    def __init__(self):
        super(FlipGradientBuilder, self).__init__()
        self.num_calls = 0

    @tf.function
    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        
        @tf.custom_gradient
        def _flip_gradients(x):
            def grad(dy):
                return tf.negative(dy) * l, None
            return x, grad
        
        self.num_calls += 1
        return _flip_gradients(x)

flip_gradient = FlipGradientBuilder()