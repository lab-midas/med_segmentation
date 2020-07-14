import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import numpy as np

# Here defines the custom class that may be used in the premodel.
# In order to run the model correctly, all the code here is not changed.
class resize_3D(Layer):
    def __init__(self, out_res=24, **kwargs):
        self.input_dim = None
        self.out_res = out_res
        super(resize_3D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[1:]
        super(resize_3D, self).build(input_shape)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'input_dim': self.input_dim,
                      'out_res': self.out_res})
        return config

    def call(self, x):
        y = K.reshape(x=x,
                      shape=(-1,
                             self.input_dim[0],
                             self.input_dim[1],
                             self.input_dim[2] * self.input_dim[3]))
        y = tf.compat.v1.image.resize_bilinear(images=y,
                                       size=(K.constant(np.array(2 * (self.out_res,),
                                                                 dtype=np.int32),
                                                        dtype=tf.int32)))
        y = K.reshape(x=y,
                      shape=(-1,
                             self.out_res,
                             self.out_res,
                             self.input_dim[2],
                             self.input_dim[3]))
        y = K.permute_dimensions(x=y, pattern=(0, 1, 3, 2, 4))
        y = K.reshape(x=y,
                      shape=(-1,
                             self.out_res,
                             self.input_dim[2],
                             self.out_res * self.input_dim[3]))
        y = tf.compat.v1.image.resize_bilinear(images=y,
                                       size=(K.constant(np.array(2 * (self.out_res,),
                                                                 dtype=np.int32),
                                                        dtype=tf.int32)))
        y = K.reshape(x=y,
                      shape=(-1,
                             self.out_res,
                             self.out_res,
                             self.out_res,
                             self.input_dim[3]))
        y = K.permute_dimensions(x=y, pattern=(0, 1, 3, 2, 4))
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + 3 * (self.out_res,) + (input_shape[-1],)
