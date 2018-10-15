from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype

from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, InputLayer
from keras.models import Model, Sequential
from keras.layers import LeakyReLU
import numpy as np

class KickPolicy(object):
    recurrent = False
     

    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            
    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        
        # with tf.variable_scope("obfilter"):
        #     self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        # obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        obz = ob

        valueFunction = Sequential()
        valueFunction.add(InputLayer(input_tensor = obz))
        valueFunction.add(Dense(75))
        valueFunction.add(LeakyReLU())
        valueFunction.add(Dense(50))
        valueFunction.add(LeakyReLU())
        valueFunction.add(Dense(23))
        valueFunction.load_weights("neural_kick")

        #for i in range(num_hid_layers):
        #    last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = self.dense(x = valueFunction.output, size = 1, name = "vffinal", weight_init = U.normc_initializer(1.0), bias = True)[:,0]

        model =  Sequential()
        model.add(InputLayer(input_tensor = obz))
        model.add(Dense(75))
        model.add(LeakyReLU())
        model.add(Dense(50))
        model.add(LeakyReLU())
        model.add(Dense(23))
        model.load_weights("neural_kick")
        # for i in range(num_hid_layers):'
        #     last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = model.output            
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.constant_initializer(-5))
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = tf.layers.dense(model.output, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
       # my_var = tf.strided_slice(mean, [0], [1], [1], shrink_axis_mask=1)
       # my_var_out = tf.identity(my_var, name='output_node')
        self.pd = pdtype.pdfromflat(pdparam)
        self.state_in = []
        self.state_out = []


        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

    def dense(self, x, size, name, weight_init=None, bias=True):
     w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
     ret = tf.matmul(x, w)
     if bias:
         b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
         return ret + b
     else:
         return ret