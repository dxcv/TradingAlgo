"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)

GAMMA = 0.9     # reward discount in TD error


class Actor:
    def __init__(
            self,
			sess, 
            n_actions,
            #n_features,
            lr=0.001,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.lr = lr

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = sess

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_account = tf.placeholder(tf.float32, [None, 3], name="account")
            #self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_obs = tf.placeholder(tf.float32, [None, 2, 60, 1], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

            self.a = tf.placeholder(tf.int32, None, name="act")
            self.td_error = tf.placeholder(tf.float32, None, name="td_error")

        from keras.models import Model
        from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization,Merge
        from keras.layers.advanced_activations import LeakyReLU

#       self.tf_obs = Input(shape = (3,), dtype='float32', name="observations")
        B = Input(shape = (3,), tensor=self.tf_account)
        b = Dense(5, activation = "relu")(B)
 
        inputs = [B]
        merges = [b]
 
        #S = Input(shape=[2, 60, 1])
        S = Input(shape=[2, 60, 1], tensor=self.tf_obs)
        inputs.append(S)

        print(B.get_shape(), S.get_shape())
 
        h = Convolution2D(2048, 3, 1, border_mode = 'same')(S)
        h = LeakyReLU(0.001)(h)
        h = Convolution2D(2048, 5, 1, border_mode = 'same')(S)
        h = LeakyReLU(0.001)(h)
        h = Convolution2D(2048, 10, 1, border_mode = 'same')(S)
        h = LeakyReLU(0.001)(h)
        h = Convolution2D(2048, 20, 1, border_mode = 'same')(S)
        h = LeakyReLU(0.001)(h)
        h = Convolution2D(2048, 40, 1, border_mode = 'same')(S)
        h = LeakyReLU(0.001)(h)
 
        h = Flatten()(h)
        h = Dense(512)(h)
        h = LeakyReLU(0.001)(h)
        merges.append(h)
 
        h = Convolution2D(2048, 60, 1, border_mode = 'same')(S)
        h = LeakyReLU(0.001)(h)
 
        h = Flatten()(h)
        h = Dense(512)(h)
        h = LeakyReLU(0.001)(h)
        merges.append(h)
 
        m = merge(merges, mode = 'concat', concat_axis = 1)
        m = Dense(1024)(m)
        m = LeakyReLU(0.001)(m)
        m = Dense(512)(m)
        m = LeakyReLU(0.001)(m)
        m = Dense(256)(m)
        m = LeakyReLU(0.001)(m)
        self.all_act = Dense(2, activation=None)(m)

        self.acts_prob = tf.nn.softmax(self.all_act, name='act_prob')  # use softmax to convert to probability

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)


    def learn(self, s, a, td):
        feed_dict = {self.tf_account: s[0], 
                    self.tf_obs:s[1],
                    self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
#        s = s[np.newaxis, :]
#        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        probs = self.sess.run(self.acts_prob, feed_dict={
            self.tf_account: s[0],
            self.tf_obs: s[1]})

        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel()), probs.ravel()   # return a int:



class Critic:
    def __init__(
            self,
			sess, 
            n_actions,
            #n_features,
            lr=0.01,
            output_graph=False,
    ):
        self.n_actions = n_actions
        #self.n_features = n_features
        self.lr = lr

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = sess

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_account = tf.placeholder(tf.float32, [None, 3], name="account")
            #self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_obs = tf.placeholder(tf.float32, [None, 2, 60, 1], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

            self.v_ = tf.placeholder(tf.float32, [1,1], name="v_next")
            self.r = tf.placeholder(tf.float32, None, name="r")

        from keras.models import Model
        from keras.layers import merge, Convolution2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Reshape, TimeDistributed, BatchNormalization,Merge
        from keras.layers.advanced_activations import LeakyReLU

#       self.tf_obs = Input(shape = (3,), dtype='float32', name="observations")
        B = Input(shape = (3,), tensor=self.tf_account)
        b = Dense(5, activation = "relu")(B)
 
        inputs = [B]
        merges = [b]
 
        #S = Input(shape=[2, 60, 1])
        S = Input(shape=[2, 60, 1], tensor=self.tf_obs)
        inputs.append(S)

        print(B.get_shape(), S.get_shape())
 
        h = Convolution2D(2048, 3, 1, border_mode = 'same')(S)
        h = LeakyReLU(0.001)(h)
        h = Convolution2D(2048, 5, 1, border_mode = 'same')(S)
        h = LeakyReLU(0.001)(h)
        h = Convolution2D(2048, 10, 1, border_mode = 'same')(S)
        h = LeakyReLU(0.001)(h)
        h = Convolution2D(2048, 20, 1, border_mode = 'same')(S)
        h = LeakyReLU(0.001)(h)
        h = Convolution2D(2048, 40, 1, border_mode = 'same')(S)
        h = LeakyReLU(0.001)(h)
 
        h = Flatten()(h)
        h = Dense(512)(h)
        h = LeakyReLU(0.001)(h)
        merges.append(h)
 
        h = Convolution2D(2048, 60, 1, border_mode = 'same')(S)
        h = LeakyReLU(0.001)(h)
 
        h = Flatten()(h)
        h = Dense(512)(h)
        h = LeakyReLU(0.001)(h)
        merges.append(h)
 
        m = merge(merges, mode = 'concat', concat_axis = 1)
        m = Dense(1024)(m)
        m = LeakyReLU(0.001)(m)
        m = Dense(512)(m)
        m = LeakyReLU(0.001)(m)
        m = Dense(256)(m)
        m = LeakyReLU(0.001)(m)
        self.v = Dense(1, activation=None, name='V')(m)

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)


    def learn(self, s, r, s_):

        v_ = self.sess.run(self.v, 
            feed_dict={
            self.tf_account: s_[0],
            self.tf_obs: s_[1]})

        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.tf_account: s[0],
                                          self.tf_obs: s[1],
                                           self.v_: v_, self.r: r})
        return td_error
