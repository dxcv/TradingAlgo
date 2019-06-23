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


class DDPG:
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
        self.tf_account = tf.placeholder(tf.float32, [None, 3], name="account")
        #self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
        self.tf_obs = tf.placeholder(tf.float32, [None, 2, 60, 1], name="observations")
        self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())


    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1



    def _build_a(self, s, scope, trainable):

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


    def _build_c(self, s, a, scope, trainable):
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


