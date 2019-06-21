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


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            #n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        #self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_account = tf.placeholder(tf.float32, [None, 3], name="account")
            #self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_obs = tf.placeholder(tf.float32, [None, 2, 60, 1], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

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

        self.all_act_prob = tf.nn.softmax(self.all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        #prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={
			self.tf_account: observation[0],
			self.tf_obs: observation[1]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action, prob_weights.ravel()

    def store_transition(self, s, a, rs):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        #self.ep_rs.append(r)
        # Modified
        self.ep_rs = rs
        print("steve: after store", len(self.ep_rs))

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        #print(discounted_ep_rs_norm.shape, self.ep_rs, self.ep_obs)
        # train on episode
        self.sess.run(self.train_op, feed_dict={
             #self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_account: self.ep_obs[0][0],
             self.tf_obs: self.ep_obs[0][1],  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        print("steve",len(self.ep_rs))
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype=float)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        print(np.mean(discounted_ep_rs), discounted_ep_rs)
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        if np.std(discounted_ep_rs) != 0:
	        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



