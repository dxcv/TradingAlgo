import os
import numpy as np

from actor_critic import Actor, Critic 
from market_env import MarketEnv
from market_model_builder import MarketPolicyGradientModelBuilder

import csv
import numpy as np 

import tensorflow as tf


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class AC_run:

	def __init__(self, env, discount = 0.99, model_filename = None, history_filename = None, max_memory=100):
		self.env = env
		self.discount = discount
		self.model_filename = model_filename
		self.history_filename = history_filename

		self.max_memory = max_memory 

	def train(self, max_episode = 10, max_path_length = 200, verbose = 0):
		env = self.env
		avg_reward_sum = 0.

		#f_eps = open("episode.csv","w")
		#write_eps = csv.write(f_eps)

		for e in range(max_episode):
			env._reset()
			observation = env._reset()
			game_over = False
			reward_sum = 0

			inputs = []
			outputs = []
			predicteds = []
			rewards = []

			#f_iter = open("episode_{0}.csv".format(e),"w")
			#write_iter = csv.writer(f_iter)
			f_episode = "episode_{0}.csv".format(e)
			os.system("rm -rf {0}".format(f_episode))

			print(observation[0].shape, observation[1].shape)


			sess = tf.Session()

			actor = Actor(sess, 
                n_actions=self.env.action_space.n
                # output_graph=True,
            )

			critic = Critic(sess, 
				n_actions=self.env.action_space.n
				)     # we need a good teacher, so the teacher should learn faster than the actor

			sess.run(tf.global_variables_initializer())

			while not game_over:

				action, aprob = actor.choose_action(observation)

				inputs.append(observation)
				predicteds.append(aprob)
				
				y = np.zeros([self.env.action_space.n])
				y[action] = 1.
				outputs.append(y)

				observation_, reward, actual_reward, game_over, info = self.env._step(action)
				reward_sum += float(actual_reward)

				#rewards.append(float(reward))
				rewards.append(float(reward_sum))
	
				# After env.step
				td_error = critic.learn(observation, reward_sum, observation_)  # gradient = grad[r + gamma * V(s_) - V(s)]
				actor.learn(observation, action, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

				# check memory for RNN model
				if len(inputs) > self.max_memory:
					del inputs[0]
					del outputs[0]
					del predicteds[0]
					del rewards[0]


				if verbose > 0:
					if env.actions[action] == "LONG" or env.actions[action] == "SHORT":
					#if env.actions[action] == "LONG" or env.actions[action] == "SHORT" or env.actions[action] == "HOLD":
						color = bcolors.FAIL if env.actions[action] == "LONG" else bcolors.OKBLUE
						print ("%s:\t%s\t%.2f\t%.2f\t" % (info["dt"], color + env.actions[action] + bcolors.ENDC, reward_sum, info["cum"]) + ("\t".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, aprob.tolist())])))
					#write_iter.writerow("%s:\t%s\t%.2f\t%.2f\t" % (info["dt"], env.actions[action], reward_sum, info["cum"]) + ("\t".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, aprob.tolist())])))
					os.system("echo %s >> %s" % ("%s:\t%s\t%.2f\t%.2f\t" % (info["dt"], env.actions[action], reward_sum, info["cum"]) + ("\t".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, aprob.tolist())])),
							  f_episode))


				avg_reward_sum = avg_reward_sum * 0.99 + reward_sum * 0.01
				toPrint = "%d\t%s\t%s\t%.2f\t%.2f" % (e, info["code"], (bcolors.FAIL if reward_sum >= 0 else bcolors.OKBLUE) + ("%.2f" % reward_sum) + bcolors.ENDC, info["cum"], avg_reward_sum)
				print (toPrint)
				if self.history_filename != None:
					os.system("echo %s >> %s" % (toPrint, self.history_filename))



				dim = len(inputs[0])
				inputs_ = [[] for i in range(dim)]
				for obs in inputs:
					for i, block in enumerate(obs):
						inputs_[i].append(block[0])
				inputs_ = [np.array(inputs_[i]) for i in range(dim)]

				outputs_ = np.vstack(outputs)
				predicteds_ = np.vstack(predicteds)
				rewards_ = np.vstack(rewards)

				print("shape: ", np.shape(rewards))

				print("fit model input.shape %s, output.shape %s" %( [inputs_[i].shape for i in range(len(inputs_))], outputs_.shape))
				
				np.set_printoptions(linewidth=200, suppress=True)
				print("currentTargetIndex:", env.currentTargetIndex)

if __name__ == "__main__":
	import sys
	import codecs

	codeListFilename = sys.argv[1]
	modelFilename = sys.argv[2] if len(sys.argv) > 2 else None
	historyFilename = sys.argv[3] if len(sys.argv) > 3 else None

	codeMap = {}
	f = codecs.open(codeListFilename, "r", "utf-8")

	for line in f:
		if line.strip() != "":
			tokens = line.strip().split(",") if not "\t" in line else line.strip().split("\t")
			codeMap[tokens[0]] = tokens[1]

	f.close()

#	env = MarketEnv(dir_path = "./data/", target_codes = codeMap.keys(), input_codes = [], start_date = "2010-08-25", end_date = "2015-08-25", sudden_death = -1.0)
	env = MarketEnv(dir_path = "../../dataset/", target_codes = codeMap.keys(), input_codes = [], start_date = "1514764800", end_date = "1560828600", sudden_death = -1.0, cumulative_reward = True)

	pg = AC_run(env, discount = 0.9, model_filename = modelFilename, history_filename = historyFilename, max_memory=60)
	pg.train(verbose = 1, max_episode=1)
