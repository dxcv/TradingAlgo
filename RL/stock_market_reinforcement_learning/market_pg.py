import os
import numpy as np

from market_env import MarketEnv
from market_model_builder import MarketPolicyGradientModelBuilder

import csv
import numpy as np 

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class PolicyGradient:

	def __init__(self, env, discount = 0.99, model_filename = None, history_filename = None):
		self.env = env
		self.discount = discount
		self.model_filename = model_filename
		self.history_filename = history_filename

		# 没有利用SGD 
		from keras.optimizers import SGD
		self.model = MarketPolicyGradientModelBuilder(modelFilename).getModel()
		sgd = SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
		# 利用rmsprop 
		self.model.compile(loss='mse', optimizer='rmsprop')

	# 更详细解释: https://blog.csdn.net/heyc861221/article/details/80132054
	def discount_rewards(self, r):
		discounted_r = np.zeros_like(r)
		running_add = 0
		r = r.flatten()

		# 从后向前推算
		for t in reversed(range(0, r.size)):
			# TODO: running_add 为 0 ?
			# 应该是参照pong game,因为游戏里面，只有游戏结束了才有一个reward。
			# 这里是一个reset,每轮游戏结束，即reward非空，重置running_add
			# Reset the running sum at a game boundary.
#			if r[t] != 0:
#				running_add = 0

			# 拆开来就是，run_add 初始为当前reward, 即r[t]
			# run_add = (r[t] * discount + r[t+1]) * discount + r[t+2] + ...
			#		   = r[t] * discount^2 + r[t+1] * discount^1 + r[t+2] + ...
            # 即公式中的。E(Discount^n * Reward)
			running_add = running_add * self.discount + r[t]
			discounted_r[t] = running_add

		return discounted_r

	def train(self, max_episode = 10, max_path_length = 200, verbose = 0):
		env = self.env
		model = self.model
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

			while not game_over:
				aprob = model.predict(observation)[0]
				inputs.append(observation)
				predicteds.append(aprob)
				
				if aprob.shape[0] > 1:
					action = np.random.choice(self.env.action_space.n, 1, p = aprob / np.sum(aprob))[0]

					y = np.zeros([self.env.action_space.n])
					y[action] = 1.

					outputs.append(y)
				else:
					#action = 0 if np.random.uniform() < aprob else 1

					# if aprob = 1.0 reduce it.
					m_aprob = 0.9 if aprob == 1.0 else aprob
					action = 0 if np.random.uniform() < m_aprob else 1

					y = [float(action)]
					outputs.append(y)

				observation, reward, game_over, info = self.env._step(action)
				reward_sum += float(reward)

				rewards.append(float(reward))

				if verbose > 0:
					if env.actions[action] == "LONG" or env.actions[action] == "SHORT":
						color = bcolors.FAIL if env.actions[action] == "LONG" else bcolors.OKBLUE
						print ("%s:\t%s\t%.2f\t%.2f\t" % (info["dt"], color + env.actions[action] + bcolors.ENDC, reward_sum, info["cum"]) + ("\t".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, aprob.tolist())])))
					#write_iter.writerow("%s:\t%s\t%.2f\t%.2f\t" % (info["dt"], env.actions[action], reward_sum, info["cum"]) + ("\t".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, aprob.tolist())])))
					os.system("echo %s >> %s" % ("%s:\t%s\t%.2f\t%.2f\t" % (info["dt"], env.actions[action], reward_sum, info["cum"]) + ("\t".join(["%s:%.2f" % (l, i) for l, i in zip(env.actions, aprob.tolist())])),
							  f_episode))

			#write_iter.close()

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

				discounted_rewards_ = self.discount_rewards(rewards_)
				# TODO: 不做均值平移应该也可以
			#	discounted_rewards_ -= np.mean(discounted_rewards_)
				if np.std(discounted_rewards_) != 0.:
					discounted_rewards_ /= np.std(discounted_rewards_)

				#outputs_ *= discounted_rewards_
				for i, r in enumerate(zip(rewards, discounted_rewards_)):
					reward, discounted_reward = r

					if verbose > 1:
#						print (outputs_[i],)
						print (outputs_[i],)
					
					#outputs_[i] = 0.5 + (2 * outputs_[i] - 1) * discounted_reward
					# 修正output, reward<0 亏钱，反转所有的output。
					# 
					if discounted_reward < 0:
						outputs_[i] = 1 - outputs_[i]
						outputs_[i] = outputs_[i] / sum(outputs_[i])

					# softmax的log函数求导后的Gradient ?
					outputs_[i] = np.minimum(1, np.maximum(0, predicteds_[i] + (outputs_[i] - predicteds_[i]) * abs(discounted_reward)))

					if verbose > 0:
						print (predicteds_[i], outputs_[i], reward, discounted_reward)

				print("fit model input.shape %s, output.shape %s" %( [inputs_[i].shape for i in range(len(inputs_))], [outputs_[i].shape for i in range(len(outputs_))]))
				#print("input dim shape: ",range(len(inputs_)))
				#for i in range(len(inputs_)):
				#	for j in range(len(inputs_[i])):
				#		print(i,j, len(inputs_[i][j]))
				#print("input dim shape end")
				
				np.set_printoptions(linewidth=200, suppress=True)
				print("currentTargetIndex:", env.currentTargetIndex)
				print(inputs_)
				model.fit(inputs_, outputs_, nb_epoch = 1, verbose = 0, shuffle = True)
				model.save_weights(self.model_filename)

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
	env = MarketEnv(dir_path = "../../dataset/", target_codes = codeMap.keys(), input_codes = [], start_date = "1514764800", end_date = "1560828600", sudden_death = -1.0)

	pg = PolicyGradient(env, discount = 0.9, model_filename = modelFilename, history_filename = historyFilename)
	pg.train(verbose = 1)
