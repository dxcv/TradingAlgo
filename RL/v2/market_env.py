from random import random
import numpy as np
import math

import gym
from gym import spaces

import csv

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		pass

	try:
		import unicodedata
		unicodedata.numeric(s)
		return True
	except (TypeError, ValueError):
		pass

	return False

class MarketEnv(gym.Env):

	# 手续费
	PENALTY = 1 #0.999756079

	def __init__(self, dir_path, target_codes, input_codes, start_date, end_date, scope = 60, sudden_death = -1., cumulative_reward = False):
		self.startDate = start_date
		self.endDate = end_date
		self.scope = scope
		self.sudden_death = sudden_death
		self.cumulative_reward = cumulative_reward

		self.inputCodes = []
		self.targetCodes = []
		self.dataMap = {}

		self.lastsum=0.

#		for code in (target_codes + input_codes):
		for code in (list(target_codes) + list(input_codes)):
			fn = dir_path + "./" + code + ".csv"

			data = {}
			lastClose = 0
			lastVolume = 0

			open_total = 0.
			high_total = 0.
			low_total = 0.
			close_total = 0.
			volume_total = 0.

			try:
				f = open(fn, "r")
				for line in f:
					if line.strip() != "":
#						dt, openPrice, high, low, close, volume = line.strip().split(",")
						dt, close, high, low, openPrice, volume = line.strip().split(",")

						if is_number(close) == False and is_number(high) == False:
							continue

						try:
							if dt >= start_date:						
								high = float(high) if high != "" else float(close)
								low = float(low) if low != "" else float(close)
								close = float(close)
							#	volume = int(volume)
								volume = float(volume)

							#	if lastClose > 0 and close > 0 and lastVolume > 0:
							#		close_ = (close - lastClose) / lastClose
							#		high_ = (high - close) / close
							#		low_ = (low - close) / close
							#		volume_ = (volume - lastVolume) / lastVolume
							#		
							#		data[dt] = (high_, low_, close_, volume_)
							#	lastClose = close
							#	lastVolume = volume

								data[dt] = (high, low, close, volume)
								high_total += high
								low_total += low
								close_total += close 
								volume_total += volume 

						except Exception as e:
							print (e, line.strip().split(","))
				f.close()
			except Exception as e:
				print (e)

			# dump close
			with open('env_data_processed.csv','w') as f:
				w = csv.writer(f)
				w.writerows(data.items())

			## !! TODO: remove if use ratio price
			mean_high = high_total / len(data)
			mean_low = low_total / len(data)
			mean_close = close_total / len(data)
			mean_volume = volume_total / len(data)
			for k,v in data.items():
				## TODO: HARD CODED !!!
				data[k] = (v[0]/mean_high, v[1]/mean_low, v[2]/mean_close, v[3]/mean_volume)

			with open("en_data.txt", "w") as txtFile:
				for line in data:
					txtFile.write(line + "\n")

			#print(len(data.keys()))
			if len(data.keys()) > scope:
				self.dataMap[code] = data
				if code in target_codes:
					self.targetCodes.append(code)
				if code in input_codes:
					self.inputCodes.append(code)

		self.actions = [
			"LONG",
#			"HOLD",
			"SHORT",
		]

		self.action_space = spaces.Discrete(len(self.actions))
		self.observation_space = spaces.Box(np.ones(scope * (len(input_codes) + 1)) * -1, np.ones(scope * (len(input_codes) + 1)))

#		self.reset()
		self._reset()
		self._seed()

	def _step(self, action):
		if self.done:
			return self.state, self.reward, self.done, {}

		self.reward = 0

		# 如果一直LONG，或者一直SHORT，返回的reward会一直为0，导致梯度无法更新。
		# 增加判断，在一直LONG或SHORT的时候，手中的position价值是否增加或者减少。
		# 即，当前position的sum与上次的sum(lastsum)的差值
		# 从而保证，每次step时，都可以更新梯度
		self.actual_reward = 0

		# 每次非LONG即SHORT
		# 如果是SHORT,会一直是SHORT(boughts列表里的数全部小于0),
		# 如果是LONG，应该一直是LONG(boughts列表里的数全部大于0)。
		# 但，SHORT中途，发生LONG，会清空SHORT的postion(给Boughts位置空list)，重置boughts后，再LONG 
		# 这样boughts > 0, 代表所持有的position为LONG；< 0，代表position为Short
		if self.actions[action] == "LONG":
			# 之前是否持有SHORT position
			if sum(self.boughts) < 0:
				for b in self.boughts:
					# b是负数，因为是SHORT，b应该小于-1才有钱赚。然后在前面加负号，转成正值
					self.reward += -(b + 1)
				if self.cumulative_reward:
					self.reward = self.reward / max(1, len(self.boughts))

				# 止跌, reward已经低于阈值, DONE
				if self.sudden_death * len(self.boughts) > self.reward:
					self.done = True

				self.boughts = []
				#更新实际盈亏
				self.actual_reward = self.reward

			# 之前也是LONG
			if sum(self.boughts) > 0:
				self.reward = sum(self.boughts) - self.lastsum
				if self.cumulative_reward:
					self.reward = self.reward / max(1, len(self.boughts))

			# 买1个
			self.boughts.append(1.0)
		elif self.actions[action] == "SHORT":
			# 之前是否持有LONG position
			if sum(self.boughts) > 0:
				for b in self.boughts:
					# b是正数，因为是LONG，b应该大于1，才有钱赚。
					self.reward += b - 1
				if self.cumulative_reward:
					self.reward = self.reward / max(1, len(self.boughts))

				# 止跌, reward已经低于阈值, DONE
				if self.sudden_death * len(self.boughts) > self.reward:
					self.done = True

				self.boughts = []
				#更新实际盈亏
				self.actual_reward = self.reward

			# 之前也是SHORT
			if sum(self.boughts) < 0:
				self.reward = (self.lastsum - sum(self.boughts))
		 		#self.reward = min(self.lastsum - sum(self.boughts), - (sum(self.boughts) + len(self.boughts)))
				if self.cumulative_reward:
					self.reward = self.reward / max(1, len(self.boughts))

			# 卖1个
			self.boughts.append(-1.0)
		else:
#			if self.actions[action] == "HOLD":
#				self.reward = sum(self.boughts) - self.lastsum
#				if self.cumulative_reward:
#					self.reward = self.reward / max(1, len(self.boughts))

			pass

		self.lastsum = sum(self.boughts)

		# close定义为，C(t+1) - C(t) / C(t) = C(t+1)/C(t) - 1
		# vari即为当前close值, 代入 cum 得到
		# cum = cum * (1 + vari) = cum * ( C(t+1)/C(t) ) 
		# 下一次更新cum时， cum = cum * C(t+1) / C(t) * C(t+2) /C(t+1) = cum * C(t+2) / C(t)
		# 这样，cum始终为当前Close值与初始值的比例值
		vari = self.target[self.targetDates[self.currentTargetIndex]][2]
		self.cum = self.cum * (1 + vari)

		# 对每个step, 每次都要更新所持有position
		# LONG的情况下，因为是+1, 
		# SHORT的情况下，因为append的是-1, 先乘上-1变为正值后，再乘以vari加1
		# 类似与上面的cum更新。
		# 保证所有bought 都更新为 boughts[i] * (1+vari) * penalty
		#                         ~~~~~~~~~~~~~~~~~~~~~
		for i in range(len(self.boughts)):
			self.boughts[i] = self.boughts[i] * MarketEnv.PENALTY * (1 + vari * (-1 if sum(self.boughts) < 0 else 1))

		self.defineState()
		# currentTargetIndex 初始为scope, 用过去scope里面的data计算，
		# 然后每次+1, 直到结束
		self.currentTargetIndex += 1
		if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[self.currentTargetIndex]:
			self.done = True

		if self.done:
			for b in self.boughts:
				self.reward += (b * (1 if sum(self.boughts) > 0 else -1)) - 1
			if self.cumulative_reward:
				self.reward = self.reward / max(1, len(self.boughts))

			self.boughts = []
			self.actual_reward = self.reward

		return self.state, self.reward, self.actual_reward, self.done, {"dt": self.targetDates[self.currentTargetIndex], "cum": self.cum, "code": self.targetCode}

	def _reset(self):
		self.targetCode = self.targetCodes[int(random() * len(self.targetCodes))]
		self.target = self.dataMap[self.targetCode]
		self.targetDates = sorted(self.target.keys())
#		self.currentTargetIndex = self.scope
		self.currentTargetIndex = 0
		self.boughts = []
		self.cum = 1.

		self.done = False
		self.reward = 0
		self.actual_reward = 0

		self.defineState()

		return self.state

	def _render(self, mode='human', close=False):
		if close:
			return
		return self.state

	'''
	def _close(self):
		pass

	def _configure(self):
		pass
	'''

	def _seed(self):
		return int(random() * 100)

	def defineState(self):
		tmpState = []

		budget = (sum(self.boughts) / len(self.boughts)) if len(self.boughts) > 0 else 1.
		size = math.log(max(1., len(self.boughts)), 100)
		position = 1. if sum(self.boughts) > 0 else 0.
		tmpState.append([[budget, size, position]])

		subject = []
		subjectVolume = []
		for i in range(self.scope):
			try:
				# data[dt] = (high_, low_, close_, volume_)
				# 从后往前一次存储 close/volume的pair, state其实就是这些pair
				subject.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][2]])
				subjectVolume.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][3]])
			except Exception as e:
				print (self.targetCode, self.currentTargetIndex, i, len(self.targetDates))
				self.done = True
		tmpState.append([[subject, subjectVolume]])

		# tmpState 有2行，第一行是[budget, size, position]]
		# 第二行是[subject, subjectVolume]
		tmpState = [np.array(i) for i in tmpState]
		self.state = tmpState


	def step_episode_data(self, steps):
		if self.currentTargetIndex + steps >= len(self.targetDates) or self.endDate <= self.targetDates[self.currentTargetIndex]:
			self.done = True

		tmpState = []

		episode_data = []
		for i in range(steps):
			try:
				# data[dt] = (high_, low_, close_, volume_)
				# 从后往前一次存储 close/volume的pair, state其实就是这些pair
				episode_data.append(
					[self.target[self.targetDates[self.currentTargetIndex]][2],
					self.target[self.targetDates[self.currentTargetIndex]][3]])
				self.currentTargetIndex += 1
			except Exception as e:
				print ("no more data", self.targetCode, self.currentTargetIndex, i, len(self.targetDates))
				self.done = True
				break

		#print("data",self.currentTargetIndex,"/" ,len(self.target))
		return episode_data, self.done
