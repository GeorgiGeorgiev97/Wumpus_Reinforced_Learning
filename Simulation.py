from tqdm import trange
from os import system
from time import sleep



class Simulation:
	"""A Simulation simulates multiple Episodes"""
	def __init__(self, env, agent, state_interpretation):
		self.env = env
		self.agent = agent
		self.interpretation = state_interpretation

	def __call__(self, n):

		for i in trange(n):
			done = False
			reward = 0
			score = 0
			state = self.env.reset()
			while not done:
				action = self.agent(reward, self.interpretation(state))
				state, reward, done, _ = self.env.step(action)
				reward = reward[0]
				score += reward
			self.agent(reward, self.interpretation(state)) # Agent needs information about the final state
			self.agent.reset()

	def show(self):
		self.env.reset()
		done = False
		state = self.env.reset()
		while not done:
			system('clear')
			self.env.render()
			action = self.agent.get_action(self.interpretation(state))
			print('Action:', action)
			state, reward, done, _ = self.env.step(action)
			sleep(0.1)

		self.env.render()