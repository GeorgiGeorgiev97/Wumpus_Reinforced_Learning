from utils import *
from random import uniform





class Qlearning:
	def __init__(self, Actions, policy=Epsilon(epsilon=0.2), alpha=0.7, gamma=0.7, q_base=lambda *_: 0):
		self.Q = Function(q_base)
		self.Actions = Actions
		self.PI = policy
		self.S = None
		self.A = None
		self.a = alpha
		self.g = gamma


	def reset(self):
		self.S = None
		self.A = None

	def get_action(self, S):
		return self.PI(S, self.Actions, self.Q)

	def __call__(self, R, S_):
		# Initialize
		if self.S is None:
			self.S = S_
			self.A = self.PI(self.S, self.Actions, self.Q)
			return self.A
		# Mainloop
		else:
			A_ = self.PI(self.S, self.Actions, self.Q)
			self.Q[self.S, self.A] += self.a * (R + self.g * max(self.Q[S_, a] for a in self.Actions(S_)) - self.Q[self.S, self.A])
			self.S = S_
			self.A = A_
			return A_


