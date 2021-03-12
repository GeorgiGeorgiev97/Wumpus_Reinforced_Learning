
import gym
import fh_ac_ai_gym
from Simulation import Simulation
from Agent import *
from utils import *
from os import system




# Q Learning
action_function = lambda state: [0, 1, 2, 3, 4, 5]

alpha = 0.5
gamma = 1.0
epsilon = 0.3
episodes = 15000


policy = Epsilon(epsilon)
interpretation = lambda obs: ', '.join(str(obs[key]) for key in ['x', 'y', 'gold', 'glitter', 'stench', 'direction'])


# Environment
env = gym.make('Wumpus-v0')

agent = Qlearning(action_function, policy, alpha, gamma, lambda *_: 0)


# Simulation
Simulation = Simulation(env, agent, interpretation)
Simulation(episodes)

# Lets have a look
while input('Print episode? y/n') != 'n':
    Simulation.show()
