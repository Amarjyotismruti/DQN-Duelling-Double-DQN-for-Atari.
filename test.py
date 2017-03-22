import envs
import gym

env = gym.make('PendulumSai-v0')
env.reset()
#env.step(1)

import IPython
IPython.embed()
