#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import sys
sys.path.append('/home/amar/Keras-1.2.2')
import argparse
import os
import random
import gym

from copy import deepcopy
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ProgbarLogger

from deeprl_hw2.dqn_naive import DQNAgent_Naive
from deeprl_hw2.objectives import mean_huber_loss, huber_loss
from deeprl_hw2.policy import UniformRandomPolicy, GreedyPolicy, GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy 
from deeprl_hw2.preprocessors import AtariPreprocessor, HistoryPreprocessor
from deeprl_hw2.core import *#ReplayMemory

 

def create_model_linear_naive(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    print (input_shape)
    state_input = Input(shape=(input_shape[0],input_shape[1],input_shape[2]*window))
    model=Flatten()(state_input)
    model = Dense(num_actions)(model)

    model_f = Model(input=state_input, output=model)

    return model_f

def load_weights(model, filepath):
    model.load_weights(filepath)
    return model

def forward(preprocessor, state, model, ge_policy ):
  state = preprocessor.process_state_for_network(state)
  q_vals = model.predict_on_batch(state)[0]
  action = ge_policy.select_action(q_vals)
  return action, q_vals

def get_recent_observation(state, observation, window_length):
    if len(state) == 0:
        for i in xrange(window_length):
            state.insert(0,observation)
    else:
        state.append(observation)
        state = state[1:]
    return state

def evaluate(env, ge_policy, preprocessor, model, num_epi, window_length, action_rep):
    tot_rewards = []
    for i in range(num_epi):
        state = []
        epi_reward = 0
        terminal = False
        observation1 = env.reset()
        while(1):
            observation = deepcopy(observation1)
            observation = preprocessor.process_state_for_memory(observation)
            state = get_recent_observation(state, observation, window_length)
            action, q = forward(preprocessor, state, model, ge_policy)
            reward1=0
            for _ in xrange(action_rep):
               observation1, reward0, terminal, info = env.step(action+1)
               reward1+=reward0
               if terminal:
                  break
            if terminal:
               break
            env.render()
            epi_reward += reward1
        print("episode reward", epi_reward)
        tot_rewards.append(epi_reward)
        
    return tot_rewards, np.sum(tot_rewards)/(num_epi + 0.00001)
    
model_name='linear_naive'
env = gym.make('SpaceInvaders-v0')
epsilon = 0.05
window = 4
state_size = (84,84,1)

num_actions = 3
action_rep = 4
epsilon = 0.4
ge_policy = GreedyEpsilonPolicy(epsilon)
preprocessor = AtariPreprocessor(state_size)
model = create_model_linear_naive(window, state_size, num_actions, model_name)   
print (model.summary())

model = load_weights(filepath='/home/sai/parameters/linear_naive-weights-440000.h5', model=model)


#TODO get weights from files and plot the rewards based on the iterations
#TODO find rewards for the final model. average over 100 episodes
num_epi = 3
rewards, avg_reward = evaluate(env, ge_policy, preprocessor, model, num_epi, window_length=window, action_rep=action_rep)

print("average reward", avg_reward)
