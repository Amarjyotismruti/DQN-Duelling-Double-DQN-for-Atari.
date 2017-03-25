#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import sys
sys.path.append('/home/amar/Keras-1.2.2')
import argparse
import os
import random
import gym

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ProgbarLogger

from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss, huber_loss
from deeprl_hw2.policy import UniformRandomPolicy, GreedyPolicy, GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy 
from deeprl_hw2.preprocessors import AtariPreprocessor, HistoryPreprocessor
from deeprl_hw2.core import *#ReplayMemory

def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    """Create the Q-network model.

    Use Keras to construct a keras.models.Model instance (you can also
    use the SequentialModel class).

    We highly recommend that you use tf.name_scope as discussed in
    class when creating the model and the layers. This will make it
    far easier to understnad your network architecture if you are
    logging with tensorboard.

    Parameters
    ----------
    window: int
      Each input to the network is a sequence of frames. This value
      defines how many frames are in the sequence.
    input_shape: tuple(int, int)
      The expected input image size.
    num_actions: int
      Number of possible actions. Defined by the gym environment       
    model_name: str
      Useful when debugging. Makes the model show up nicer in tensorboard.

    Returns
    -------
    keras.models.Model
      The Q-model.
    """
    print (input_shape)
    state_input = Input(shape=(input_shape[0],input_shape[1],input_shape[2]*window))
    model=Flatten()(state_input)
    model = Dense(num_actions)(model)

    model_f = Model(input=state_input, output=model)

    return model_f



def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir

import envs
def main():  # noqa: D103
    #(SpaceInvaders-v0
    # Enduro-v0
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')

    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    #parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    #parser.add_argument('--env', default='PendulumSai-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    #args.input_shape = tuple(args.input_shape)

    #args.output = get_output_folder(args.output, args.env)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.
    model_name='linear'
    env = gym.make(args.env)
    num_iter = 2000000
    max_epi_iter = 1000
    
    epsilon = 0.4
    window = 4
    gamma = 0.99
    target_update_freq = 5000
    train_freq = 1
    batch_size = 32
    num_burn_in = 5000
    num_actions = 3 #env.action_space.n
    state_size = (84,84,1)
    new_size = state_size
    max_size = 1000000
    
    lr = 0.00020
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon2 = 1e-08
    decay = 0.0

    u_policy = UniformRandomPolicy( num_actions)
    ge_policy = GreedyEpsilonPolicy(epsilon)
    g_policy = GreedyPolicy() 
    policy = {'u_policy' : u_policy,
            'ge_policy': ge_policy,
            'g_policy': g_policy
            }
    #preprocessor = PreprocessorSequence([AtariPreprocessor(new_size), HistoryPreprocessor(window)])
    preprocessor = AtariPreprocessor(new_size)
    memory = SequentialMemory(max_size=max_size, window_length=window)

    model = create_model(window, state_size, num_actions)   
    print (model.summary())
    dqnA = DQNAgent(q_network=model,
             preprocessor=preprocessor,
             memory=memory,
             policy=policy,
             gamma=gamma,
             target_update_freq=target_update_freq,
             num_burn_in=num_burn_in,
             train_freq=train_freq,
             batch_size=batch_size,model_name=model_name)
    #testing
    #selected_action = dqnA.select_action( np.random.rand(1,210,160,12), train=1, warmup_phase=0)
    h_loss = huber_loss
    optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon2, decay=decay)
    dqnA.compile(optimizer, h_loss)
    #callback1 = ProgbarLogger(count_mode='samples')

    dqnA.fit(env, num_iterations=num_iter, max_episode_length=max_epi_iter)


if __name__ == '__main__':
    main()
