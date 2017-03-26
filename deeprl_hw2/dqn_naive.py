import sys
from ipdb import set_trace as debug
from keras.models import model_from_config
from objectives import huber_loss
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Lambda
from keras.models import Model
from copy import deepcopy
import numpy as np
from utils import *
import time
import matplotlib.pyplot as plt
#plt.ion()
tot_rewards = []

def q_pred_m(y_true, y_pred):
    return K.mean(K.mean(y_pred, axis=-1))

def q_pred_d_m(y_true, y_pred):
    return K.mean(K.mean( K.abs(y_pred-y_true) , axis=-1))

def mean_max_tq(y_true, y_pred):
    return K.mean(K.max(y_true, axis=-1))

def mean_max_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))

"""Main DQN agent."""
class DQNAgent_Naive:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size, model_name):
        self.u_policy = policy['u_policy']
        self.g_policy = policy['g_policy']
        self.ge_policy = policy['ge_policy']
        self.model = q_network
        self.num_burn_in = num_burn_in
        self.target_update_freq = target_update_freq
        self.train_freq = train_freq 
        self.batch_size = batch_size
        self.preprocessor=preprocessor
        self.gamma=gamma
        self.process_reward=0
        self.model_name=model_name


    def compile(self, optimizer, loss_func, metrics=[]):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        self.optimizer = optimizer 
        self.loss_func = loss_func 
        metrics += [mean_max_q, mean_max_tq]
        
        self.model.compile(optimizer='adam', loss='mse')
        self.max_grad = 1.0 
        def masked_error(args):
            y_true, y_pred, mask = args
            loss = loss_func(y_pred, y_true, self.max_grad)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)

        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.model.output_shape[1],))
        mask = Input(name='mask', shape=(self.model.output_shape[1],))
        # since we using mask we need seperate layer
        loss_out = Lambda(masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])
        
        trainable_model = Model(input=[self.model.input] + [y_true, mask], output=[loss_out, y_pred])
        prop_metrics = {trainable_model.output_names[1]: metrics}

        losses = [
                lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
                lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
            ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=prop_metrics)
        self.trainable_model = trainable_model
        self.writer=tf.summary.FileWriter("logs/"+self.model_name)


    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        #pass
        q_vals = self.model.predict_on_batch(state)
        return q_vals

    def select_action(self, state, train, warmup_phase, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        #assuming single state, not batch
        q_vals = []
        if train:
            if warmup_phase:
                #uniform ranom
                selected_action = self.u_policy.select_action()
            else:
                #e greedy
                q_vals = self.calc_q_values(state)[0]
                selected_action = self.ge_policy.select_action(q_vals)
        else:
            #greedy
            #TODO need low e greedy?
            q_vals = self.calc_q_values(state)[0]
            selected_action = self.g_policy.select_action(q_vals)

        return selected_action, q_vals


    def fit(self, env, num_iterations, max_episode_length=None, rep_action=4, process_reward=1, num_actions=0):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        episode=0
        iterations=0
        observation=None
        prev_observation=0
        episode_reward=0#None
        episode_iter=0#None
        episode_metric=0
        episode_no=0
        self.step=0 
        self.state=[]
        self.window_length=4
        if num_actions == 0:
            self.nA=3 #env.action_space.n
        else:
            if num_actions < env.action_space.n:
                self.process_action = 1
            self.nA=num_actions
        action_rep=rep_action
        self.process_reward=0#process_reward


        while self.step<num_iterations:
          if self.step%10000==0:
            self.save_weights("parameters/"+self.model_name+"-weights-"+str(self.step)+".h5")
          if observation is None:
            print("new start")
            observation = deepcopy(env.reset())
            self.observation=self.preprocessor.process_state_for_memory(observation)
            self.get_recent_observation(self.observation)
          
          action, q_vals=self.forward()
          reward1=0
          #Take the same action four times to reduce reaction frequency.
          for _ in xrange(action_rep):
             observation, reward0, terminal, info = env.step(action+1)
             if self.process_reward == 1:
                reward0=self.preprocessor.process_reward(reward0)
             reward1+=reward0
             if terminal:
                 break
          env.render()
          
          observation1=deepcopy(observation)
          observation=self.preprocessor.process_state_for_memory(observation1)
          self.curr_state = deepcopy(self.state)
          self.get_recent_observation(observation)#puts next state in self.state
          
          self.action = action
          self.reward = reward1
          self.terminal = terminal
          self.curr_q_vals = q_vals
          
          #Do backward pass parameter update.
          metric = self.backward()
          
          episode_metric += metric
          episode_reward+=reward1
          episode_iter+=1
          self.step+=1

          #End large episodes abruptly.
          if episode_iter>max_episode_length:
            terminal=True
          #Reset environment if terminal.
          if terminal:
            print("Iteration no.-->",self.step,'/',num_iterations)
            print("Episode reward-->", episode_reward)
            episode_no+=1
            #Logging episode metrics.
            save_scalar(episode_no, 'Episode_reward',episode_reward, self.writer)
            save_scalar(episode_no, 'Episode_length',episode_iter, self.writer)
            save_scalar(self.step, 'Action', action, self.writer)
            episode_iter,episode_reward,episode_metric=0,0,0
            observation = None
            self.state = []


          #End large episodes abruptly.

          #Write metrics in tensorflow filewriter.
          """ Loss, mean_q_values, episode_reward, episode_iter
          """
          loss_s=metric[0]
          mean_q_s=metric[1]
          save_scalar(self.step, 'Loss', loss_s, self.writer)
          save_scalar(self.step, 'Mean_Q', mean_q_s, self.writer)
          save_scalar(self.step, 'Action', action, self.writer)
          save_scalar(self.step, 'Iteration_reward', reward1, self.writer)
    
            

    def get_recent_observation(self,observation):
        
        if len(self.state) == 0:
            for i in xrange(self.window_length):
                self.state.insert(0,observation)
        else:
            #pass
            self.state.append(observation)
            self.state = self.state[1:]
            


    def forward(self):

      state=self.preprocessor.process_state_for_network(self.state)
      action, q_vals=self.select_action(state=state, train=True, warmup_phase=False)
      return action, q_vals


    def backward(self):
      if self.step%self.train_freq==0:
        #next state
        n_state=self.preprocessor.process_state_for_network(self.state)
        target_values = self.calc_q_values(n_state)
        
        next_q = np.max(target_values)
        q = np.max(self.curr_q_vals)

        #Used for setting up target batch for training.
        targets = np.zeros((self.batch_size, self.nA))
        dummy_targets = np.zeros((self.batch_size,))
        masks = np.zeros((self.batch_size, self.nA))
        discounted_reward = self.gamma * next_q

        #Set discounted reward to zero for terminal states.
        ter = 1
        if self.terminal:
            ter = 0
        discounted_reward *= ter
        target_qvalue = self.reward + discounted_reward
        #Set up the targets.
        R = target_qvalue
        targets[0,self.action] = R 
        dummy_targets[0] = R
        masks[0,self.action] = 1. 

        state_batch = self.preprocessor.process_batch([self.curr_state])
        metrics = self.trainable_model.train_on_batch([state_batch, targets, masks], [dummy_targets, targets])

        metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]
        return np.array(metrics)











    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        


    def save_weights(self, filepath, overwrite=True):
      """Save network parameters periodically"""
      self.model.save_weights(filepath, overwrite=overwrite)


    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()
