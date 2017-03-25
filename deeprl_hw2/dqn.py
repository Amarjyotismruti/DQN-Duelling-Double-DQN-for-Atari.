import sys
# sys.path.append('/home/amar/Keras-1.2.2')
from keras.models import model_from_config
from deeprl_hw2.objectives import huber_loss
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Lambda
from keras.models import Model
from copy import deepcopy
import numpy as np
from deeprl_hw2.utils import *

def q_pred_m(y_true, y_pred):
    return K.mean(K.mean(y_pred, axis=-1))

def q_pred_d_m(y_true, y_pred):
    return K.mean(K.mean( K.abs(y_pred-y_true) , axis=-1))

def mean_max_tq(y_true, y_pred):

    return K.mean(K.max(y_true, axis=-1))

def mean_max_q(y_true, y_pred):

    return K.mean(K.max(y_pred, axis=-1))

"""Main DQN agent."""
class DQNAgent:
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
        self.memory = memory
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
        #metrics += [q_pred_m, mean_max_q]
        #metrics += [q_pred_d_m]
        
        # create target network with random optimizer
        config = {
            'class_name': self.model.__class__.__name__,
            'config': self.model.get_config(),
        }
        self.target = model_from_config(config, custom_objects={})#custom_objects)
        self.target.set_weights(self.model.get_weights())
        self.target.compile(optimizer='adam', loss='mse')
        self.model.compile(optimizer='adam', loss='mse')
        

        #Update the target network using soft updates.
        if self.target_update_freq < 1.:
          updates = get_soft_target_model_updates(self.target, self.model, self.target_update_freq)
          optimizer = UpdatesOptimizer(optimizer, updates)

        #TODO: target model weights update sperately while updating network
        self.max_grad = 1.0 
        def masked_error(args):
            y_true, y_pred, mask = args
            #loss = loss_func(y_true, y_pred, self.max_grad)
            loss = loss_func(y_pred, y_true, self.max_grad)
            loss *= mask  # apply element-wise mask
            #print loss
            return K.sum(loss, axis=-1)

        y_pred = self.model.output
        y_true = Input(name='y_true', shape=(self.model.output_shape[1],))
        mask = Input(name='mask', shape=(self.model.output_shape[1],))
        # since we using mask we need seperate layer
        loss_out = Lambda(masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])
        
        trainable_model = Model(input=[self.model.input] + [y_true, mask], output=[loss_out , y_pred])
        prop_metrics = {trainable_model.output_names[1]: metrics}

        # TODO not sure why this is needed
        losses = [
                lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
                lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
            ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=prop_metrics)
        self.trainable_model = trainable_model
        self.writer=tf.summary.FileWriter("logs/"+self.model_name)
        #self.load_weights('parameters/linear-weights-0.h5')

        #def get_activations(model, layer, X_batch):
        #    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
        #    activations = get_activations([X_batch,0])
        #    return activations





    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        #pass
        #TODO process state
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
        #
        #states wont be in batch 
        #train + warmup_phase
        #
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

        return selected_action


    def fit(self, env, num_iterations, max_episode_length=None):
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
        episode_reward=0
        episode_iter=0
        episode_metric=0
        episode_no=0
        self.step=0 
        self.nA=3#env.action_space.n
        action_rep=4


        while self.step<num_iterations:
          #TODO respect max_episode length

          if observation is None:
          #Initiate training.(warmup phase to fill up the replay memory)
            print("Warmup start")
            observation = deepcopy(env.reset())

            for _ in xrange(self.num_burn_in):
              observation1=deepcopy(observation)
              action = self.select_action(observation1, train=True, warmup_phase=True)
              reward1=0
              for _ in xrange(action_rep):
                 observation, reward0, terminal, info = env.step(action+1)

                 if self.process_reward == 1:
                    reward0=self.preprocessor.process_reward(reward0)
                 reward1+=reward0
                 if terminal:
                    break
              
              self.observation=self.preprocessor.process_state_for_memory(observation)
              self.memory.append(self.observation,action,reward1,terminal)

              if terminal:
                observation=deepcopy(env.reset())
            print("Warmup end.")
                # TODO doesnt make sense to break here
                #break
          #Save model parameters.
          if self.step%100000==0:
            self.save_weights("parameters/"+self.model_name+"-weights-"+str(self.step)+".h5")

          action=self.forward(self.observation)
          reward1=0
          #Take the same action four times to reduce reaction frequency.
          for _ in xrange(action_rep):
             observation, reward0, terminal, info = env.step(action+1)
             if self.process_reward == 1:
                reward0=self.preprocessor.process_reward(reward0)
             reward1+=reward0
             if terminal:
                 break
          #Add the sample to replay memory.
          #env.render()
          observation1=deepcopy(observation)
          self.observation=self.preprocessor.process_state_for_memory(observation1)
          self.memory.append(self.observation,action,reward1,terminal)

          #Do backward pass parameter update.
          metric = self.backward()
          episode_reward+=reward1
          episode_iter+=1
          self.step+=1


            


          #End large episodes abruptly.
          if episode_iter>max_episode_length:
            terminal=True
          #Reset environment if terminal.
          if terminal:
            print("Iteration no.-->",self.step,"/",num_iterations)
            print("Episode reward-->",episode_reward)
            episode_no+=1
            #Logging episode metrics.
            save_scalar(episode_no, 'Episode_reward',episode_reward, self.writer)
            save_scalar(episode_no, 'Episode_length',episode_iter, self.writer)
            episode_iter,episode_reward,episode_metric=0,0,0
            observation=env.reset()
            observation = self.preprocessor.process_state_for_memory(observation)



          #Write metrics in tensorflow filewriter.
          """ Loss, mean_q_values, episode_reward, episode_iter
          """
          loss_s=metric[0]
          mean_q_s=metric[1]
          save_scalar(self.step, 'Loss', loss_s, self.writer)
          save_scalar(self.step, 'Mean_Q', mean_q_s, self.writer)
    
            




    def forward(self, observation):

      state=self.memory.get_recent_observation(observation)
      state=self.preprocessor.process_state_for_network(state)
      action=self.select_action(state=state, train=True, warmup_phase=False)
      return action


    def backward(self):
      #self.memory.append(self.recent_observation, self.recent_action, reward, terminal)

      if self.step%self.train_freq==0:

        experiences = self.memory.sample(self.batch_size)

        #Extract the parameters out of experience data structure.
        state_batch = []
        reward_batch = []
        action_batch = []
        terminal_batch = []
        next_state_batch = []
        for exp in experiences:
            state_batch.append(exp.state)
            next_state_batch.append(exp.next_state)
            reward_batch.append(exp.reward)
            action_batch.append(exp.action)
            terminal_batch.append(0. if exp.terminal else 1.)

        state_batch = self.preprocessor.process_batch(state_batch)
        next_state_batch = self.preprocessor.process_batch(next_state_batch)
        terminal_batch=np.array(terminal_batch)
        reward_batch=np.array(reward_batch)

        #compute target q_values
        # TODO add rewards
        target_values = self.target.predict_on_batch(next_state_batch)    
        q_batch = np.max(target_values, axis=1).flatten()

        #Used for setting up target batch for training.
        targets = np.zeros((self.batch_size, self.nA))
        dummy_targets = np.zeros((self.batch_size,))
        masks = np.zeros((self.batch_size, self.nA))
        discounted_reward_batch = self.gamma * q_batch
 


        #Set discounted reward to zero for terminal states.
        discounted_reward_batch *= terminal_batch
        target_qvalue=reward_batch + discounted_reward_batch

        #Set up the targets.
        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, target_qvalue, action_batch)):
          target[action] = R 
          dummy_targets[idx] = R
          mask[action] = 1. 

        metrics = self.trainable_model.train_on_batch([state_batch, targets, masks], [dummy_targets, targets])

        if self.target_update_freq>=1 and self.step % self.target_update_freq == 0:
          self.update_target_model_hard()
        metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]
        return np.array(metrics)



    def update_target_model_hard(self):

      self.target.set_weights(self.model.get_weights())








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
        


    def save_weights(self, filepath ,overwrite=True):
      """Save network parameters periodically"""
      self.model.save_weights(filepath, overwrite=overwrite)


    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()
