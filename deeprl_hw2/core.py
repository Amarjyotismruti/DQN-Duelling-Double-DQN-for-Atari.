"""Core classes."""
from collections import deque, namedtuple
import random
import numpy as np
import tensorflow

Experience = namedtuple('Experience', 'state, action, reward, next_state, terminal')


class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    def __init__(self,state,action,next_state,reward,terminal=False):

        self.sample=(state,action,next_state,reward,terminal)




class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.

        """
        return state

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.

        """
        return state

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass


class RingBuffer(object):
  """Data structure to store the interaction experience
  All individual states(images) ,actions and rewards will be stored in this container"""

  def __init__(self,max_len):
    self.max_len=max_len
    self.start=0
    self.size=0
    self.data=[None for i in range(max_len)]

  def __len__(self):
    return self.size

  def __getitem__(self,idx):
    # if idx>=self.size or idx<0:
    #   raise IndexError()

    return self.data[(self.start + idx) % self.max_len]

  def append(self,value):

    if self.size<self.max_len:
      self.size+=1

    elif self.size==self.max_len:
      self.start=(self.start + 1) % self.max_len


    self.data[(self.size + self.start -1) % self.max_len ] = value




class ReplayMemory(object):
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw saamples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size=10000, window_length=4):
        """Setup memory.

        You should specify the maximum size of the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.window_length=window_length
        self.max_size=max_size
        #Queue to store the on-line observation to give as input to the network.
        self.current_observation=deque(maxlen=window_length)


    def append(self, observation, action, reward, terminal):
        raise NotImplementedError('This method should be overridden')
    # def end_episode(self, final_state, is_terminal):
    #     raise NotImplementedError('This method should be overridden')

    def sample(self, batch_size, indexes=None):
        raise NotImplementedError('This method should be overridden')

    def clear(self):
        raise NotImplementedError('This method should be overridden')





class SequentialMemory(ReplayMemory):

    def __init__(self, **kwargs):
      super(SequentialMemory, self).__init__(**kwargs)

      self.actions = RingBuffer(self.max_size)
      self.rewards = RingBuffer(self.max_size)
      self.terminals = RingBuffer(self.max_size)
      self.observations = RingBuffer(self.max_size)
      self.memsize=0


    def get_recent_observation(self,recent_observation):

      obs=[recent_observation]
      for idx in range(1, self.window_length):
        obs.insert(0,self.current_observation[self.window_length-idx])
      return np.array(obs)

    def append(self,observation,action,reward,terminal):
      #Add the observations to the replay buffer.
      self.current_observation.append(observation)

      self.actions.append(action)
      self.rewards.append(reward)
      self.observations.append(observation)
      self.terminals.append(terminal)
      self.memsize=len(self.actions)

   
    def sample_batch_indexes(self, low, high, size):

      if high - low >= size:
          r = range(low, high)
          batch_indexes = random.sample(r, size)
      else:
          batch_indexes = np.random.randint(low, high - 1, size=size)

      return np.array(batch_indexes)

    def sample(self, batch_size):
      """Sample batches from the replay memory for training"""
      
      sam = 0
      #Compile the experience as a batch
      batch=[]
      
      while len(batch)<batch_size:
        no_mix = 1
        idx=self.sample_batch_indexes(self.window_length, self.memsize-1, 1)[0]

        #Check for indexes that can go lower than 0.
        if idx<self.window_length:
          idx+=self.window_length

        state=[]
        for i in range(self.window_length):
          state.insert(0,self.observations[idx-i])
          if self.terminals[idx-i]:
            no_mix = 0
            break
        if no_mix == 1:
          next_state=[self.observations[idx-i+1] for i in reversed(range(self.window_length))]
          action=self.actions[idx+1]
          # TODO what happens if the any state or next state is terminal inside a batch? I think we should sample again.(can we ignore it?)
          # TODO is the +1 valid? for reward and terminal state(corrected)
          terminal=self.terminals[idx+1]
          reward=self.rewards[idx+1]

          batch.append(Experience(state, action, reward, next_state, terminal))
        #else:
        #    debug()


      return batch

    def sample_old(self, batch_size, indexes=None):
      """Sample batches from the replay memory for training"""
      
      if indexes==None:
        indexes=self.sample_batch_indexes(0, self.memsize-1, batch_size)

      #Compile the experience as a batch
      batch=[]
      for idx in indexes:
      #Check for indexes that can go lower than 0.
        if idx<self.window_length:
          idx+=self.window_length

        state=[]
        for i in range(self.window_length):
          state.insert(0,self.observations[idx-i])

        next_state=[self.observations[idx-i+1] for i in reversed(range(self.window_length))]
        action=self.actions[idx+1]
        # TODO what happens if the any state or next state is terminal inside a batch? I think we should sample again.(can we ignore it?)
        # TODO is the +1 valid? for reward and terminal state(corrected)
        terminal=self.terminals[idx+1]
        reward=self.rewards[idx+1]

        batch.append(Experience(state, action, reward, next_state, terminal))


      return batch


    def clear(self):
      #Reinitialize the buffer.
      self.actions = RingBuffer(max_size)
      self.rewards = RingBuffer(max_size)
      self.terminals = RingBuffer(max_size)
      self.observations = RingBuffer(max_size)


    def get_config(self):

      config={'window_length':self.window_length,'max_size':max_size}
      return config


if __name__=="__main__":

  memory=SequentialMemory(max_size=4,window_length=3)
  memory.append(1,2,3,False)
  memory.append(4,5,6,False)
  memory.append(7,8,9,False)
  memory.append(10,11,12,False)


  
