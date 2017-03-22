"""Common functions you may find useful in your implementation."""

import semver
import tensorflow as tf
import keras.optimizers as optimizers



def get_uninitialized_variables(variables=None):
    """Return a list of uninitialized tf variables.

    Parameters
    ----------
    variables: tf.Variable, list(tf.Variable), optional
      Filter variable list to only those that are uninitialized. If no
      variables are specified the list of all variables in the graph
      will be used.

    Returns
    -------
    list(tf.Variable)
      List of uninitialized tf variables.
    """
    sess = tf.get_default_session()
    if variables is None:
        variables = tf.global_variables()
    else:
        variables = list(variables)

    if len(variables) == 0:
        return []

    if semver.match(tf.__version__, '<1.0.0'):
        init_flag = sess.run(
            tf.pack([tf.is_variable_initialized(v) for v in variables]))
    else:
        init_flag = sess.run(
            tf.stack([tf.is_variable_initialized(v) for v in variables]))
    return [v for v, f in zip(variables, init_flag) if not f]


def get_soft_target_model_updates(target, source, tau):
    """Return list of target model update ops.

    These are soft target updates. Meaning that the target values are
    slowly adjusted, rather than directly copied over from the source
    model.

    The update is of the form:

    $W' \gets (1- \tau) W' + \tau W$ where $W'$ is the target weight
    and $W$ is the source weight.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.
    tau: float
      The weight of the source weights to the target weights used
      during update.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    target_params = target.trainable_weights + sum([i.non_trainable_weights for i in target.layers], [])
    source_params = source.trainable_weights + sum([i.non_trainable_weights for i in source.layers], [])

    update = []
    for T, S in zip(target_params, source_params):
        update.append((T, tau * S + (1. - tau) * T))
    return update
    


class UpdatesOptimizer(optimizers.Optimizer):
    def __init__(self, optimizer, extra_updates):
        super(UpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.extra_updates = extra_updates

    def get_updates(self, params, constraints, loss):
        updates = self.optimizer.get_updates(params, constraints, loss)
        updates += self.extra_updates
        self.updates = updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()


#Callbacks for the models.
from keras.callbacks import Callback as keras_callback, CallbackList as KerasCallbackList

class Callback(keras_callback):
    def _set_env(self, env):
        self.env = env

    def on_episode_begin(self, episode, logs={}):
        pass

    def on_episode_end(self, episode, logs={}):
        pass

    def on_step_begin(self, step, logs={}):
        pass

    def on_step_end(self, step, logs={}):
        pass

    def on_action_begin(self, action, logs={}):
        pass

    def on_action_end(self, action, logs={}):
        pass

def save_scalar(step, name, value, writer):
  """Save a scalar value to tensorboard.
  
  Parameters
  ----------
  step: int
    Training step (sets the position on x-axis of tensorboard graph.
  name: str
    Name of variable. Will be the name of the graph in tensorboard.
  value: float
    The value of the variable at this step.
  writer: tf.FileWriter
    The tensorboard FileWriter instance.
  """
  summary = tf.Summary()
  summary_value = summary.value.add()
  summary_value.simple_value = float(value)
  summary_value.tag = name
  writer.add_summary(summary, step)