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
    


def get_hard_target_model_updates(target, source):
    """Return list of target model update ops.

    These are hard target updates. The source weights are copied
    directly to the target network.

    Parameters
    ----------
    target: keras.models.Model
      The target model. Should have same architecture as source model.
    source: keras.models.Model
      The source model. Should have same architecture as target model.

    Returns
    -------
    list(tf.Tensor)
      List of tensor update ops.
    """
    pass


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