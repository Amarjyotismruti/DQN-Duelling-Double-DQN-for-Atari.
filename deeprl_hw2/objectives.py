"""Loss functions"""

import tensorflow as tf
import numpy as np

def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    assert max_grad>0.0
    #Find the absolute loss.
    a = abs(y_true - y_pred)
    #condition that chooses the loss types.
    condition = a<max_grad
    linear_loss=max_grad*(a-0.5*max_grad)
    squared_loss=0.5*tf.square(a)

    return tf.where(condition,squared_loss,linear_loss)

    





def mean_huber_loss(y_true, y_pred, max_grad=1.):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """
    loss=huber_loss(y_true,y_pred,max_grad=1.)
    return tf.reduce_mean(loss)
