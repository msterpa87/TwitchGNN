import tensorflow as tf
import numpy as np
from spektral.transforms import LayerPreprocess, AdjToSpTensor
from spektral.utils import one_hot
from spektral.layers import GCNConv
from spektral.datasets import Citation
from tensorflow.keras.losses import CategoricalCrossentropy

def flat_tensor_list(tensor_list):
  """ equivalent of np.flatten() on a list of tensors

  Args:
      tensor_list (list of tensors): the list of tensors to be flatted

  Returns:
      tensor: a single tensor containing all the elements given in input
  """
  return tf.concat([tf.reshape(g, -1) for g in tensor_list], axis=0)

def compute_fisher(model, inputs, y, mask_tr):
  """ Returns the square of gradients computing the loss function
      of model on the nodes in y[mask_tr]

  Args:
      model (tf.keras.Model): an instantiated keras model
      inputs (pair of tensors): node features and adjacency matrix
      y (tensor): labels
      mask_tr (list of bool): training mask

  Returns:
      [type]: [description]
  """
  n = mask_tr.sum()

  loss_fn = CategoricalCrossentropy(reduction='none')

  with tf.GradientTape() as tape:
      pred = model(inputs)
      loss = loss_fn(pred[mask_tr], y[mask_tr])

  gradients = tape.jacobian(loss, model.trainable_variables)
  gradient_list = tf.concat([tf.reshape(g, (n, -1)) for g in gradients], axis=1)
  return tf.reduce_mean(gradient_list**2, axis=0)

def _idx_to_mask(idx, l):
  """ Computes a boolean mask given a set of indices

  Args:
      idx (list of int): indices list
      l (int): the size of the array from which the indices come from

  Returns:
      list of bool: output mask
  """
  mask = np.zeros(l)
  mask[idx] = 1
  return np.array(mask, dtype=np.bool)

def load_tasks(train_size=0.6, val_size=0.2, seed=0):
  """ divides the Citeseer dataset group by pairs of labels (6 are available in total),
      each node will be classified in one 3 of three classes where the third represents
      the fact that the true class is one of the remaining 4.
      
      Example: select the first two classes
               y = [0,0,0,1,0,0] -> [0,0,1]

  Args:
      train_size (float, optional): The size of the training set. Defaults to 0.6.
      val_size (float, optional): The size of the validation set. Defaults to 0.2.
      seed (int, optional): Numpy seed for reproducibility. Defaults to 0.

  Returns:
      list of tuples: returns a list where each element corresponds to a task
                      and is made of two tuples:
                      - (node features, adjacency matrix, labels)
                      - 3 masks, resp. training, validation, test
  """
  dataset = Citation("citeseer", normalize_x=True, transforms=[LayerPreprocess(GCNConv), AdjToSpTensor()])

  graph = dataset[0]
  x, adj, y = graph.x, graph.a, graph.y

  # split dataset for continual learning
  y_labels = y.argmax(axis=1)
  n = len(y_labels)
  tasks = []

  np.random.seed(seed)

  for i in range(3):
    idx1 = (y_labels == i)
    idx2 = (y_labels == i+1)
    idx3 = (y_labels != i) & (y_labels != i+1)

    y_new = np.zeros(n)
    y_new[idx1] = 0
    y_new[idx2] = 1
    y_new[idx3] = 2
    y_new = one_hot(y_new, 3)

    mask_idx = np.arange(n)
    np.random.shuffle(mask_idx)

    n_tr, n_va = int(n*train_size), int(n*val_size)
    idx_tr, idx_va, idx_te = mask_idx[:n_tr], mask_idx[n_tr:(n_tr+n_va)], mask_idx[(n_tr+n_va):]

    mask_tr = _idx_to_mask(idx_tr, n)
    mask_va = _idx_to_mask(idx_va, n)
    mask_te = _idx_to_mask(idx_te, n)
        
    tasks.append([(x, adj, y_new), (mask_tr, mask_va, mask_te)])

  return tasks

class GlobalAccuracy(tf.keras.metrics.Metric):
  def __init__(self, name="global_accuracy", **kwargs):
    super(GlobalAccuracy, self).__init__()
    # cumulative accuracy
    self.total_accuracy = self.add_weight(name='total', initializer='zeros')

    # counter to compute final average
    self.count = self.add_weight(name='count', initializer='zeros')

  def update_state(self, y_true, y_pred):
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    val = tf.reduce_mean(tf.cast(y_true == y_pred, self.dtype))
    self.total_accuracy.assign_add(val)
    self.count.assign_add(1)

  def result(self):
    return self.total_accuracy / self.count

class BackwardTransfer(tf.keras.metrics.Metric):
  def __init__(self, num_tasks, name="backward_transfer", **kwargs):
    super(BackwardTransfer, self).__init__()
    # accuracy matrix
    self.R = self.add_weight(name='total', shape=(num_tasks,num_tasks), initializer='zeros')

    # to keep track of the max number of training tasks
    self.count = self.add_weight(name='count', initializer='zeros')

  def update_state(self, task_i, task_j, y_true, y_pred):
    """ Updates the accuracy matrix and the counter with the accuracy on
        task i after training on task j

    Args:
        task_i (int): test task
        task_j (int): training task
        y_true (int matrix): one-hot encoding of true labels
        y_pred (int matrix): logits of prediction
    """
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    val = tf.reduce_mean(tf.cast(y_true == y_pred, self.dtype))
    self.R[task_i, task_j].assign(val)
    self.count.assign(tf.maximum(self.count, task_j))

  def result(self):
    t = self.count
    total = 0
    shape = self.R.shape

    for j in range(1, shape[1]):
      for i in range(j):
        total += (self.R[i,j] - self.R[i,i]).numpy()
    
    return total / (0.5 * t*(t-1))