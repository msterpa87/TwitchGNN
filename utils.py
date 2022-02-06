from libcst import Global
import tensorflow as tf
import numpy as np
from spektral.transforms import LayerPreprocess, AdjToSpTensor
from scipy.sparse import csr_matrix
from spektral.utils import one_hot
from sklearn.model_selection import train_test_split
from spektral.layers import GCNConv
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.datasets import Citation
from tensorflow.keras.losses import CategoricalCrossentropy

def flat_tensor_list(tensor_list):
  return tf.concat([tf.reshape(g, -1) for g in tensor_list], axis=0)

def compute_fisher(model, inputs, y, mask_tr):
  n = mask_tr.sum()

  loss_fn = CategoricalCrossentropy(reduction='none')

  with tf.GradientTape() as tape:
      pred = model(inputs)
      loss = loss_fn(pred[mask_tr], y[mask_tr])

  gradients = tape.jacobian(loss, model.trainable_variables)
  gradient_list = tf.concat([tf.reshape(g, (n, -1)) for g in gradients], axis=1)
  return tf.reduce_mean(gradient_list**2, axis=0)

def apx_fisher(model, inputs, prior_weights, num_samples=50):
  """ Approximate the fisher matrices of the model parameters """
  fisher_matrices = {p: tf.zeros_like(v.value()) for p,v in enumerate(prior_weights)}
  idx = np.random.randint(inputs[0].shape[0])

  # sample u.a.r. num_samples prediction and compute fisher on those
  for i in range(num_samples):
    with tf.GradientTape() as tape:
      pred = model(inputs)[idx]

      # get loglikelihood of predictions
      loglike = tf.nn.log_softmax(pred)
    
    # compute gradient wrt loglikelihood
    loglike_grads = tape.gradient(loglike, prior_weights)

    # approximate i-th Fisher matrix with gradients squared
    for i,g in enumerate(loglike_grads):
      fisher_matrices[i] += tf.reduce_mean(g**2, axis=0)
  
  # return average fisher matrix across samples
  return {p: mat / num_samples for p,mat in fisher_matrices.items()}

def old_penalty_loss(fisher_matrix, current_weights, prior_weights, lambda_=0.1):
  loss = 0

  for u, v, w in zip(fisher_matrix, current_weights, prior_weights):
      loss += tf.reduce_sum(u*tf.square(v-w))
  
  return 0.5 * lambda_ * loss

def _idx_to_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_tasks_new(train_size=0.6, val_size=0.2, seed=0):
  """ divides the Citeseer dataset group by pairs of labels (6 are available in total),
      each node will be classified in one 3 of three classes where the third represents
      the fact that the true class is one of the remaining 4.
      
      Example: select the first two classes
               y = [0,0,0,1,0,0] -> [0,0,1]
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

def load_tasks():
  """ divides the Citeseer dataset group by pairs of labels (6 are available in total """
  dataset = Citation("citeseer", normalize_x=True, transforms=[LayerPreprocess(GCNConv)])

  graph = dataset[0]
  x, adj, y = graph.x, graph.a, graph.y

  # split dataset for continual learning
  y_labels = y.argmax(axis=1)
  tasks = []

  for i in range(3):
    idx = (y_labels == i) | (y_labels == (i+1))
    csr_adj = csr_matrix(adj[idx,:][:,idx])
    sparse_adj = sp_matrix_to_sp_tensor(csr_adj)

    y_sliced = y[idx,:].argmax(axis=1)
    cur_idx = np.arange(y_sliced.shape[0])

    idx_tr, idx_te, _, y_te = train_test_split(cur_idx, y_sliced, train_size=0.6, stratify=y_sliced)
    idx_va, idx_te = train_test_split(idx_te, train_size=0.2, stratify=y_te)

    l = len(cur_idx)
    mask_tr = _idx_to_mask(cur_idx[idx_tr], l)
    mask_va = _idx_to_mask(cur_idx[idx_va], l)
    mask_te = _idx_to_mask(cur_idx[idx_te], l)

    # going from 6 to 2 one-hot encoding
    y1 = one_hot(y.argmax(axis=1)[idx]-i, 2)
        
    tasks.append([(x[idx,:], sparse_adj, y1), (mask_tr, mask_va, mask_te)])

  return tasks

class GlobalAccuracy(tf.keras.metrics.Metric):
  def __init__(self, name="global_accuracy", **kwargs):
    super(GlobalAccuracy, self).__init__()
    self.total_accuracy = self.add_weight(name='total', initializer='zeros')
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
    self.R = self.add_weight(name='total', shape=(num_tasks,num_tasks), initializer='zeros')
    self.count = self.add_weight(name='count', initializer='zeros')

  def update_state(self, task_i, task_j, y_true, y_pred):
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