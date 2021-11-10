import os
import logging
import numpy as np
from sklearn.metrics import accuracy_score
import torch

class LabelDict():
    def __init__(self, dataset='cifar-10'):
        self.dataset = dataset
        if dataset == 'cifar-10':
            self.label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 
                         4: 'deer',     5: 'dog',        6: 'frog', 7: 'horse',
                         8: 'ship',     9: 'truck'}
        if dataset == 'luna':
            self.label_dict = {0: 'benign', 1: 'malignant'}

        self.class_dict = {v: k for k, v in self.label_dict.items()}

    def label2class(self, label):
        assert label in self.label_dict, 'the label %d is not in %s' % (label, self.dataset)
        return self.label_dict[label]

    def class2label(self, _class):
        assert isinstance(_class, str)
        assert _class in self.class_dict, 'the class %s is not in %s' % (_class, self.dataset)
        return self.class_dict[_class]

def list2cuda(_list):
    array = np.array(_list)
    return numpy2cuda(array)

def numpy2cuda(array):
    tensor = torch.from_numpy(array)

    return tensor2cuda(tensor)

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda().float()

    return tensor

def one_hot(ids, n_class):
    # --------------------- 
    # author：ke1th 
    # source：CSDN 
    # artical：https://blog.csdn.net/u012436149/article/details/77017832 
    b"""
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """

    assert len(ids.shape) == 1, 'the ids should be 1-D'
    # ids = torch.LongTensor(ids).view(-1,1) 

    out_tensor = torch.zeros(len(ids), n_class)

    out_tensor.scatter_(1, ids.cpu().unsqueeze(1), 1.)

    return out_tensor
    
def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()

def evaluate_(_input, _target):
    results = []
    for n in range(_input.shape[-1]):
        input_vec = _input[:,n]
        target_vec = _target[:,n]
        results.append(accuracy_score(target_vec, input_vec))
    return results

def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger

def makedirs(path):
    if not os.path.exists(path):
        print(path)
        os.makedirs(path)

def load_model(model, file_name):
    model.load_state_dict(
            torch.load(file_name, map_location=lambda storage, loc: storage))

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)

def count_parameters(model):
    # copy from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    # baldassarre.fe's reply
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


############################################ utils for computing cka ############################################

def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.
    Gram (correlation) matrix is defined as correlation between 
    actvtivations across test samples.
  Args:
    x: A num_examples x num_features matrix of features.
  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.
  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)
  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.
  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.
  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.
  Returns:
    A symmetric matrix with centered columns and rows.
  """
  if not np.allclose(gram, gram.T):
    raise ValueError('Input must be a symmetric matrix.')
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
  """Compute CKA.
  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.
  Returns:
    The value of CKA between X and Y.
  """
  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

  normalization_x = np.linalg.norm(gram_x)
  normalization_y = np.linalg.norm(gram_y)
  return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
  """Compute CKA with a linear kernel, in feature space.
  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.
  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. CKA may still be
      biased. Note that this estimator may be negative.
  Returns:
    The value of CKA between X and Y.
  """
  features_x = features_x - np.mean(features_x, 0, keepdims=True)
  features_y = features_y - np.mean(features_y, 0, keepdims=True)

  dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
  normalization_x = np.linalg.norm(features_x.T.dot(features_x))
  normalization_y = np.linalg.norm(features_y.T.dot(features_y))

  if debiased:
    n = features_x.shape[0]
    # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
    sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
    sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
    squared_norm_x = np.sum(sum_squared_rows_x)
    squared_norm_y = np.sum(sum_squared_rows_y)

    dot_product_similarity = _debiased_dot_product_similarity_helper(
        dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
        squared_norm_x, squared_norm_y, n)
    normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
        squared_norm_x, squared_norm_x, n))
    normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
        normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
        squared_norm_y, squared_norm_y, n))

  return dot_product_similarity / (normalization_x * normalization_y)