from spektral.data.loaders import BatchLoader
import numpy as np

def idx_to_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return mask