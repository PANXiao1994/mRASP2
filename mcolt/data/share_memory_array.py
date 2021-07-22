import torch
import numpy as np


class ShareMemoryArray(object):
    def __init__(self, array):
        assert isinstance(array, np.ndarray)
        self._array = torch.from_numpy(array).share_memory_()

    @property
    def array(self):
        return self._array.numpy()
