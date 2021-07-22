# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
from fairseq.data import BaseWrapperDataset, plasma_utils


logger = logging.getLogger(__name__)


class StrictResamplingDataset(BaseWrapperDataset):
    """Randomly samples from a given dataset at each epoch.

    Get non-overlapped subset across epochs.

    Args:
        dataset (~torch.utils.data.Dataset): dataset on which to sample.
        weights (List[float]): list of probability weights
            (default: None, which corresponds to uniform sampling).
        size_ratio (float): the ratio to subsample to; must be positive. 1.0 means no sampling.
            (default: 1.0).
        batch_by_size (bool): whether or not to batch by sequence length
            (default: True).
        seed (int): RNG seed to use (default: 0).
        epoch (int): starting epoch number (default: 1).
    """

    def __init__(
        self,
        dataset,
        weights=None,
        step_size=2,
        batch_by_size=True,
        seed=0,
        epoch=1,
    ):
        super().__init__(dataset)

        if weights is None:
            self.weights = None

        else:
            assert len(weights) == len(dataset)
            weights_arr = np.array(weights, dtype=np.float64)
            weights_arr /= weights_arr.sum()
            self.weights = plasma_utils.PlasmaArray(weights_arr)

        self.step_size = step_size
        self.batch_by_size = batch_by_size
        self.seed = seed
        self._cur_indices = None
        self.actual_size = len(self.dataset) // self.step_size
        self.set_epoch(epoch)

    def __getitem__(self, index):
        return self.dataset[self._cur_indices.array[index]]

    def __len__(self):
        return self.actual_size

    @property
    def sizes(self):
        if isinstance(self.dataset.sizes, list):
            return [s[self._cur_indices.array] for s in self.dataset.sizes]
        return self.dataset.sizes[self._cur_indices.array]

    def num_tokens(self, index):
        return self.dataset.num_tokens(self._cur_indices.array[index])

    def size(self, index):
        return self.dataset.size(self._cur_indices.array[index])

    def ordered_indices(self):
        if self.batch_by_size:
            order = [
                np.arange(len(self)),
                self.sizes,
            ]  # No need to handle `self.shuffle == True`
            return np.lexsort(order)
        else:
            return np.arange(len(self))

    def prefetch(self, indices):
        self.dataset.prefetch(self._cur_indices.array[indices])

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def set_epoch(self, epoch):
        super().set_epoch(epoch)

        _inner_epoch = self.seed % self.step_size
        self._cur_indices = plasma_utils.PlasmaArray(
            np.remainder(np.arange(0, len(self.dataset), self.step_size)+_inner_epoch, len(self.dataset))
        )
        logger.info(
            "Inner:{}, first 3 ids are: {}".format(_inner_epoch, ",".join([str(_i) for _i in self._cur_indices.array[:3]])))