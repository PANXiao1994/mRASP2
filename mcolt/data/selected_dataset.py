from abc import ABC

from fairseq.data import BaseWrapperDataset
from fairseq.data.plasma_utils import PlasmaArray

import numpy as np


class SelectedDataset(BaseWrapperDataset):
    """
    This dataset will select from orignal dataset based index list.
    """
    def __init__(
            self,
            dataset,
            batch_by_size=True,
            seed=0,
            epoch=0,
            selected_index=None,
    ):
        super().__init__(dataset)

        self._cur_epoch = None
        self._cur_indices = None

        self._set_index_list(selected_index, dataset)

        self.batch_by_size = batch_by_size
        self.seed = seed

        self.set_epoch(epoch)

    def _set_index_list(self, selected_index, dataset):
        if selected_index is None:
            self._cur_indices = PlasmaArray(np.arange(len(dataset)))
            self.actual_size = np.int(len(dataset))
        else:
            if isinstance(selected_index, PlasmaArray):
                pass
            elif isinstance(selected_index, np.ndarray):
                selected_index = PlasmaArray(selected_index)
            else:
                raise ValueError(f"The selected_index type {type(selected_index)} does not support")
            self.actual_size = np.int(len(selected_index.array))
            self._cur_indices = selected_index

    def set_index_list(self, selected_index):
        self._set_index_list(selected_index, self.dataset)

    def __len__(self):
        return self.actual_size

    def __getitem__(self, index):
        return self.dataset[self._cur_indices.array[index]]

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

        if epoch == self._cur_epoch:
            return

        self._cur_epoch = epoch
