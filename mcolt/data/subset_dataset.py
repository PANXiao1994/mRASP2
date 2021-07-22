from fairseq.data import BaseWrapperDataset, LanguagePairDataset

import numpy as np

from .share_memory_array import ShareMemoryArray


class SubSetDataset(BaseWrapperDataset):
    """
    This dataset will select from orignal dataset based index list.
    """
    def __init__(
            self,
            dataset,
            split_num,
            split_index,
    ):
        super().__init__(dataset)
        total_size = len(dataset)
        self.batch_by_size = True
        self.actual_size = (total_size // split_num) + (total_size % split_num > split_index)
        self._indexes = ShareMemoryArray(np.arange(self.actual_size) * split_num + split_index)

    @property
    def indexes(self):
        return self._indexes.array

    def __len__(self):
        return self.actual_size

    def __getitem__(self, index):
        return self.dataset[self.indexes[index]]

    @property
    def sizes(self):
        return self.dataset.sizes[self.indexes]

    def num_tokens(self, index):
        return self.dataset.num_tokens(self.indexes[index])

    def size(self, index):
        return self.dataset.size(self.indexes[index])

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
        self.dataset.prefetch(self.indexes[indices])


def SubsetLanguagePairDataset(dataset, split_num, split_index):
    assert isinstance(dataset, LanguagePairDataset)
    assert isinstance(split_num, int) and isinstance(split_index, int)
    assert split_num > 0
    assert 0 <= split_index < split_num
    src = SubSetDataset(dataset.src, split_num=split_num, split_index=split_index)
    if dataset.tgt is not None:
        tgt = SubSetDataset(dataset.tgt, split_num=split_num, split_index=split_index)
        tgt_sizes = tgt.sizes
    else:
        tgt = None
        tgt_sizes = None
    return LanguagePairDataset(
        src, src.sizes, dataset.src_dict,
        tgt, tgt_sizes, dataset.tgt_dict,
        left_pad_source=dataset.left_pad_source,
        left_pad_target=dataset.left_pad_target,
    )