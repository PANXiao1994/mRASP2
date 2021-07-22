from fairseq.data import BaseWrapperDataset, LanguagePairDataset, plasma_utils
import numpy as np

import logging

logger = logging.getLogger(__name__)


class SubsampleLanguagePairDataset(BaseWrapperDataset):
    """Subsamples a given dataset by a specified ratio. Subsampling is done on the number of examples

    Args:
        dataset (~torch.utils.data.Dataset): dataset to subsample
        size_ratio(float): the ratio to subsample to. must be between 0 and 1 (exclusive)
    """
    
    def __init__(self, dataset, size_ratio, weights=None, replace=False, seed=0, epoch=1):
        super().__init__(dataset)
        assert size_ratio <= 1
        self.actual_size = np.ceil(len(dataset) * size_ratio).astype(int)
        logger.info(
            "subsampled dataset from {} to {} (ratio={})".format(
                len(self.dataset), self.actual_size, size_ratio
            )
        )
        self.src_dict = self.dataset.src_dict
        self.tgt_dict = self.dataset.tgt_dict
        self.left_pad_source = self.dataset.left_pad_source
        self.left_pad_target = self.dataset.left_pad_target
        self.seed = seed
        self._cur_epoch = None
        self._cur_indices = None
        self.replace = replace
        if weights is None:
            self.weights = None
        else:
            assert len(weights) == len(dataset)
            weights_arr = np.array(weights, dtype=np.float64)
            weights_arr /= weights_arr.sum()
            self.weights = plasma_utils.PlasmaArray(weights_arr)
        self.set_epoch(epoch)
    
    def __getitem__(self, index):
        index = self._cur_indices.array[index]
        return self.dataset.__getitem__(index)
    
    def __len__(self):
        return self.actual_size
    
    @property
    def sizes(self):
        return self.dataset.sizes[self._cur_indices.array]

    @property
    def src_sizes(self):
        return self.dataset.src_sizes[self._cur_indices.array]

    @property
    def tgt_sizes(self):
        return self.dataset.tgt_sizes[self._cur_indices.array]
    
    @property
    def name(self):
        return self.dataset.name
    
    def num_tokens(self, index):
        index = self._cur_indices.array[index]
        return self.dataset.num_tokens(index)
    
    def size(self, index):
        index = self._cur_indices.array[index]
        return self.dataset.size(index)
    
    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        # sort by target length, then source length
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
        return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
    
    def prefetch(self, indices):
        indices = self._cur_indices.array[indices]
        self.dataset.prefetch(indices)
    
    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False
    
    def set_epoch(self, epoch):
        logger.info("SubsampleLanguagePairDataset.set_epoch: {}".format(epoch))
        super().set_epoch(epoch)
        
        if epoch == self._cur_epoch:
            return
        
        self._cur_epoch = epoch
        
        # Generate a weighted sample of indices as a function of the
        # random seed and the current epoch.
        
        rng = np.random.RandomState(
            [
                42,  # magic number
                self.seed % (2 ** 32),  # global seed
                self._cur_epoch,  # epoch index
            ]
        )
        self._cur_indices = plasma_utils.PlasmaArray(
            rng.choice(
                len(self.dataset),
                self.actual_size,
                replace=self.replace,
                p=(None if self.weights is None else self.weights.array),
            )
        )
        
        logger.info(
            "Dataset is sub-sampled: {} -> {}, first 3 ids are: {}".format(len(self.dataset), self.actual_size,
                                                                           ",".join(
                                                                               [str(_i) for _i in
                                                                                self._cur_indices.array[:3]])))
