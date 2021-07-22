from fairseq.data import BaseWrapperDataset
import numpy as np

import logging

logger = logging.getLogger(__name__)


class ResetLanguagePairDataset(BaseWrapperDataset):
    """LanguagePairDataset that resets epoch

    Args:
        dataset (~torch.utils.data.Dataset): dataset to subsample
    """

    def __init__(self, dataset, epoch=1):
        super().__init__(dataset)
        self.set_epoch(epoch)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.dataset.src.set_epoch(epoch)
        self.dataset.tgt.set_epoch(epoch)
