import sys
import types

import numpy as np

from mcolt.data.data_utils_fast import batch_by_size_fast_return_index


class DynamicSampler(object):
    def __init__(self,
                 indices,
                 num_tokens_fn,
                 max_tokens,
                 max_sentences,
                 required_batch_size_multiple):
        self.indices = indices
        self._index = self._init_index_buffer(
            indices,
            num_tokens_fn,
            max_tokens,
            max_sentences,
            required_batch_size_multiple
        )
        self._now = 0

    def _init_index_buffer(self,
                           indices,
                           num_tokens_fn,
                           max_tokens,
                           max_sentences,
                           required_batch_size_multiple
                           ):
        max_tokens = max_tokens if max_tokens is not None else sys.maxsize
        max_sentences = max_sentences if max_sentences is not None else sys.maxsize
        if isinstance(indices, types.GeneratorType):
            indices = np.fromiter(indices, dtype=np.int64, count=-1)
        batch_sampler_index = batch_by_size_fast_return_index(
            indices, num_tokens_fn, max_tokens, max_sentences,
            required_batch_size_multiple,
        )
        return batch_sampler_index

    # def __next__(self):
    #     if self._now >= len(self):
    #         raise StopIteration
    #     b, e = self._index[self._now]
    #     res = self.indices[b:e+1]
    #     self._now += 1
    #     return res

    def __len__(self):
        return self._index.shape[0]

    def __iter__(self):
        return map(lambda x: self.indices[self._index[x, 0]:self._index[x, 1]+1],
                   range(len(self)))

    def shuffle(self,):
        np.random.shuffle(self._index)
        # self._now = 0
