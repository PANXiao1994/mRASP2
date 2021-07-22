# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data import BaseWrapperDataset
import random
from datetime import datetime
from collections import defaultdict
import json
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class Node(object):
    """
    node
    """
    
    def __init__(self, str='', is_root=False):
        self._next_p = {}
        self.fail = None
        self.is_root = is_root
        self.str = str
        self.parent = None
    
    def __iter__(self):
        return iter(self._next_p.keys())
    
    def __getitem__(self, item):
        return self._next_p[item]
    
    def __setitem__(self, key, value):
        _u = self._next_p.setdefault(key, value)
        _u.parent = self
    
    def __repr__(self):
        return "<Node object '%s' at %s>" % \
               (self.str, object.__repr__(self)[1:-1].split('at')[-1])
    
    def __str__(self):
        return self.__repr__()


class AhoCorasick(object):
    """
    Ac object
    """
    
    def __init__(self, *words):
        self.words_set = set(words)
        self.words = list(self.words_set)
        self.words.sort(key=lambda x: len(x))
        self._root = Node(is_root=True)
        self._node_meta = defaultdict(set)
        self._node_all = [(0, self._root)]
        _a = {}
        for word in self.words:
            for w in word:
                _a.setdefault(w.item(), set())
                _a[w.item()].add(tuple([_w.item() for _w in word]))
        
        def node_append(keyword):
            keyword = tuple(keyword.numpy())
            assert len(keyword) > 0
            _ = self._root
            for _i, k in enumerate(keyword):
                k = k.item()
                node = Node(k)
                if k in _:
                    pass
                else:
                    _[k] = node
                    self._node_all.append((_i + 1, _[k]))
                if _i >= 1:
                    for _j in _a[k]:
                        if keyword[:_i + 1][-len(_j):] == _j:
                            self._node_meta[id(_[k])].add((_j, len(_j)))
                _ = _[k]
            else:
                if _ != self._root:
                    self._node_meta[id(_)].add((keyword, len(keyword)))
        
        for word in self.words:
            node_append(word)
        self._node_all.sort(key=lambda x: x[0])
        self._make()
    
    def _make(self):
        """
        build ac tree
        :return:
        """
        for _level, node in self._node_all:
            if node == self._root or _level <= 1:
                node.fail = self._root
            else:
                _node = node.parent.fail
                while True:
                    if node.str in _node:
                        node.fail = _node[node.str]
                        break
                    else:
                        if _node == self._root:
                            node.fail = self._root
                            break
                        else:
                            _node = _node.fail
    
    def search(self, content, with_index=False):
        result = set()
        node = self._root
        index = 0
        for i in content:
            while 1:
                if i not in node:
                    if node == self._root:
                        break
                    else:
                        node = node.fail
                else:
                    for keyword, keyword_len in self._node_meta.get(id(node[i]), set()):
                        if not with_index:
                            result.add(keyword)
                        else:
                            result.add((keyword, (index - keyword_len + 1, index + 1)))
                    node = node[i]
                    break
            index += 1
        return result


class ReplaceTokenSpanDataset(BaseWrapperDataset):
    """Replaces token spans found in the dataset by a specified replacement token

    Args:
        dataset (~torch.utils.data.Dataset): dataset to replace token spans in
        replace_map (Dictionary[str,str]): map of tokens to replace -> replacement tokens
        offsets (List[int]): do not replace tokens before (from left if pos, right if neg) this offset. should be
        as many as the number of objects returned by the underlying dataset __getitem__ method.
    """

    def __init__(self, dataset, src_dict, args):
        super().__init__(dataset)
        self.args = args
        self.source_dictionary = src_dict
        with open(self.args.ras_dict, 'r') as f:
            self.dicts = json.load(f)
        self.replace_map = self.load_replace_map()
        if self.args.encoder_langtok == "src" and self.args.collate_encoder_langtok is None:
            self._lang_index = 1
        else:
            self._lang_index = 0
        self.mask_idx = self.source_dictionary.index("<mask>")

    def __getitem__(self, index):
        idx, tensors = self.dataset[index]
        source = tensors["source"]
        try:
            if torch.equal(tensors["source"], tensors["target"]):
                is_mono = True
            else:
                is_mono = False
        except:
            is_mono = False
        ret_source, unchanged = self.replace_one_sent(source)
        if is_mono and unchanged is True:  # for monolingual data, we must modify the source side
            ret_source = self.mask_one_sentence(source)
        tensors["source"] = ret_source
        return idx, tensors

    def load_replace_map(self):
        replace_map = {}
        for lang in self.dicts.keys():
            if self.args.langs and lang not in self.args.langs:
                continue  # skip unrelated languages
            logger.info(" Start building AC Automachine for {}".format(lang))
            patterns = []
            for word in self.dicts[lang].keys():
                if len(word) == 0:
                    continue
                pattern = []
                flag = True
                for token in word.strip().split('-'):
                    try:
                        pattern.append(int(token))
                    except:
                        flag = False
                        break
                if flag:
                    patterns.append(torch.Tensor(pattern).int())
            tree = AhoCorasick(*patterns)  # construct AC machine
            replace_map[lang] = tree
            logger.info(" Finished building AC Automachine for {}".format(lang))
        return replace_map

    def get_lang(self, tokens):
        _index = tokens[self._lang_index].numpy()
        lang = self.source_dictionary[_index]
        return lang.strip('_')

    # replace tokens in one sentence
    def replace_one_sent(self, tokens):
        """

        :param lang: the language of the sentence
        :param tokens: the token id(s) of the sentence
        :return:
        """

        lang = self.get_lang(tokens)
        if lang not in self.replace_map:
            return tokens, False
        dictionary_tree = self.replace_map[lang]
        match_patterns = dictionary_tree.search(tokens.numpy(), True)
        _match_patterns = sorted(match_patterns, key=lambda kv: kv[1][0])
        _first = 0
        _last = 0
        token_intervals = []
        for key, span in _match_patterns:
            if span[0] < _last:
                continue  # skip if overlap
            _first = span[0]
            _key = "-".join([str(id) for id in key])
            _dict = self.dicts[lang][_key]
            if len(_dict) == 0:
                continue
            if _first > _last:
                token_intervals.append(tokens[_last: _first])
            _max_depth = max([int(d) for d in _dict.keys()])
            _candidates = []
            for dep in range(1, _max_depth + 1):
                # words with smaller depth has larger proba to be selected
                if str(dep) in _dict:
                    _candidates.extend(_dict[str(dep)] * pow(2, _max_depth - dep))
            _rep_token = None
            random.seed(datetime.now())
            if random.random() < self.args.ras_replace_prob:
                _iter = 0
                while (_rep_token is None or _key == _rep_token) and _iter < 1:
                    _choice = random.choice(_candidates)
                    if self.args.ras_target_lang_dict is not None and _choice[1] not in self.args.ras_target_lang_dict:
                        continue
                    if _choice[1] not in self.args.langs:
                        continue
                    _rep_token = _choice[0]
                    _iter += 1
            if _rep_token:
                rep_tokens = torch.Tensor([int(_id) for _id in _rep_token.split('-')]).long()
                token_intervals.append(rep_tokens)  # replaced
            else:
                token_intervals.append(tokens[span[0]: span[1]])  # original
            _last = span[1]  # set cursor to be the right point
        if _last < len(tokens):
            token_intervals.append(tokens[_last:])
        ret_tokens = torch.cat(token_intervals, dim=0)
        if ret_tokens.shape[0] > self.args.max_source_positions:
            return tokens, True
        return ret_tokens, ret_tokens.shape == tokens.shape and torch.all(ret_tokens.eq(tokens)).numpy()

    def mask_one_sentence(self, tokens):
        # decide elements to mask
        assert (
                self.mask_idx not in tokens
        ), "Dataset contains mask_idx (={}), this is not expected!".format(
            self.mask_idx,
        )
        sz = len(tokens) - 2  # language token and eos
        random.seed(datetime.now())
        mask = np.full(sz, False)
        num_mask = min(int(np.floor(
            # add a random number for probabilistic rounding
            self.args.mask_prob * sz
            + np.random.rand()
        )), sz)
        mask[np.random.choice(sz, num_mask, replace=False)] = True
        mask = np.insert(mask, 0, False)
        mask = np.append(mask, False)
        new_item = np.copy(tokens)
        new_item[mask] = self.mask_idx
        return torch.from_numpy(new_item)
