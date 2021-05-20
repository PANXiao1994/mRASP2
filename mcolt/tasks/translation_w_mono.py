# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    LanguagePairDataset)

from ..data import SubsampleLanguagePairDataset

import logging
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset

logger = logging.getLogger(__name__)


def concat_language_pair_dataset(*language_pair_datasets, up_sample_ratio=None,
                                 all_dataset_upsample_ratio=None):
    logger.info("To cancat the language pairs")
    dataset_number = len(language_pair_datasets)
    if dataset_number == 1:
        return language_pair_datasets[0]
    elif dataset_number < 1:
        raise ValueError("concat_language_pair_dataset needs at least on dataset")
    # for dataset in language_pair_datasets:
    #     assert isinstance(dataset, LanguagePairDataset), "concat_language_pair_dataset can only concat language pair" \
    #                                                      "dataset"
    
    src_list = [language_pair_datasets[0].src]
    tgt_list = [language_pair_datasets[0].tgt]
    src_dict = language_pair_datasets[0].src_dict
    tgt_dict = language_pair_datasets[0].tgt_dict
    left_pad_source = language_pair_datasets[0].left_pad_source
    left_pad_target = language_pair_datasets[0].left_pad_target
    
    logger.info("To construct the source dataset list and the target dataset list")
    for dataset in language_pair_datasets[1:]:
        assert dataset.src_dict == src_dict
        assert dataset.tgt_dict == tgt_dict
        assert dataset.left_pad_source == left_pad_source
        assert dataset.left_pad_target == left_pad_target
        src_list.append(dataset.src)
        tgt_list.append(dataset.tgt)
    logger.info("Have constructed the source dataset list and the target dataset list")
    
    if all_dataset_upsample_ratio is None:
        sample_ratio = [1] * len(src_list)
        sample_ratio[0] = up_sample_ratio
    else:
        sample_ratio = [int(t) for t in all_dataset_upsample_ratio.strip().split(",")]
        assert len(sample_ratio) == len(src_list)
    src_dataset = ConcatDataset(src_list, sample_ratios=sample_ratio)
    tgt_dataset = ConcatDataset(tgt_list, sample_ratios=sample_ratio)
    res = LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
    )
    logger.info("Have created the concat language pair dataset")
    return res


@register_task('translation_w_mono')
class TranslationWithMonoTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """
    
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--mono-data', default=None, help='monolingual data, split by :')
        parser.add_argument('--mono-one-split-each-epoch', action='store_true', default=False, help='use on split of monolingual data at each epoch')
        parser.add_argument('--parallel-ratio', default=1.0, type=float, help='subsample ratio of parallel data')
        parser.add_argument('--mono-ratio', default=1.0, type=float, help='subsample ratio of mono data')
    
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.update_number = 0
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'
        
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')
        
        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        
        return cls(args, src_dict, tgt_dict)
    
    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        logger.info("To load the dataset {}".format(split))
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]
        
        mono_paths = utils.split_paths(self.args.mono_data)
        
        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang
        
        parallel_data = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
        )
        if split == "train":
            parallel_data = SubsampleLanguagePairDataset(parallel_data, size_ratio=self.args.parallel_ratio,
                                                         seed=self.args.seed,
                                                         epoch=epoch)
            if self.args.mono_one_split_each_epoch:
                mono_path = mono_paths[(epoch - 1) % len(mono_paths)]  # each at one epoch
                mono_data = load_langpair_dataset(
                    mono_path, split, src, self.src_dict, tgt, self.tgt_dict,
                    combine=combine, dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=self.args.max_source_positions,
                    shuffle=(split != "test"),
                    max_target_positions=self.args.max_target_positions,
                )
                mono_data = SubsampleLanguagePairDataset(mono_data, size_ratio=self.args.mono_ratio,
                                                         seed=self.args.seed,
                                                         epoch=epoch)
                all_dataset = [parallel_data, mono_data]
            else:
                mono_datas = []
                for mono_path in mono_paths:
                    mono_data = load_langpair_dataset(
                        mono_path, split, src, self.src_dict, tgt, self.tgt_dict,
                        combine=combine, dataset_impl=self.args.dataset_impl,
                        upsample_primary=self.args.upsample_primary,
                        left_pad_source=self.args.left_pad_source,
                        left_pad_target=self.args.left_pad_target,
                        max_source_positions=self.args.max_source_positions,
                        shuffle=(split != "test"),
                        max_target_positions=self.args.max_target_positions,
                    )
                    mono_data = SubsampleLanguagePairDataset(mono_data, size_ratio=self.args.mono_ratio,
                                                             seed=self.args.seed,
                                                             epoch=epoch)
                    mono_datas.append(mono_data)
                all_dataset = [parallel_data] + mono_datas
            self.datasets[split] = ConcatDataset(all_dataset)
        else:
            self.datasets[split] = parallel_data
