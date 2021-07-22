#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""
import os
import signal
import sys

from collections import namedtuple
import fileinput

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders, data_utils, LanguagePairDataset

import torch.multiprocessing as mp

from fairseq.logging.meters import StopwatchMeter
from tqdm import tqdm

import time
from fairseq.options import get_parser, add_dataset_args, add_generation_args


def add_split_args(parser, prefix=""):
    group = parser.add_argument_group('SplitArgs')
    # fmt: off
    group.add_argument(f'--{prefix}split-num', type=int, default=1,
                       help="The split number of dataset")
    group.add_argument(f'--{prefix}split-index', type=int, default=0,
                       help="The split index of dataset")
    # fmt: on


def add_biginfer_args(parser):
    group = parser.add_argument_group('BigInfer')
    # fmt: off
    group.add_argument('--buffer-size', default=0, type=int, metavar='N',
                       help='read this many sentences into a buffer before processing them')
    group.add_argument('--input', default='-', type=str, metavar='FILE',
                       help='file to read from; use - for stdin')
    group.add_argument('--input-dataset', action="store_true",
                       help="Decide whether to use the input as a dataset")
    # fmt: on


def get_biginfer_parser():
    default_task = 'translation'
    parser = get_parser('Generation', default_task)
    add_dataset_args(parser, gen=True)
    add_generation_args(parser)
    add_biginfer_args(parser)
    add_split_args(parser, "biginfer-")
    return parser


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')

buffer_end = "BUFFER_END"
process_end = "END"


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    if not args.input_dataset:
        tokens = [
            task.source_dictionary.encode_line(
                encode_fn(src_str), add_if_not_exist=False
            ).long()
            for src_str in lines
        ]
        lengths = torch.LongTensor([t.numel() for t in tokens])
        # sys.stderr.write("ignore invalid data:{}\n".format(args.skip_invalid_size_inputs_valid_test))
        inference_dataset = task.build_dataset_for_inference(tokens, lengths)
    else:
        inference_dataset = lines
    from mcolt.data.subset_dataset import SubsetLanguagePairDataset
    inference_dataset = SubsetLanguagePairDataset(inference_dataset, split_num=args.biginfer_split_num,
                                                  split_index=args.biginfer_split_index)
    itr = task.get_batch_iterator(
        dataset=inference_dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        num_workers=args.num_workers if args.input_dataset else 0,
        num_shards=1,
        shard_id=0,
    ).next_epoch_itr(shuffle=False)
    for batch in tqdm(itr):
        if len(batch) == 0:
            continue
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


class InputDataLoader(object):
    def __init__(self,
                 args,
                 max_positions):
        self.args = args
        self.max_positions = max_positions
        ctx = mp.get_context('spawn')
        self._queue = ctx.Queue(maxsize=args.buffer_size * 3)
        # self._queue = ctx.SimpleQueue()
        self.lock = mp.Lock()
        self.lock.acquire()
        # self.event.clear()
        # self.event,
        self._spawn_context = mp.spawn(
            fn=self._process_fn,
            args=(self.lock,),
            join=False
        )

    def _process_fn(self, i, lock):
        args = self.args
        input = args.input
        max_positions = self.max_positions
        task = tasks.setup_task(args)
        try:
            for inputs in tqdm(buffered_read(input, args.buffer_size)):
                for batch in make_batches(inputs, args, task, max_positions, lambda x: x):
                    self._queue.put(Batch(
                        ids=batch.ids.share_memory_(),
                        src_tokens=batch.src_tokens.share_memory_(),
                        src_lengths=batch.src_lengths.share_memory_(),
                    ))
                self._queue.put(buffer_end)
        except Exception as e:
            sys.stderr.write("{}\n".format(e))
            raise e
        finally:
            self._queue.put(process_end)
        print("Process all input", flush=True)
        print("Now wait for the main process end", flush=True)
        # event.wait()
        while True:
            pass
        # lock.acquire()
        # print("main process ended", flush=True)
        # try:
        #     print("event state is:{}".format(event.is_set()), flush=True)
        #     while not event.is_set():
        #         print("sleep", flush=True)
        #         time.sleep(10)
        # except Exception as e:
        #     print("event error:{}\n".format(e), flush=True)
        #     raise e

    def iter(self):
        while True:
            batch = self._queue.get()
            if batch == process_end:
                break
            yield batch

    def join(self):
        # self.lock.release()
        for pid in self._spawn_context.pids():
            os.kill(pid, signal.SIGTERM)
        # self._spawn_context.join()


def _post_process(align_dict, remove_bpe, nbest, decode_fn, results, src_dict, tgt_dict, translated_number_,
                  fill_gap=True, r2l=False, split_num=1, split_index=0):
    previous = 0
    print("new preprocess", flush=True)
    for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
        id = id * split_num + split_index

        # insert the gap when the input is invalid
        # print("Add The gap", flush=True)
        if fill_gap:
            for _ii in range(previous, id):
                print("H-{}\t{}\t{}".format(_ii, None, ""))
        previous = id + 1
        # print("End The gap", flush=True)

        translated_number_ += 1
        if translated_number_ % 100000 == 0:
            sys.stderr.write("{} finished\n".format(translated_number_))
        if src_dict is not None:
            src_str = src_dict.string(src_tokens, remove_bpe)
            print('S-{}\t{}\t{}'.format(id, 0.0, src_str))
        else:
            src_str = None

        # Process top predictions
        for hypo in hypos[:min(len(hypos), nbest)]:
            # print("hypo:{}".format(hypo), flush=True)
            if r2l:
                hypo['tokens'] = reversed(hypo['tokens'])
            hypo['tokens'] = hypo['tokens'].int()
            bpe_tokens = tgt_dict.string(hypo['tokens'])
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int(),
                src_str=src_str,
                alignment=None,
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=remove_bpe,
            )
            # hypo_str = decode_fn(hypo_str)
            # print("B-{}\t{}\t{}".format(id, bpe_tokens), flush=True)
            print('H-{}\t{}\t{}'.format(id, 0.0, hypo_str), flush=True)
    return translated_number_


class PostProcess(object):

    def __init__(self, remove_bpe, nbest, src_dict, tgt_dict,
                 fill_gap, r2l=False, split_index=0, split_num=1):
        ctx = mp.get_context('spawn')
        self._queue = ctx.SimpleQueue()
        # self._queue = mp.SimpleQueue()
        # self._event = mp.Event()
        # self._event.clear()
        self.remove_bpe = remove_bpe
        self.nbest = nbest
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.fill_gap = fill_gap
        self.r2l = r2l
        self.split_index = split_index
        self.split_num = split_num
        print("Fill gap:{}".format(self.fill_gap))
        self._spawn_context = mp.spawn(
            fn=self._process_fn,
            args=(),
            join=False
        )

    def _process_fn(self, i):
        translated_number = 0

        def identity(x):
            return x

        while True:
            results = self._queue.get()
            if isinstance(results, str) and results == process_end:
                return
            try:
                translated_number = _post_process(None,
                                                  self.remove_bpe,
                                                  self.nbest,
                                                  identity,
                                                  results,
                                                  self.src_dict,
                                                  self.tgt_dict,
                                                  translated_number,
                                                  fill_gap=self.fill_gap,
                                                  r2l=self.r2l,
                                                  split_index=self.split_index,
                                                  split_num=self.split_num,
                                                  )
            except Exception as e:
                print("{}".format(e), flush=True)

    def __call__(self, o):
        self._queue.put(o)

    def join(self):
        self._spawn_context.join()


def make_data_bin_dataset(data_path, src_dict, task):
    dataset = data_utils.load_indexed_dataset(data_path, src_dict)
    return task.build_dataset_for_inference(dataset, dataset.sizes)


def make_input_iter(args, task, max_positions, encode_fn):
    if not args.input_dataset:
        for inputs in buffered_read(args.input, args.buffer_size):
            yield from make_batches(inputs, args, task, max_positions, encode_fn)
            yield buffer_end
    else:
        bin_dataset = make_data_bin_dataset(args.input, task.source_dictionary, task)
        for i, batch in enumerate(make_batches(bin_dataset, args, task, max_positions, encode_fn)):
            yield batch
            if i % args.buffer_size:
                yield buffer_end
        yield buffer_end


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    nbest = args.nbest
    assert not args.sampling or nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.batch_size or args.batch_size <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task
    )

    if getattr(args, "use_checkpoint_task", False):
        args.r2l = getattr(_model_args, "r2l", False)
    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(models=models, args=args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
    )
    sys.stderr.write(f"| Max positions is: {max_positions}")
    sys.stderr.flush()

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    translated_number_ = 0
    # start_id = 0
    post_process_object = PostProcess(
        args.remove_bpe, nbest,
        src_dict,
        tgt_dict,
        fill_gap=not args.input_dataset,
        r2l=getattr(args, "r2l", False),
        split_num=args.biginfer_split_num,
        split_index=args.biginfer_split_index,
    )
    try:
        # input_dataloader = InputDataLoader(args, max_positions)
        input_dataloader = make_input_iter(args, task, max_positions, encode_fn)
        results = []
        for batch in input_dataloader:
            # for inputs in buffered_read(args.input, args.buffer_size):
            #     results = []
            #     for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            if batch == buffer_end:
                post_process_object(results)
                results = []
                continue

            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            translations = task.inference_step(generator, models, sample)

            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                # src_tokens_i = None
                # results.append((id, src_tokens_i, hypos))
                results.append((id, src_tokens_i,
                                [{'tokens': hypo['tokens'].cpu().share_memory_(), 'score': hypo['score']} for hypo in
                                 hypos[:min(len(hypos), nbest)]]))

                # sort output to match input order
            # post_process_object(results)
            # _post_process(align_dict, args.remove_bpe, args.nbest, decode_fn, results, tgt_dict, translated_number_)

            # update running id counter
            # start_id += len(inputs)
    finally:
        post_process_object(process_end)
        post_process_object.join()
    # input_dataloader.join()


def cli_main():
    timer = StopwatchMeter()
    timer.start()
    parser = get_biginfer_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
    timer.stop(1)
    print("Used times:{}".format(timer.avg))


if __name__ == '__main__':
    cli_main()
