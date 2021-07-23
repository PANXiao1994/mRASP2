#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Python tools for BUCC bitext mining

import argparse


def BuccOptimize(candidate2score, gold):
    items = sorted(candidate2score.items(), key=lambda x: -x[1])
    ngold = len(gold)
    nextract = ncorrect = 0
    threshold = 0
    best_f1 = 0
    best_p = 0
    best_r = 0
    best_n_extract = 0
    for i in range(len(items)):
        nextract += 1
        if '\t'.join(items[i][0]) in gold:
            ncorrect += 1
        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / ngold
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_p = precision
                best_r = recall
                best_n_extract = nextract
                threshold = (items[i][1] + items[i + 1][1]) / 2
    return best_n_extract, best_f1, best_p, best_r, threshold


parser = argparse.ArgumentParser(description='LASER: tools for BUCC bitext mining')
parser.add_argument('--encoding', default='utf-8',
                    help='character encoding for input/output')
parser.add_argument('--src-lang', required=True,
                    help='the source language id')
parser.add_argument('--trg-lang', required=True,
                    help='the target language id')
parser.add_argument('--candidates', required=True,
                    help='File name of candidate alignments')
parser.add_argument('--gold', default=None,
                    help='File name of gold alignments')
parser.add_argument('--verbose', action='store_true',
                    help='Detailed output')
args = parser.parse_args()

if __name__ == "__main__":
    
    assert args.gold is not None, '--gold must be specified'
    if args.verbose:
        print(' - reading sentences and IDs')
    
    if args.verbose:
        print(' - reading candidates {}'.format(args.candidates))
    
    candidate2score = {}
    with open(args.candidates, encoding=args.encoding, errors='surrogateescape') as f:
        for line in f:
            score, src_id, trg_id = line.strip().split('\t')
            score = float(score)
            candidate2score[(src_id, trg_id)] = score
    
    if args.gold:
        if args.verbose:
            print(' - optimizing threshold on gold alignments {}'.format(args.gold))
        gold = {line.strip() for line in open(args.gold)}
        n_extract, f1, precision, recall, threshold = BuccOptimize(candidate2score, gold)
        
        print(n_extract, threshold, "{:.2f}".format(precision * 100), "{:.2f}".format(recall * 100),
              "{:.2f}".format(f1 * 100), sep="\t")
