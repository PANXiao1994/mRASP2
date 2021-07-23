# !/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import glob
from subprocess import check_call
import faiss
import numpy as np

GB = 1024 * 1024 * 1024


def call(cmd):
    print(cmd)
    check_call(cmd, shell=True)


def get_batches(directory, lang, prefix="sent_avg_pool"):
    print(f"Finding in {directory}/{prefix}.{lang}*")
    files = glob.glob(f"{directory}/{prefix}.{lang}*")
    emb_files = []
    txt_files = []
    for emb_fi in files:
        emb_files.append(emb_fi)
        txt_fi = emb_fi.replace(prefix, "sentences")
        txt_files.append(txt_fi)
    return emb_files, txt_files


def load_batch(emb_file):
    with open(emb_file) as embFile:
        embeddings = [line.split(',') for line in embFile]
    embeddings = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    return embeddings


def knnGPU_sharded(x_batches_f, y_batches_f, dim, k, direction="x2y"):
    sims = []
    inds = []
    xfrom = 0
    xto = 0
    for x_batch_f in x_batches_f:
        yfrom = 0
        yto = 0
        x_batch = load_batch(x_batch_f)
        xto = xfrom + x_batch.shape[0]
        bsims, binds = [], []
        for y_batch_f in y_batches_f:
            y_batch = load_batch(y_batch_f)
            neighbor_size = min(k, y_batch.shape[0])
            yto = yfrom + y_batch.shape[0]
            print("{}-{}  ->  {}-{}".format(xfrom, xto, yfrom, yto))
            idx = faiss.IndexFlatIP(dim)
            idx = faiss.index_cpu_to_all_gpus(idx)
            idx.add(y_batch)
            bsim, bind = idx.search(x_batch, neighbor_size)
            
            bsims.append(bsim)
            binds.append(bind + yfrom)
            yfrom += y_batch.shape[0]
            del idx
            del y_batch
        bsims = np.concatenate(bsims, axis=1)
        binds = np.concatenate(binds, axis=1)
        aux = np.argsort(-bsims, axis=1)
        sim_batch = np.zeros((x_batch.shape[0], k), dtype=np.float32)
        ind_batch = np.zeros((x_batch.shape[0], k), dtype=np.int64)
        for i in range(x_batch.shape[0]):
            for j in range(k):
                sim_batch[i, j] = bsims[i, aux[i, j]]
                ind_batch[i, j] = binds[i, aux[i, j]]
        sims.append(sim_batch)
        inds.append(ind_batch)
        xfrom += x_batch.shape[0]
        del x_batch
    sim = np.concatenate(sims, axis=0)
    ind = np.concatenate(inds, axis=0)
    return sim, ind


def score(sim, fwd_mean, bwd_mean, margin):
    return margin(sim, (fwd_mean + bwd_mean) / 2)


def score_candidates(
        sim_mat, candidate_inds, fwd_mean, bwd_mean, margin, verbose=False
):
    print(" - scoring {:d} candidates".format(sim_mat.shape[0]))
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = int(candidate_inds[i, j])
            scores[i, j] = score(sim_mat[i, j], fwd_mean[i], bwd_mean[k], margin)
    return scores


def load_text(files):
    all_sentences = []
    for fi in files:
        with open(fi) as sentence_fi:
            for line in sentence_fi:
                all_sentences.append(line.strip())
    print(f"Read {len(all_sentences)} sentences")
    return all_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine bitext")
    parser.add_argument("--src-lang", help="Source language")
    parser.add_argument("--tgt-lang", help="Target language")
    parser.add_argument(
        "--dict-path", help="Path to dictionary file", default="dict.txt"
    )
    parser.add_argument("--dim", type=int, default=1024, help="Embedding dimension")
    parser.add_argument("--src-dir", help="Source directory")
    parser.add_argument("--tgt-dir", help="Target directory")
    parser.add_argument("--output", help="Output path")
    parser.add_argument(
        "--neighborhood", type=int, default=4, help="Embedding dimension"
    )
    parser.add_argument(
        "--threshold", type=float, default=1.06, help="Threshold on mined bitext"
    )
    args = parser.parse_args()
    
    x_batches_f, x_sents_f = get_batches(args.src_dir, args.src_lang)
    y_batches_f, y_sents_f = get_batches(args.tgt_dir, args.tgt_lang)
    margin = lambda a, b: a / b
    y2x_sim, y2x_ind = knnGPU_sharded(
        y_batches_f, x_batches_f, args.dim, args.neighborhood, direction="y2x"
    )
    x2y_sim, x2y_ind = knnGPU_sharded(
        x_batches_f, y_batches_f, args.dim, args.neighborhood, direction="x2y"
    )
    
    x2y_mean = x2y_sim.mean(axis=1)
    y2x_mean = y2x_sim.mean(axis=1)
    fwd_scores = score_candidates(x2y_sim, x2y_ind, x2y_mean, y2x_mean, margin)
    bwd_scores = score_candidates(y2x_sim, y2x_ind, y2x_mean, x2y_mean, margin)
    fwd_best = x2y_ind[np.arange(x2y_sim.shape[0]), fwd_scores.argmax(axis=1)]
    bwd_best = y2x_ind[np.arange(y2x_sim.shape[0]), bwd_scores.argmax(axis=1)]
    indices = np.stack(
        (
            np.concatenate((np.arange(x2y_ind.shape[0]), bwd_best)),
            np.concatenate((fwd_best, np.arange(y2x_ind.shape[0]))),
        ),
        axis=1,
    )
    scores = np.concatenate((fwd_scores.max(axis=1), bwd_scores.max(axis=1)))
    
    x_sentences = load_text(x_sents_f)
    y_sentences = load_text(y_sents_f)
    
    threshold = args.threshold
    seen_src, seen_trg = set(), set()
    directory = args.output
    call(f"mkdir -p {directory}")
    src_out = open(
        f"{directory}/all.{args.src_lang}",
        mode="w",
        encoding="utf-8",
        errors="surrogateescape",
    )
    tgt_out = open(
        f"{directory}/all.{args.tgt_lang}",
        mode="w",
        encoding="utf-8",
        errors="surrogateescape",
    )
    scores_out = open(
        f"{directory}/all.scores", mode="w", encoding="utf-8", errors="surrogateescape"
    )
    cand_out = open(
        f"{directory}/all.cand", mode="w", encoding="utf-8", errors="surrogateescape"
    )
    count = 0
    for i in np.argsort(-scores):
        src_ind, trg_ind = indices[i]
        if src_ind not in seen_src and trg_ind not in seen_trg:
            seen_src.add(src_ind)
            seen_trg.add(trg_ind)
            if scores[i] > threshold:
                if x_sentences[src_ind]:
                    print(scores[i], file=scores_out)
                    print(x_sentences[src_ind], file=src_out)
                    print(y_sentences[trg_ind], file=tgt_out)
                    _src_id = int(x_sentences[src_ind].split('\t')[0]) + 1
                    _tgt_id = int(y_sentences[trg_ind].split('\t')[0]) + 1
                    _line = "{}\t{}-{}\t{}-{}".format(scores[i], args.src_lang, str(_src_id).zfill(9), args.tgt_lang, str(_tgt_id).zfill(9))
                    print(_line, file=cand_out)
                    count += 1
                else:
                    print(f"Ignoring sentence: {x_sentences[src_ind]}")
    src_out.close()
    tgt_out.close()
    scores_out.close()
    cand_out.close()
    
    print(f"Found {count} pairs for threshold={threshold}")
