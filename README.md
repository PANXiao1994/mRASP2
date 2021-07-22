# Contrastive Learning for Many-to-many Multilingual Neural Machine Translation(mCOLT/mRASP2), ACL2021
The code for training mCOLT/mRASP2, a multilingual NMT training framework, implemented based on [fairseq](https://github.com/pytorch/fairseq).

**mRASP2**: [paper](https://arxiv.org/abs/2105.09501)

**mRASP**: [paper](https://www.aclweb.org/anthology/2020.emnlp-main.210.pdf),
[code](https://github.com/linzehui/mRASP)

---
## News
We have released two versions, this version is the new one. In this implementation:
- You should binarize training data seperately.
- AA/RAS can be done on-line.
- We have no ready-to-use checkpoint for this implementation.

**Original implementation**: https://github.com/PANXiao1994/mRASP2 (master branch)


## Introduction

mRASP2/mCOLT, representing multilingual Contrastive Learning for Transformer, is a multilingual neural machine translation model that supports complete many-to-many multilingual machine translation. It employs both parallel corpora and multilingual corpora in a unified training framework. For detailed information please refer to the paper.  

![img.png](docs/img.png)

## Pre-requisite
```bash
pip install -r requirements.txt
```

## Dataset
| Name | Preprocessed | Binarized |
| --- | --- | --- |
| Parallel-pub-100 | [parallel_pub100_prep](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/parallel_pub100_prep/download.sh) |[parallel_pub100_bin](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/parallel_pub100_bin/download.sh) |
| Mono-pub | [mono_pub_prep](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/mono_prep/download.sh) | [mono_pub_bin](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/mono_bin/download.sh) |
| Dev-pub | [test_pub_prep](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/mono_prep/download.sh) | [test_pub_bin](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/test_bin/download.sh) |

## Vocabulary
* We adopt unigram in [sentencepiece](https://github.com/google/sentencepiece) to learn the subword vocabulary jointly on our in-house dataset of 150 languages. The total size of vocabulary is 100k.

[vocab-spm-100k.dict](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/vocab-spm-100k.dict)

[vocab-spm-100k.model](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/vocab-spm-100k.model)

[vocab-spm-100k.vocab](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/vocab-spm-100k.vocab)

## Training
```bash
export NUM_GPU=4 && bash train_multilingual.sh ${model_config}
# or
export NUM_GPU=4 && bash train_multilingual_w_mono.sh ${model_config}
```
* We give example of `${model_config}` in `${PROJECT_REPO}/examples/configs/parallel-12e12d-emb1024-contrast-ras.yml`
* The `step_size` parameter in `${model_config}` should be set to 1 if the dataset is not extremely large.

## Evaluation
```bash
export NUM_GPU=4 && bash eval.sh ${test_config} ${model_config} ${spm_model} ${bleutype}
# `bleutype` must be 'tok' or 'detok'
```
* We give example of `${model_config}` in `${PROJECT_REPO}/examples/configs/eval_benchmarks.yml`

## Generate / Big-infer
```base
export NUM_GPU=4 && export NUM_CPU=10 && bash biginfer.sh ${data_config} ${model_config} ${extra_config}
```
* We give example in `${PROJECT_REPO}/examples/configs/biginfer`

## Synonym dictionaries
We use the bilingual synonym dictionaries provised by [MUSE](https://github.com/facebookresearch/MUSE).

We generate multilingual synonym dictionaries using [this script](https://github.com/linzehui/mRASP/blob/master/preprocess/tools/ras/multi_way_word_graph.py), and apply 
RAS using [this script](https://github.com/linzehui/mRASP/blob/master/preprocess/tools/ras/random_alignment_substitution_w_multi.sh).

| Description | File | Size |
| --- | --- | --- |
| dep=1 | [synonym_dict_raw_dep1](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/synonym_dict_raw_dep1) | 138.0 M |
| dep=2 | [synonym_dict_raw_dep2](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/synonym_dict_raw_dep2) | 1.6 G |
| dep=3 | [synonym_dict_raw_dep3](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/synonym_dict_raw_dep3) | 2.2 G |

* The synonym dictionaries should be preprocessed if you want to use on-line AA(RAS). Check [this script](https://github.com/PANXiao1994/mRASP2/tree/new_impl/preprocess/form_id_dicts.py). We also provide a preprocessed file that is ready to use: [synonym_dict_id_dep1](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/synonym_dict_id_dep1)

## Contact
Please contact me via e-mail `panxiao94@163.com` or via [wechat/zhihu](https://fork-ball-95c.notion.site/mRASP2-4e9b3450d5aa4137ae1a2c46d5f3c1fa)

## Citation
Please cite as:
```
@inproceedings{mrasp2,
  title = {Contrastive Learning for Many-to-many Multilingual Neural Machine Translation},
  author= {Xiao Pan and
           Mingxuan Wang and
           Liwei Wu and
           Lei Li},
  booktitle = {Proceedings of ACL 2021},
  year = {2021},
}
```