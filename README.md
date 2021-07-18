# Contrastive Learning for Many-to-many Multilingual Neural Machine Translation(mCOLT/mRASP2), ACL2021
The code for training mCOLT/mRASP2, a multilingual NMT training framework, implemented based on [fairseq](https://github.com/pytorch/fairseq).

Arxiv: [paper](https://arxiv.org/abs/2105.09501)

**mRASP** 
* paper: https://www.aclweb.org/anthology/2020.emnlp-main.210.pdf
* code: https://github.com/linzehui/mRASP

## Introduction

mRASP2/mCOLT, representing multilingual Contrastive Learning for Transformer, is a multilingual neural machine translation model that supports complete many-to-many multilingual machine translation. It employs both parallel corpora and multilingual corpora in a unified training framework. For detailed information please refer to the paper.  

![img.png](docs/img.png)

## Pre-requisite
```bash
pip install -r requirements.txt
```

## Training Data and Checkpoints
We release our preprocessed training data and checkpoints in the following.
### Dataset

We merge 32 English-centric language pairs, resulting in 64 directed translation pairs in total. The original 32 language pairs corpus contains about 197M pairs of sentences. We get about 262M pairs of sentences after applying RAS, since we keep both the original sentences and the substituted sentences. We release both the original dataset and dataset after applying RAS.

| Dataset | #Pair |
| --- | --- |
| [32-lang-pairs-TRAIN](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bin_parallel/download.sh) | 197603294 |
| [32-lang-pairs-RAS-TRAIN](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bin_parallel_ras/download.sh) | 262662792 |
| [mono-split-a](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bin_mono_split_a/download.sh) | - |
| [mono-split-b](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bin_mono_split_b/download.sh) | - |
| [mono-split-c](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bin_mono_split_c/download.sh) | - |
| [mono-split-d](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bin_mono_split_d/download.sh) | - |
| [mono-split-e](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bin_mono_split_e/download.sh) | - |
| [mono-split-de-fr-en](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bin_mono_de_fr_en/download.sh) | - |
| [mono-split-nl-pl-pt](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bin_mono_nl_pl_pt/download.sh) | - |
| [32-lang-pairs-DEV-en-centric](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bin_dev_en_centric/download.sh) | - |
| [32-lang-pairs-DEV-many-to-many](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bin_dev_m2m/download.sh) | - |
| [Vocab](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bpe_vocab) | - |
| [BPE Code](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/bpe_vocab) | - |


### Checkpoints
Note that the provided checkpoint is sightly different from that in the paper.

[mRASP2-12e12d](http://sf3-ttcdn-tos.pstatp.com/obj/nlp-opensource/acl2021/mrasp2/12e12d_last.pt)


## Training
```bash
bash train_w_mono.sh ${model_config}
```
* We give example of `${model_config}` in `${PROJECT_REPO}/examples/configs/parallel_mono_12e12d_contrastive.yml`

## Inference
```bash
fairseq-generate ${test_path} \
    --user-dir ${repo_dir}/mcolt \
    -s ${src} \
    -t ${tgt} \
    --skip-invalid-size-inputs-valid-test \
    --path ${ckpts} \
    --max-tokens ${batch_size} \
    --task translation_w_langtok \
    ${options} \
    --lang-prefix-tok "LANG_TOK_"`echo "${tgt} " | tr '[a-z]' '[A-Z]'` \
    --max-source-positions ${max_source_positions} \
    --max-target-positions ${max_target_positions} \
    --nbest 1 | grep -E '[S|H|P|T]-[0-9]+' > ${final_res_file}
python3 ${repo_dir}/scripts/utils.py ${res_file} ${ref_file} || exit 1;
```

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