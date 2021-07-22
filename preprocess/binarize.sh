#!/usr/bin/env bash
input=$1
output=$2
vocab=$3
prefix=$4
langpair=$5

N=${NUM_CPU}


IFS='-' read -r -a langs <<< ${langpair}
src_lang="${langs[0]}"
if [[ ${prefix} == "dev" ]]; then
  if [[ ! -z ${langs[1]} ]]; then
    tgt_lang="${langs[1]}"
    fairseq-preprocess --source-lang ${src_lang} --target-lang ${tgt_lang} \
                        --srcdict ${vocab} \
                        --tgtdict ${vocab} \
                        --testpref ${input}/${prefix} \
                        --destdir ${output} \
                        --workers $N
  else
    fairseq-preprocess --source-lang ${src_lang} --target-lang ${src_lang} \
                        --srcdict ${vocab} \
                        --tgtdict ${vocab} \
                        --testpref ${input}/${prefix} \
                        --destdir ${output} \
                        --only-source \
                        --workers $N
  fi
else
  if [[ ! -z ${langs[1]} ]]; then
    tgt_lang="${langs[1]}"
    fairseq-preprocess --source-lang ${src_lang} --target-lang ${tgt_lang} \
                        --srcdict ${vocab} \
                        --tgtdict ${vocab} \
                        --trainpref ${input}/${prefix} \
                        --destdir ${output} \
                        --workers $N
  else
    fairseq-preprocess --source-lang ${src_lang} --target-lang ${src_lang} \
                        --srcdict ${vocab} \
                        --tgtdict ${vocab} \
                        --trainpref ${input}/${prefix} \
                        --destdir ${output} \
                        --only-source \
                        --workers $N
  fi
fi
