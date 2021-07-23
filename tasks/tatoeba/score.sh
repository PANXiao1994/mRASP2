#!/usr/bin/env bash

ckpt_name="checkpoint_best"

echo -e "$(date "+%Y-%m-%d %H:%M:%S") ---> Accuracy calculation Started <---\n"

for lang in "${language_list[@]}"
do
    langpair=${lang}-en
    data_dir=${DEST}/${langpair}

    # x->en
    x2en=`python3 ${repo_dir}/tasks/tatoeba/score.py ${data_dir} --src $lang --tgt en`

    # en->x
    en2x=`python3 ${repo_dir}/tasks/tatoeba/score.py ${data_dir} --src en --tgt $lang`

    # write to log
    str="$(date "+%Y-%m-%d %H:%M:%S") \t ${ckpt_name} \t ${lang}->en: ${x2en} \t en->${lang}: ${en2x}"
    echo -e "${str}"
done
