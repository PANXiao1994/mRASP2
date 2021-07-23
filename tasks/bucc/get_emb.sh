#!/usr/bin/env bash
echo -e "$(date "+%Y-%m-%d %H:%M:%S") ---> Embedding calculation Start <---\n"

for lang in "${language_list[@]}"
do
    for split in "${split_list[@]}"
    do
        python3 ${repo_dir}/tasks/utils/get_emb.py ${BIN_PATH}/${lang}-en/${split}/src \
            --user-dir ${repo_dir}/mlnlc_mt \
            --ckpt ${MODEL_PATH}/checkpoint.pt \
            --gen-subset train --task ${task} -s ${lang} -t en --lang-pairs ${lang}-en \
            --architect ${arch} \
            --langs ${langs} \
            --dest ${DEST}/${lang}-en/${split} \
            --batch-size ${batch_size}
    done
done

for lang in "${language_list[@]}"
do
    for split in "${split_list[@]}"
    do
        python3 ${repo_dir}/tasks/utils/get_emb.py ${BIN_PATH}/${lang}-en/${split}/tgt \
            --user-dir ${repo_dir}/mlnlc_mt \
            --ckpt ${MODEL_PATH}/checkpoint.pt \
            --gen-subset train --task ${task} -s en -t ${lang} --lang-pairs en-${lang} \
            --architect ${arch} \
            --langs ${langs} \
            --dest ${DEST}/${lang}-en/${split} \
            --batch-size ${batch_size}
    done
done

echo -e "$(date "+%Y-%m-%d %H:%M:%S") ---> Embedding calculation Finished <---\n"
