#!/usr/bin/env bash
echo -e "$(date "+%Y-%m-%d %H:%M:%S") ---> Embedding calculation Start <---\n"

for lang in "${language_list[@]}"
do
    python3 ${repo_dir}/tasks/utils/get_emb.py ${BIN_PATH}/${lang}-en \
        --user-dir ${repo_dir}/mlnlc_mt \
        --ckpt ${MODEL_PATH}/checkpoint.pt \
        --gen-subset test --task ${task} -s ${lang} -t en --lang-pairs ${lang}-en \
        --architect ${arch} \
        --langs ${langs} \
        --dest ${DEST}/${lang}-en \
        --batch-size ${batch_size}
done

for lang in "${language_list[@]}"
do
    python3 ${repo_dir}/tasks/utils/get_emb.py ${BIN_PATH}/${lang}-en \
        --user-dir ${repo_dir}/mlnlc_mt \
        --ckpt ${MODEL_PATH}/checkpoint.pt \
        --gen-subset test --task ${task} -s en -t ${lang} --lang-pairs en-${lang} \
        --architect ${arch} \
        --langs ${langs} \
        --dest ${DEST}/${lang}-en \
        --batch-size ${batch_size}
done

echo -e "$(date "+%Y-%m-%d %H:%M:%S") ---> Embedding calculation Finished <---\n"
