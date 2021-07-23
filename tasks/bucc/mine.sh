#!/usr/bin/env bash

dict=${BIN_PATH}/${lang}-en/${split}/src/dict.en.txt
ckpt_name="checkpoint_best"

echo -e "$(date "+%Y-%m-%d %H:%M:%S") ---> Bitext mining Started <---\n"

for lang in "${language_list[@]}"
do
    for split in "${split_list[@]}"
    do
        if [[ $split == "test" ]]; then
            continue
        fi
        langpair=${lang}-en
        data_dir=${DEST}/${langpair}/${split}
        output_path=${OUTPUT_PATH}/${langpair}/$split
        gold_path=${GOLD_PATH}/${langpair}/$split.gold

        echo "split into batches..."
        # en
        split -l ${mine_batch_size} ${data_dir}/sent_avg_pool.en ${data_dir}/sent_avg_pool.en.
        rm ${data_dir}/sent_avg_pool.en
        split -l ${mine_batch_size} ${data_dir}/sentences.en ${data_dir}/sentences.en.
        rm ${data_dir}/sentences.en

        # $lang
        split -l ${mine_batch_size} ${data_dir}/sent_avg_pool.${lang} ${data_dir}/sent_avg_pool.${lang}.
        rm ${data_dir}/sent_avg_pool.${lang}
        split -l ${mine_batch_size} ${data_dir}/sentences.${lang} ${data_dir}/sentences.${lang}.
        rm ${data_dir}/sentences.${lang}

        # mine
        mkdir -p ${output_path}

        # mine in batches
        python3 ${repo_dir}/tasks/bucc/mine_bitexts.py --src-lang $lang --tgt-lang en --dict-path ${dict} \
                --dim ${dim} --src-dir ${data_dir} --tgt-dir ${data_dir} \
                --output ${output_path} --neighborhood ${neighbor} \
                --threshold ${threshold}

        # score and write to log
        results=`python3 ${repo_dir}/tasks/bucc/score.py --src-lang $lang --trg-lang en --gold ${gold_path} --candidates ${output_path}/all.cand`
        str="$(date "+%Y-%m-%d %H:%M:%S")\t${ckpt_name}\t${langpair}\t${split}\t${results}"
        echo -e "$str"
    done
done
