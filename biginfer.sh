#!/usr/bin/env bash

# repo_dir: root directory of the project
repo_dir="$( cd "$( dirname "$0" )" && pwd )"
echo "==== Working directory: ====" >&2
echo "${repo_dir}" >&2
echo "============================" >&2

cd ${repo_dir}
python3 setup.py build_ext --inplace

mkdir -p ${repo_dir}/model ${repo_dir}/results

data_config=$1  # bin_dir, src, tgt
model_config=$2  # model_ckpts
extra_config=$3  # beam, lenpen, nbest
source ${repo_dir}/scripts/load_config.sh ${data_config} ${repo_dir}
source ${repo_dir}/scripts/load_config.sh ${model_config} ${repo_dir}
source ${repo_dir}/scripts/load_config.sh ${extra_config} ${repo_dir}

IFS=',' read -r -a model_ckpt_list <<< ${model_ckpts}
for ckpt in "${model_ckpt_list[@]}"
do
  if [[ -z ${checkpoint_paths} ]]; then
    checkpoint_paths=${ckpt}
  else
    checkpoint_paths=${checkpoint_paths},${ckpt}
  fi
done
if [[ ! -z ${meta_langs} ]]; then
    langs=`cat ${meta_langs}`
fi

cp ${bin_dir}/dict.${src}.txt ${bin_dir}/dict.${tgt}.txt
[[ ! -z ${encoder_langtok} ]] && langtok_option=${langtok_option}" --encoder-langtok ${encoder_langtok}"
[[ ! -z ${decoder_langtok} ]] && langtok_option=${langtok_option}" --decoder-langtok"

N=${NUM_GPU}
i=0
for i in `seq 0 $((N-1))`; do
echo ${i}
 CUDA_VISIBLE_DEVICES=${i} python3 -m biginfer ${bin_dir} \
         --user-dir ${repo_dir}/mcolt \
         --path ${checkpoint_paths} \
         --source-lang ${src} --target-lang ${tgt} \
         --lang-pairs ${src}-${tgt} \
         --langs ${langs} \
         ${langtok_option} \
         --task translation_multi_simple_epoch_mcolt \
         --remove-bpe 'sentencepiece' \
         --beam ${beam} --lenpen ${lenpen} --max-tokens ${max_tokens} \
         --buffer-size 1000000 --fp16 \
         --skip-invalid-size-inputs-valid-test \
         --input-dataset \
         --num-workers 1 \
         --nbest ${nbest} \
         --shard-id ${i} \
         --biginfer-split-index ${i} \
         --biginfer-split-num $N \
         --num-shards $N \
         --input ${bin_dir}/train.${src}-${src}.${src} > ${repo_dir}/results/all_${i}.txt &
done

wait

[[ -f ${repo_dir}/results/res.txt ]] && rm ${repo_dir}/results/res.txt
[[ -f ${repo_dir}/results/src.txt ]] && rm ${repo_dir}/results/src.txt
cat ${repo_dir}/results/all_*.txt | grep -E "^H-[0-9]+" | cut -c 3- >> ${repo_dir}/results/res.txt
cat ${repo_dir}/results/all_*.txt | grep -E "^S-[0-9]+" | cut -c 3- >> ${repo_dir}/results/src.txt

source ${repo_dir}/scripts/multiprocess.sh
cat ${repo_dir}/results/src.txt | multiprocess_pipeline "python3 ${repo_dir}/scripts/post_process.py " ${NUM_CPU} | sort -k1 -n --parallel=${NUM_CPU} > ${repo_dir}/results/sort_src.txt
cat ${repo_dir}/results/res.txt | multiprocess_pipeline "python3 ${repo_dir}/scripts/post_process.py " ${NUM_CPU} | sort -k1 -n --parallel=${NUM_CPU} > ${repo_dir}/results/sort_res.txt

cut -f 2 ${repo_dir}/results/sort_src.txt > ${repo_dir}/results/train.${src}
cut -f 2 ${repo_dir}/results/sort_res.txt > ${repo_dir}/results/train.${tgt}
