#!/usr/bin/env bash

# repo_dir: root directory of the project
repo_dir="$( cd "$( dirname "$0" )" && pwd )"
echo "==== Working directory: ====" >&2
echo "${repo_dir}" >&2
echo "============================" >&2

cd ${repo_dir}
python3 setup.py build_ext --inplace

main_config=$1
source ${repo_dir}/scripts/load_config.sh ${main_config} ${repo_dir}
model_config=$2
source ${repo_dir}/scripts/load_config.sh ${model_config} ${repo_dir}
languages=$3

langs=`cat ${meta_langs}`

N=${NUM_GPU}
TASK_PATH=${repo_dir}/data/bucc2018
MODEL_PATH=${repo_dir}/model

mkdir -p ${MODEL_PATH} ${TASK_PATH}

echo -e "$(date "+%Y-%m-%d %H:%M:%S") ---> Download model and data Started <---\n"
cp ${meta_model_dir}/checkpoint_best.pt ${MODEL_PATH}/checkpoint.pt
cp ${meta_model_dir}/checkpoint_best-model_part-0.pt ${MODEL_PATH}/checkpoint.pt
echo -e "$(date "+%Y-%m-%d %H:%M:%S") ---> Download model and data Finished <---\n"


BIN_PATH=${TASK_PATH}/bin
DEST=${TASK_PATH}/emb
GOLD_PATH=${TASK_PATH}/gold
OUTPUT_PATH=${TASK_PATH}/output

mkdir -p ${GOLD_PATH}
hadoop fs -get ${meta_bucc_path}/gold/* ${GOLD_PATH}/

IFS=',' read -r -a language_list <<< ${languages}
IFS=',' read -r -a split_list <<< ${splits}

for lang in "${language_list[@]}"
do
   for split in "${split_list[@]}"
   do
      mkdir -p ${BIN_PATH}/${lang}-en/${split}/src ${BIN_PATH}/${lang}-en/${split}/tgt
      cp ${meta_bucc_path}/bin/${lang}-en/${split}/${split}.${lang}-en.${lang}.* ${BIN_PATH}/${lang}-en/${split}/src/
      cp ${meta_bucc_path}/bin/${lang}-en/${split}/dict.*.txt ${BIN_PATH}/${lang}-en/${split}/src/
      cp ${meta_bucc_path}/bin/${lang}-en/${split}/${split}.${lang}-en.en.* ${BIN_PATH}/${lang}-en/${split}/tgt/
      cp ${meta_bucc_path}/bin/${lang}-en/${split}/dict.*.txt ${BIN_PATH}/${lang}-en/${split}/tgt/
   done
done

source ${repo_dir}/tasks/bucc/get_emb.sh && source ${repo_dir}/tasks/bucc/mine.sh

