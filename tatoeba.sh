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
TASK_PATH=${repo_dir}/data/tatoeba
MODEL_PATH=${repo_dir}/model

mkdir -p ${MODEL_PATH} ${TASK_PATH}

echo -e "$(date "+%Y-%m-%d %H:%M:%S") ---> Copy model and data Started <---\n"
cp ${meta_model_dir}/checkpoint_best.pt ${MODEL_PATH}/checkpoint.pt
cp ${meta_model_dir}/checkpoint_best-model_part-0.pt ${MODEL_PATH}/checkpoint.pt
cp ${meta_tatoeba_path}/* ${TASK_PATH}/ || { echo "no ${meta_tatoeba_path}" && exit 2; }
echo -e "$(date "+%Y-%m-%d %H:%M:%S") ---> Copy model and data Finished <---\n"


BIN_PATH=${TASK_PATH}/bin
DEST=${TASK_PATH}/emb

IFS=',' read -r -a language_list <<< ${languages}

source ${repo_dir}/tasks/tatoeba/get_emb.sh && source ${repo_dir}/tasks/tatoeba/score.sh

