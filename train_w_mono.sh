#!/usr/bin/env bash

# repo_dir: root directory of the project
repo_dir="$( cd "$( dirname "$0" )" && pwd )"
echo "==== Working directory: ====" >&2
echo "${repo_dir}" >&2
echo "============================" >&2

main_config=$1
source ${repo_dir}/scripts/load_config.sh ${main_config} ${repo_dir}

model_dir=${repo_dir}/model
data_dir=${repo_dir}/data

mkdir -p ${model_dir} ${data_dir}/mono


# parallel data
data_var=data_1
i=1
data=""
while [[ ! -z ${!data_var} ]]; do
    if [[ $data == "" ]]; then
        data=${!data_var}
    else
        data=$data:${!data_var}
    fi
    i=$((i+1))
    data_var=data_$i
done

# mono data
mono_data_var=data_mono_1
y=1
mono_data=""
while [[ ! -z ${!mono_data_var} ]]; do
    if [[ ${mono_data} == "" ]]; then
        mono_data=${!mono_data_var}
    else
        mono_data=${mono_data}:${!mono_data_var}
    fi
    y=$((y+1))
    mono_data_var=data_mono_$y
done


command="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} fairseq-train ${data} \
  --user-dir ${repo_dir}/mcolt \
  --save-dir ${model_dir} \
  --mono-data ${mono_data} \
  ${options} \
  --ddp-backend no_c10d 1>&2"

echo $command
eval $command

