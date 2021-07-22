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

model_dir=${repo_dir}/model
data_dir=${repo_dir}/data
mkdir ${model_dir} ${data_dir}

# get model
cp ${meta_model_dir}/checkpoint_last*.pt ${model_dir}/
meta_log_file=${meta_model_dir}/info.log


if [[ ! -z ${meta_lang_pairs} ]]; then
    lang_pairs=`cat ${meta_lang_pairs}`
    options=${options}" --lang-pairs ${lang_pairs}"
fi

langs=`cat ${meta_langs}`
options=${options}" --langs ${langs}"

# download RAS synonym dicts
if [[ ! -z ${meta_ras_dict} ]]; then
    resource_dir=${repo_dir}/resource
    mkdir ${resource_dir}
    cp ${meta_ras_dict} ${resource_dir}/
    ras_dict_name=`basename ${meta_ras_dict}`
    options=${options}" --ras-dict ${resource_dir}/${ras_dict_name}"
fi

# download data
IFS=',' read -r -a lang_pair_list <<< ${lang_pairs}
for langpair in "${lang_pair_list[@]}"
do
    echo "${langpair}"
    IFS='-' read -r -a lang_pair <<< ${langpair}
    src="${lang_pair[0]}"
    tgt="${lang_pair[1]}"
    if [[ -d ${meta_test_data}/${langpair} ]]; then
        cp ${meta_test_data}/${langpair}/* ${data_dir}/ &
    fi
    if [[ -d ${meta_test_data}/${tgt}-${src} ]]; then
        cp ${meta_test_data}/${tgt}-${src}/* ${data_dir}/ &
    fi
done

wait


fairseq-train ${data_dir} \
  --user-dir ${repo_dir}/mcolt \
  --save-dir ${model_dir} \
  ${options} \
  --ddp-backend no_c10d 1>&2


