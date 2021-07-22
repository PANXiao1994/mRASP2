#!/usr/bin/env bash

# repo_dir: root directory of the project
repo_dir="$( cd "$( dirname "$0" )" && pwd )"
echo "==== Working directory: ====" >&2
echo "${repo_dir}" >&2
echo "============================" >&2

input_data=$1
output_data=$2
spm_model=$3

cd ${repo_dir}

mkdir data output
if [[ -d ${input_data} ]]; then
  for file in `ls ${input_data}/*`
  do
      filename=`basename ${file}`
      python3 ${repo_dir}/spm.py ${spm_model} < data/${filename} > output/${filename}
  done
  hadoop fs -mkdir -p ${output_data}
  hadoop fs -put -f output/* ${output_data}/
else
  hadoop fs -get ${input_data} data/
  output_dir=${output_data%/*}
  filename=`basename ${input_data}`
  python3 ${repo_dir}/spm.py ${spm_model} < data/${filename} > output/${filename}
  hadoop fs -mkdir -p ${output_dir}
  hadoop fs -put -f output/* ${output_dir}/
fi


