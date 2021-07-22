#!/usr/bin/env bash

# repo_dir: root directory of the project
repo_dir="$( cd "$( dirname "$0" )" && pwd )"
echo "==== Working directory: ====" >&2
echo "${repo_dir}" >&2
echo "============================" >&2


test_config=$1
source ${repo_dir}/scripts/load_config.sh ${test_config} ${repo_dir}
model_dir=$2
choice=$3  # all|best|last

model_dir=${repo_dir}/model
data_dir=${repo_dir}/data
res_path=${model_dir}/results

mkdir -p ${model_dir} ${data_dir} ${res_path}

testset_name=data_testset_1_name
testset_path=data_testset_1_path
testset_ref=data_testset_1_ref
testset_direc=data_testset_1_direction
i=1
testsets=""
while [[ ! -z ${!testset_path} && ! -z ${!testset_direc} ]]; do
    dataname=${!testset_name}
    mkdir -p ${data_dir}/${!testset_direc}/${dataname} ${data_dir}/ref/${!testset_direc}/${dataname}
    cp ${!testset_path}/* ${data_dir}/${!testset_direc}/${dataname}/
    cp ${!testset_ref}/* ${data_dir}/ref/${!testset_direc}/${dataname}/
    if [[ $testsets == "" ]]; then
        testsets=${!testset_direc}/${dataname}
    else
        testsets=${testsets}:${!testset_direc}/${dataname}
    fi
    i=$((i+1))
    testset_name=testset_${i}_name
    testset_path=testset_${i}_path
    testset_ref=testset_${i}_ref
    testset_direc=testset_${i}_direction
done

IFS=':' read -r -a testset_list <<< ${testsets}


bleu () {
    src=$1
    tgt=$2
    res_file=$3
    ref_file=$4
    if [[ -f ${res_file} ]]; then
        f_dirname=`dirname ${res_file}`
        python3 ${repo_dir}/scripts/utils.py ${res_file} ${ref_file} || exit 1;
        input_file="${f_dirname}/hypo.out.nobpe"
        output_file="${f_dirname}/hypo.out.nobpe.final"
        # form command
        cmd="cat ${input_file}"
        lang_token="LANG_TOK_"`echo "${tgt} " | tr '[a-z]' '[A-Z]'`
        if [[ $tgt == "fr" ]]; then
            cmd=$cmd" | sed -Ee 's/\"([^\"]*)\"/« \1 »/g'"
        elif [[ $tgt == "zh" ]]; then
            tokenizer="zh"
        elif [[ $tgt == "ja" ]]; then
            tokenizer="ja-mecab"
        fi
        [[ -z $tokenizer ]] && tokenizer="none"
        cmd=$cmd" | sed -e s'|${lang_token} ||g' > ${output_file}"
        eval $cmd || { echo "$cmd FAILED !"; exit 1; }
        cat ${output_file} | sacrebleu -l ${src}-${tgt} -tok $tokenizer --short "${f_dirname}/ref.out" | awk '{print $3}'
    else
        echo "${res_file} not exist!" >&2 && exit 1;
    fi
}

# monitor
# ${ckptname}/${direction}/${testname}/orig.txt
(inotifywait -r -m -e close_write ${res_path} |
while read path action file; do
    if [[ "$file" =~ .*txt$ ]]; then
        tmp_str="${path%/*}"
        testname="${tmp_str##*/}"
        tmp_str="${tmp_str%/*}"
        direction="${tmp_str##*/}"
        tmp_str="${tmp_str%/*}"
        ckptname="${tmp_str##*/}"
        src_lang="${direction%2*}"
        tgt_lang="${direction##*2}"
        res_file=$path$file
        ref_file=${data_dir}/ref/${direction}/${testname}/dev.${tgt_lang}
        bleuscore=`bleu ${src_lang} ${tgt_lang} ${res_file} ${ref_file}`
        bleu_str="$(date "+%Y-%m-%d %H:%M:%S")\t${ckptname}\t${direction}/${testname}\t$bleuscore"
        echo -e ${bleu_str}  # to stdout
        echo -e ${bleu_str} >> ${model_dir}/summary.log
    fi
done) &


if [[ ${choice} == "all" ]]; then
    filelist=`ls -la ${model_dir} | sort -k6,7 -r | awk '{print $NF}' | grep .pt$ | tr '\n' ' '`
elif [[ ${choice} == "best" ]]; then
    filelist="${model_dir}/checkpoint_best.pt"
elif [[ ${choice} == "last" ]]; then
    filelist="${model_dir}/checkpoint_last.pt"
else
    echo "invalid choice!" && exit 2;
fi

N=${NUM_GPU}
#export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $(($N - 1)) )


infer_test () {
    test_path=$1
    ckpts=$2
    gpu=$3
    final_res_file=$4
    src=$5
    tgt=$6
    gpu_cmd="CUDA_VISIBLE_DEVICES=$gpu "
    lang_token="LANG_TOK_"`echo "${tgt} " | tr '[a-z]' '[A-Z]'`
    [[ -z ${max_source_positions} ]] && max_source_positions=1024
    [[ -z ${max_target_positions} ]] && max_target_positions=1024
    command=${gpu_cmd}"fairseq-generate ${test_path} \
    --user-dir ${repo_dir}/mcolt \
    -s ${src} \
    -t ${tgt} \
    --skip-invalid-size-inputs-valid-test \
    --path ${ckpts} \
    --max-tokens 1024 \
    --task translation_w_langtok \
    ${options} \
    --lang-prefix-tok ${lang_token} \
    --max-source-positions ${max_source_positions} \
    --max-target-positions ${max_target_positions} \
    --nbest 1 | grep -E '[S|H|P|T]-[0-9]+' > ${final_res_file}
    "
    echo "$command"
}

export -f infer_test
i=0
(for ckpt in ${filelist}
do
    for testset in "${testset_list[@]}"
    do
        ckptbase=`basename $ckpt`
        ckptname="${ckptbase%.*}"
        direction="${testset%/*}"
        testname="${testset##*/}"
        src_lang="${direction%2*}"
        tgt_lang="${direction##*2}"

        ((i=i%N)); ((i++==0)) && wait
        test_path=${data_dir}/${testset}

        echo "-----> "${ckptname}" | "${direction}/$testname" <-----" >&2
        if [[ ! -d ${res_path}/${ckptname}/${direction}/${testname} ]]; then
            mkdir -p ${res_path}/${ckptname}/${direction}/${testname}
        fi
        final_res_file="${res_path}/${ckptname}/${direction}/${testname}/orig.txt"
        command=`infer_test ${test_path} ${model_dir}/${ckptname}.pt $((i-1)) ${final_res_file} ${src_lang} ${tgt_lang}`
        echo "${command}"
        eval $command &
    done
done)
