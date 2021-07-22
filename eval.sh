#!/usr/bin/env bash

# repo_dir: root directory of the project
repo_dir="$( cd "$( dirname "$0" )" && pwd )"
echo "==== Working directory: ====" >&2
echo "${repo_dir}" >&2
echo "============================" >&2

cd ${repo_dir}/..
git clone https://github.com/google/sentencepiece.git
cd sentencepiece
mkdir build
cd build
cmake ..
make -j 10
sudo make install
sudo ldconfig -v

cd ${repo_dir}
python3 setup.py build_ext --inplace

test_config=$1
source ${repo_dir}/scripts/load_config.sh ${test_config} ${repo_dir}
model_config=$2
source ${repo_dir}/scripts/load_config.sh ${model_config} ${repo_dir}
if [[ ! -z ${meta_langs} ]]; then
    langs=`cat ${meta_langs}`
fi
spm_model=$3
bleutype=$4
[[ -z ${bleutype} ]] && bleutype="tok"
if [[ $bleutype != "tok" && $bleutype != "detok" ]]; then
    echo "bleutype must be either 'tok' or 'detok', you provided $bleutype ! " && exit 1;
fi

echo "installing dependencies......"
bash ${repo_dir}/scripts/install_dependency.sh
cd ${repo_dir}

model_dir=${repo_dir}/model
data_dir=${repo_dir}/data
res_path=${model_dir}/results

mkdir -p ${model_dir} ${data_dir} ${res_path}

summary_file=${model_dir}/summary.log


data_testset_name=data_testset_1_name
data_testset_path=data_testset_1_path
data_testset_ref=data_testset_1_ref
testset_direc=data_testset_1_direction
i=1
testsets=""
while [[ ! -z ${!data_testset_path} && ! -z ${!testset_direc} ]]; do
    dataname=${!data_testset_name}
    mkdir -p ${data_dir}/${!testset_direc}/${dataname} ${data_dir}/ref/${!testset_direc}/${dataname}
    cp ${!data_testset_path}/* ${data_dir}/${!testset_direc}/${dataname}/
    cp ${!data_testset_ref}/* ${data_dir}/ref/${!testset_direc}/${dataname}/
    if [[ $testsets == "" ]]; then
        testsets=${!testset_direc}/${dataname}
    else
        testsets=${testsets}:${!testset_direc}/${dataname}
    fi
    i=$((i+1))
    data_testset_name=data_testset_${i}_name
    data_testset_path=data_testset_${i}_path
    data_testset_ref=data_testset_${i}_ref
    testset_direc=data_testset_${i}_direction
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
        if [[ ${bleutype} == "tok" ]]; then
            cat ${f_dirname}/ref.out | sh ${repo_dir}/scripts/tok.sh > ${f_dirname}/ref.out.final
            mv ${f_dirname}/ref.out.final ${f_dirname}/ref.out
            if [[ $tgt == "zh" ]]; then
                tokenizer="zh"
            elif [[ $tgt == "ja" ]]; then
                tokenizer="ja-mecab"
            fi
            [[ -z $tokenizer ]] && tokenizer="none"
            if [[ $tgt == "zh" ]]; then
                cat ${input_file} | spm_decode --model=${spm_model} --input_format=piece > ${output_file}
            else
                cat ${input_file} | spm_decode --model=${spm_model} --input_format=piece | sh ${repo_dir}/scripts/tok.sh ${tgt} > ${output_file}
            fi
            cat ${output_file} | sacrebleu -l ${src}-${tgt} -tok $tokenizer --short "${f_dirname}/ref.out" | awk '{print $3}'
        else
            cat ${input_file} | spm_decode --model=${spm_model} --input_format=piece > ${output_file}
            cat ${output_file} | sacrebleu -l ${src}-${tgt} --short "${f_dirname}/ref.out" | awk '{print $3}'
        fi
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
        bleu_str="$(date "+%Y-%m-%d %H:%M:%S")\t${ckptname}\t${direction}/${testname}\t$bleutype\t$bleuscore"
        echo -e ${bleu_str}  # to stdout
        echo -e ${bleu_str} >> ${summary_file}
    fi
done) &


N=${NUM_GPU}

infer_test () {
    test_path=$1
    ckpts=$2
    gpu=$3
    final_res_file=$4
    src=$5
    tgt=$6
    gpu_cmd="CUDA_VISIBLE_DEVICES=$gpu "
    [[ -z ${max_source_positions} ]] && max_source_positions=1024
    [[ -z ${max_target_positions} ]] && max_target_positions=1024
    [[ ! -z ${collate_encoder_langtok} ]] && langtok_option="--collate-encoder-langtok"
    [[ ! -z ${encoder_langtok} ]] && langtok_option=${langtok_option}" --encoder-langtok ${encoder_langtok}"
    [[ ! -z ${decoder_langtok} ]] && langtok_option=${langtok_option}" --decoder-langtok"
    command=${gpu_cmd}"fairseq-generate ${test_path} \
    --user-dir ${repo_dir}/mcolt \
    -s ${src} \
    -t ${tgt} \
    --lang-pairs ${src}-${tgt} \
    --langs ${langs} \
    --skip-invalid-size-inputs-valid-test \
    --path ${ckpts} \
    --max-tokens 1024 \
    --task translation_multi_simple_epoch_mcolt \
    ${langtok_option} \
    --max-source-positions ${max_source_positions} \
    --max-target-positions ${max_target_positions} \
    --nbest 1 | grep -E '[S|H|P|T]-[0-9]+' > ${final_res_file}
    "
    echo "$command"
}

export -f infer_test

# best ckpt
best_ckptname="checkpoint_best_$(date '+%m%d')" #
cp ${meta_model_dir}/checkpoint_best.pt ${model_dir}/${best_ckptname}.pt
cp ${meta_model_dir}/checkpoint_best-model_part-0.pt ${model_dir}/${best_ckptname}.pt


echo "start evaluating ${best_ckptname}"
i=0
(for testset in "${testset_list[@]}"
    do
    direction="${testset%/*}"
    testname="${testset##*/}"
    src_lang="${direction%2*}"
    tgt_lang="${direction##*2}"

    ((i=i%N)); ((i++==0)) && wait
    test_path=${data_dir}/${testset}

    echo "-----> "${best_ckptname}" | "${direction}/$testname" <-----" >&2
    if [[ ! -d ${res_path}/${best_ckptname}/${direction}/${testname} ]]; then
        mkdir -p ${res_path}/${best_ckptname}/${direction}/${testname}
    fi
    final_res_file="${res_path}/${best_ckptname}/${direction}/${testname}/orig.txt"
    command=`infer_test ${test_path} ${model_dir}/${best_ckptname}.pt $((i-1)) ${final_res_file} ${src_lang} ${tgt_lang}`
    echo "${command}"
    eval $command &
done)

