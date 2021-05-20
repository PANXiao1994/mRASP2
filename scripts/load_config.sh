#!/usr/bin/env bash


function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}
main_config_yml=$1
local_root=$2
if [[ ${main_config_yml} == "hdfs://"* ]]; then
    config_filename=`basename ${main_config_yml}`
    echo 'download config from ${main_config_yml}...'
    local_config="${local_root}/config" && mkdir -p ${local_config}
    hadoop fs -get ${main_config_yml} ${local_config}/
    echo 'finish download config from ${main_config_yml}...'
    main_config_yml=${local_config}/${config_filename}
fi

compgen -A variable > ~/.env-vars
eval $(parse_yaml ${main_config_yml})

# set option flags
options=""
for var in `compgen -A variable | grep -Fxvf  ~/.env-vars`
do
    if [[ ${var} == "model_"* || ${var} == "data_"* || ${var} == "options" ]]; then
        continue
    fi
    if [[ ${!var} == "true" ]]; then
        varname=`echo ${var} | sed 's/\_/\-/g'`
        options=${options}" --${varname}"
    else
        varname=`echo ${var} | sed 's/\_/\-/g'`
        options=${options}" --${varname} ${!var}"
    fi
done
