#!/usr/bin/env bash

function arg_parser(){
    while [[ $# -gt 0 ]]
    do
    key="$1"

    case ${key} in
     -d | --device)
        DEVICE="$2"
        shift
        shift
        ;;

     *)    # unknown option
        POSITIONAL+=("$1") # save it in an array for later
        shift # past argument
        ;;
    #    case $i in
    esac
    done
    set -- "${POSITIONAL[@]}" # restore positional parameters
}

DEVICE="GPU"
arg_parser $*

cd ..


if [[ ${DEVICE} = "GPU" ]]
then
    echo $@
    docker run --network host --runtime=nvidia --user $(id -u):$(id -g) \
    -v $(pwd):/salt-mask-vae -it --rm salt-mask-vae:1.0 "${POSITIONAL[@]}"
else
    echo ">>${POSITIONAL[@]}"
    docker run --network host --user $(id -u):$(id -g) \
    -v $(pwd):/salt-mask-vae -it --rm salt-mask-vae:cpu-1.0 "${POSITIONAL[@]}"
fi