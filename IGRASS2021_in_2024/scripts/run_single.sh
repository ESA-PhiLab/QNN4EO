#!/bin/bash
if [ -z "$1" ]
then
    echo "Missing model_name (--model_name)"
    exit 1
fi

if [ -z "$2" ]
then
    echo "Missing class1 (--c1)"
    exit 1
fi

if [ -z "$3" ]
then
    echo "Missing class2 (--c2)"
    exit 1
fi

if [ -z "$4" ]
then
    echo "Missing seed (--seed)"
    exit 1
fi

docker run --rm -e argg="1" \
    -v /home/lorenzo/ESA/code/ESAgit_On_Circuit_based_Hybrid_Quantum_Neural_Networks/QNN4EO_my:/work/project/ \
    -v /home/lorenzo/ESA/dataset:/work/dataset/ \
    -v /home/lorenzo/ESA/results:/work/save_models/ \
    -u 1001:1001 --ipc host --gpus all lorenzopapa5/cuda12.1.0-python3.8-pytorch2.4.0-esa /usr/bin/python3 /work/project/main_binary.py \
    --model_name "${1}" --c1 "${2}" --c2 "${3}" --seed "${4}"