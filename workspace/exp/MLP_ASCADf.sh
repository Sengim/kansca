#!/bin/bash
# Train MLP model with ASCAD fixed-key dataset.

result=/workspace/results/MLP_ASCADf
cd ..

python train_DNN.py --run \
    model=MLP \
    dataset@train=ASCADf_profiling \
    dataset@test=ASCADf_attack \
    save_path=${result} \
    label_transforms=hw

python eval_DNN.py --run \
    model=MLP \
    dataset@train=ASCADf_profiling \
    dataset@test=ASCADf_attack \
    save_path=${result} \
    label_transforms=hw

cd exp
