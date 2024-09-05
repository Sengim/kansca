#!/bin/bash
# This script trains MLP with ASCADv1 fixed/variable-key dataset only with LSB labeling.
# Outputs: Trained model, Confusion matrix, Graph of guessing entropy.
# Sensitivity of the trained model is plot by Sensitivity_MLP.ipynb.

cd /workspace

# ASCADf
result=/workspace/results/ascad/MLP_ASCADf
python train_DNN.py --run \
    model=MLP_DDLAexp \
    model.train_params.steps=10000 \
    model.train_params.batch=1024 \
    dataset@train=ASCADf_profiling \
    dataset@test=ASCADf_attack \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0 \
    trace_transforms=void \
    save_path=${result} \
    n_attack_traces=2000

python eval_DNN.py --run \
    model=MLP_DDLAexp \
    model.train_params.steps=10000 \
    model.train_params.batch=1024 \
    dataset@train=ASCADf_profiling \
    dataset@test=ASCADf_attack \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0 \
    trace_transforms=void \
    save_path=${result} \
    n_attack_traces=2000

# ASCADv
result=/workspace/results/ascad/MLP_ASCADv3
python train_DNN.py --run \
    model=MLP_DDLAexp \
    model.train_params.steps=10000 \
    model.train_params.batch=1024 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0 \
    trace_transforms=void \
    save_path=${result} \
    n_attack_traces=2000

python eval_DNN.py --run \
    model=MLP_DDLAexp \
    model.train_params.steps=10000 \
    model.train_params.batch=1024 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0 \
    trace_transforms=void \
    save_path=${result} \
    n_attack_traces=2000
