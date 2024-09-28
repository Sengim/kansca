#!/bin/bash
# [Warning] Train/evaluate for each model takes about 6 hour with RTX4060 (Laptop): total runs are 8(fixed) + 8(variable).
# This script trains KAN[#n_input,5,1,2] with ASCADv1 fixed/variable-key dataset.
# Outputs: Trained model, Confusion matrix, Graph of guessing entropy.
# Sensitivity of the trained model is plot by Sensitivity.ipynb.

cd /workspace

# ASCADf
# result=/workspace/results/ascad/KAN_ASCADf
# python train_KAN.py --multirun \
#     model=KAN2h \
#     model.model.width.1=5 \
#     model.model.width.2=1 \
#     model.train_params.steps=5000 \
#     dataset@train=ASCADf_profiling \
#     dataset@test=ASCADf_attack \
#     label_transforms=bit \
#     label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
#     trace_transforms=void \
#     save_path=${result}/\${label_transforms.transforms.3.pos} \
#     n_attack_traces=2000

# python eval_KAN.py --multirun \
#     model=KAN2h \
#     model.model.width.1=5 \
#     model.model.width.2=1 \
#     model.train_params.steps=5000 \
#     dataset@train=ASCADf_profiling \
#     dataset@test=ASCADf_attack \
#     label_transforms=bit \
#     label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
#     trace_transforms=void \
#     save_path=${result}/\${label_transforms.transforms.3.pos} \
#     n_attack_traces=2000

# ASCADv
result=/workspace/results/ascad/KAN_ASCADv
python train_KAN.py --multirun \
    model=KAN2h \
    model.model.width.1=5 \
    model.model.width.2=1 \
    model.train_params.steps=5000 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    trace_transforms=void \
    save_path=${result}/\${label_transforms.transforms.3.pos} \
    n_attack_traces=2000

python eval_KAN.py --multirun \0
    model=KAN2h \
    model.model.width.1=5 \
    model.model.width.2=1 \
    model.train_params.steps=5000 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    trace_transforms=void \
    save_path=${result}/\${label_transforms.transforms.3.pos} \
    n_attack_traces=2000
