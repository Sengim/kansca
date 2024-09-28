#!/bin/bash
# This script trains KAN[2,1,2] with ASCADv1 fixed/variable-key dataset.
# PoI of input traces is set to $snr4$ and $snr5$.
# Outputs: Trained model, Confusion matrix, Graph of guessing entropy.
# Trained model with LSB label are visualized by KAN_snr_grid3.ipynb.

cd /workspace

# ASCADf
result=/workspace/results/ascad/KAN_ASCADf_snr
python train_KAN.py --multirun \
    model=KAN1h \
    model.model.width.1=1 \
    model.train_params.steps=5000 \
    dataset@train=ASCADf_profiling \
    dataset@test=ASCADf_attack \
    trace_transforms=set_poi \
    trace_transforms.transforms.0.pois="[[156, 157, 1],[517, 518, 1]]" \
    trace_transforms.output_size=2 \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.3.pos} \
    n_attack_traces=2000

python eval_KAN.py --multirun \
    model=KAN1h \
    model.model.width.1=1 \
    model.train_params.steps=5000 \
    dataset@train=ASCADf_profiling \
    dataset@test=ASCADf_attack \
    trace_transforms=set_poi \
    trace_transforms.transforms.0.pois="[[156, 157, 1],[517, 518, 1]]" \
    trace_transforms.output_size=2 \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.3.pos} \
    n_attack_traces=2000

# ASCADv
result=/workspace/results/ascad/KAN_ASCADv_snr
python train_KAN.py --multirun \
    model=KAN1h \
    model.model.width.1=1 \
    model.train_params.steps=5000 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    trace_transforms=set_poi \
    trace_transforms.transforms.0.pois="[[188, 189, 1],[1071, 1072, 1]]" \
    trace_transforms.output_size=2 \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.3.pos} \
    n_attack_traces=2000

python eval_KAN.py --multirun \
    model=KAN1h \
    model.model.width.1=1 \
    model.train_params.steps=5000 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    trace_transforms=set_poi \
    trace_transforms.transforms.0.pois="[[188, 189, 1],[1071, 1072, 1]]" \
    trace_transforms.output_size=2 \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.3.pos} \
    n_attack_traces=2000
