#!/bin/bash
# Train KAN model with ASCAD variable-key dataset. (3rd-byte)
# Training traces are acquired with variable (random) keys.
# Attack traces are acquired with fixed keys.
# Input trace is Z-score-normalized based on statistics of profiling traces.
# Use all sample points in the traces (including all leakages)
# Target label is each bit of (unmasked) Sbox output.
# KAN architecture is [1400, 2, 2]
# (1400 points of trace input, 5 hidden nodes, 2 class probability)

result=/workspace/results/KAN_ASCADv_ALL

cd ..

python train_KAN.py --multirun \
    model=KAN1h \
    model.model.width.1=5 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.3.pos}

python eval_KAN.py --multirun \
    model=KAN1h \
    model.model.width.1=5 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    ssave_path=${result}/\${label_transforms.transforms.3.pos}

python plot_KAN.py --multirun \
    model=KAN1h \
    model.model.width.1=5 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.3.pos} \
    n_attack_traces=10000

cd exp
