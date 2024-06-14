#!/bin/bash
# Train KAN model with ASCAD variable-key dataset. (3rd-byte)
# Training traces are acquired with variable (random) keys.
# Attack traces are acquired with fixed keys.
# Input trace is Z-score-normalized based on statistics of profiling traces.
# Use a known leakage peaks (please refer the ASCADv_SNR.ipynb)
# - Masked Sbox output (Sbox(p[2]^k[2])^r_out), PoI: 1071 (HW), 1063 & 1071 (bit)
# Target label is each bit of (unmasked) Sbox output.
# KAN architecture is [1, 2, 2]
# (1 leakages input, 2 hidden nodes, 2 class probability)

result=/workspace/results/ASCADfPoIBoth

cd ..

python train_KAN.py --multirun \
    model=KAN1h \
    model.model.width.1=2 \
    dataset@train=ASCADf_profiling \
    dataset@test=ASCADf_attack \
    trace_transforms=set_poi \
    trace_transforms.transforms.0.pois="[[156, 157, 1],[517, 518, 1]]" \
    trace_transforms.output_size=2 \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.3.pos}

python eval_KAN.py --multirun \
    model=KAN1h \
    model.model.width.1=2 \
    dataset@train=ASCADf_profiling \
    dataset@test=ASCADf_attack \
    trace_transforms=set_poi \
    trace_transforms.transforms.0.pois="[[156, 157, 1],[517, 518, 1]]" \
    trace_transforms.output_size=2 \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.3.pos}

python plot_KAN.py --multirun \
    model=KAN1h \
    model.model.width.1=2 \
    dataset@train=ASCADf_profiling \
    dataset@test=ASCADf_attack \
    trace_transforms=set_poi \
    trace_transforms.transforms.0.pois="[[156, 157, 1],[517, 518, 1]]" \
    trace_transforms.output_size=2 \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.3.pos} \
    n_attack_traces=10000

cd exp
