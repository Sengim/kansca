#!/bin/bash
# Train KAN for all bit

cd /workspace

result=/workspace/results/simulation/KAN_All
python train_KAN.py --multirun \
    model=KAN1h \
    model.model.width.0=2 \
    model.model.width.1=1 \
    model.model.grid=3 \
    model.train_params.steps=3000 \
    dataset@train=masking_sim \
    dataset@test=masking_sim \
    trace_transforms=void \
    label_transforms=bit_sim \
    label_transforms.transforms.0.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.0.pos} \
    n_attack_traces=2000

python eval_KAN.py --multirun \
    model=KAN1h \
    model.model.width.0=2 \
    model.model.width.1=1 \
    model.model.grid=3 \
    model.train_params.steps=3000 \
    dataset@train=masking_sim \
    dataset@test=masking_sim \
    trace_transforms=void \
    label_transforms=bit_sim \
    label_transforms.transforms.0.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.0.pos} \
    n_attack_traces=2000
