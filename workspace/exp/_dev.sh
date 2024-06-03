#!/bin/bash
# Train KAN model with ASCAD variable-key dataset.
# ,1,2,3,4,5,6,7

result=/workspace/results/KAN_ASCADv_maskedSboxOut

cd ..

python train_KAN.py --multirun \
    model=KAN1h \
    model.model.width.1=3 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    trace_transforms=ASCADv_leakage \
    trace_transforms.transforms.0.pois="[[1071, 1072, 1]]" \
    trace_transforms.output_size=1 \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.3.pos}

python eval_KAN.py --run \
    model=KAN1h \
    model.model.width.1=3 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    trace_transforms=ASCADv_leakage \
    trace_transforms.transforms.0.pois="[[1071, 1072, 1]]" \
    trace_transforms.output_size=1 \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.3.pos}

python plot_KAN.py --run \
    model=KAN1h \
    model.model.width.1=3 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    trace_transforms=ASCADv_leakage \
    trace_transforms.transforms.0.pois="[[1071, 1072, 1]]" \
    trace_transforms.output_size=1 \
    label_transforms=bit \
    label_transforms.transforms.3.pos=0,1,2,3,4,5,6,7 \
    save_path=${result}/\${label_transforms.transforms.3.pos} \
    n_attack_traces=10000

cd exp
