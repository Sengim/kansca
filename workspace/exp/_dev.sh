#!/bin/bash
# Train KAN model with ASCAD variable-key dataset.

result=/workspace/results/KAN_ASCADv

cd ..

python train_KAN.py --run \
    model=KAN1h \
    model.model.width.1=2 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    trace_transforms=ASCADv_leakage \
    trace_transforms.transforms.0.pois="[[188, 189, 1]]" \
    trace_transforms.output_size=1 \
    save_path=${result} \
    model.plot_graph=true \
    label_transforms=lsb

python eval_KAN.py --run \
    model=KAN1h \
    model.model.width.1=2 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    trace_transforms=ASCADv_leakage \
    trace_transforms.transforms.0.pois="[[1071, 1072, 1]]" \
    trace_transforms.output_size=1 \
    save_path=${result} \
    label_transforms=lsb

python plot_KAN.py --run \
    model=KAN1h \
    model.model.width.1=2 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    trace_transforms=ASCADv_leakage \
    trace_transforms.transforms.0.pois="[[1071, 1072, 1]]" \
    trace_transforms.output_size=1 \
    save_path=${result} \
    label_transforms=lsb \
    n_attack_traces=10000

cd exp
