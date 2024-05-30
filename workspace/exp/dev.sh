#!/bin/bash
# Train KAN model with ASCAD variable-key dataset.

result=/workspace/results/KAN_ASCADv

cd ..

# python train_KAN.py --run \
#     model=KAN1h \
#     model.model.width.1=5 \
#     dataset@train=ASCADv_profiling \
#     dataset@test=ASCADv_attack \
#     trace_transforms=ASCADv_leakage \
#     save_path=${result} \
#     model.plot_graph=true \
#     label_transforms=lsb

python eval_KAN.py --run \
    model=KAN1h \
    model.model.width.1=5 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    trace_transforms=ASCADv_leakage \
    save_path=${result} \
    label_transforms=lsb \

python plot_KAN.py --run \
    model=KAN1h \
    model.model.width.1=5 \
    dataset@train=ASCADv_profiling \
    dataset@test=ASCADv_attack \
    trace_transforms=ASCADv_leakage \
    save_path=${result} \
    label_transforms=lsb \
    n_attack_traces=10000

cd exp
