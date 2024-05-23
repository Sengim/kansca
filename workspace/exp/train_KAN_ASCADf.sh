#!/bin/bash
# Train KAN model with ASCAD fixed-key dataset.

result=/workspace/results/KAN_ASCADf

cd ..

# python train_KAN.py --run \
#     model=KAN1h \
#     dataset@train=ASCADf_profiling \
#     dataset@test=ASCADf_attack \
#     save_path=${result} \
#     model.plot_graph=false \
#     label_transforms=hw3

python eval_KAN.py --run \
    model=KAN1h \
    dataset@train=ASCADf_profiling \
    dataset@test=ASCADf_attack \
    save_path=${result} \
    label_transforms=hw3

cd exp
