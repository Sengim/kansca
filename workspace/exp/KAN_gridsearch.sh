#!/bin/bash

result=/workspace/results/KAN_ASCADsim
cd ..

python train_KAN.py --multirun \
    model=KAN2h \
    model.model.width.1=1,2,3,4,5 \
    model.model.width.2=1,2,3,4,5 \
    dataset@train=ASCAD_sim \
    train.dataset.sigma=0.0 \
    train.dataset.n_traces=50000 \
    dataset@test=ASCAD_sim \
    test.dataset.sigma=0.0 \
    test.dataset.n_traces=500 \
    label_transforms=hw \
    model.train_params.steps=3000 \
    save_path=${result}/\${model.model.width.1}/\${model.model.width.2} \
    model.plot_graph=true

python eval_KAN.py --multirun \
    model=KAN2h \
    model.model.width.1=1,2,3,4,5 \
    model.model.width.2=1,2,3,4,5 \
    dataset@train=ASCAD_sim \
    dataset@test=ASCAD_sim \
    test.dataset.sigma=0.0 \
    test.dataset.n_traces=500 \
    label_transforms=hw \
    n_attack_traces=100 \
    n_trials=40 \
    save_path=${result}/\${model.model.width.1}/\${model.model.width.2}

cd exp
