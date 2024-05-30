#!/bin/bash
# Train KAN model with simulated ASCAD traces.

result=/workspace/results/KAN_ASCADsim
cd ..

python train_KAN.py --run \
    model=KAN2h \
    model.model.width.1=3 \
    model.model.width.2=1 \
    dataset@train=ASCAD_sim \
    train.dataset.sigma=0.0 \
    train.dataset.n_traces=50000 \
    dataset@test=ASCAD_sim \
    test.dataset.sigma=0.0 \
    test.dataset.n_traces=500 \
    label_transforms=hw \
    model.train_params.steps=3000 \
    save_path=${result} \
    model.plot_graph=true

python eval_KAN.py --run \
    model=KAN2h \
    model.model.width.1=3 \
    model.model.width.2=1 \
    dataset@train=ASCAD_sim \
    dataset@test=ASCAD_sim \
    test.dataset.sigma=0.0 \
    test.dataset.n_traces=500 \
    label_transforms=hw \
    n_attack_traces=100 \
    n_trials=40 \
    save_path=${result}

cd exp
