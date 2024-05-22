#!/bin/bash
# Stack KAN and MLP model.
# (1) Train KAN model with simulated ASCAD traces
# (2) Stack initialized-MLP and trained-KAN model
# (3) Fix KAN model parameters and train MLP model parameters
#     with ASCADf traces. 

result=/workspace/results/stackedKAN

cd ..

python train_stackedKAN_dev.py --run \
    model=KAN1h \
    dataset@train=ASCAD_sim \
    train.dataset.sigma=0.0 \
    train.dataset.n_traces=20000 \
    dataset@test=ASCAD_sim \
    test.dataset.sigma=0.0 \
    test.dataset.n_traces=500 \
    label_transforms=hw \
    model.train_params.steps=3000 \
    save_path=${result} \
    model.plot_graph=true
