#!/bin/bash
# Stack KAN and MLP model.
# (1) Train KAN model with simulated ASCAD traces
# (2) Stack initialized-MLP and trained-KAN model
# (3) Fix KAN model parameters and train MLP model parameters
#     with ASCADf traces. 

result=/workspace/results/stackedKAN

cd ..

# Train KAN model
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

# Train stacked model
# Note: model, train, test is defined to load KAN model
# +stack, +train_stack, +test_stack is defined to 
# construct/train stacked model
python train_stackedKAN.py --run \
    model=KAN2h \
    model.model.width.1=3 \
    model.model.width.2=1 \
    dataset@train=ASCAD_sim \
    dataset@test=ASCAD_sim \
    +model@stack=MLP_stack \
    +dataset@train_stack=ASCADf_profiling \
    +dataset@test_stack=ASCADf_attack \
    label_transforms=hw \
    model.train_params.steps=3000 \
    save_path=${result}

python eval_stackedKAN.py --run \
    model=KAN2h \
    model.model.width.1=3 \
    model.model.width.2=1 \
    dataset@train=ASCAD_sim \
    dataset@test=ASCAD_sim \
    +model@stack=MLP_stack \
    +dataset@train_stack=ASCADf_profiling \
    +dataset@test_stack=ASCADf_attack \
    label_transforms=hw \
    model.train_params.steps=3000 \
    save_path=${result}
