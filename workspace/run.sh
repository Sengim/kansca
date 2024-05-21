#!/bin/bash

##################
# ASCADf dataset #
##################
# KAN
python train_KAN.py --run \
    model=KAN \
    dataset@train=ASCADf_profiling \
    dataset@test=ASCADf_attack

python eval_KAN.py --run \
    model=KAN \
    dataset@test=ASCADf_attack

# MLP
python train_DNN.py --run \
    model=MLP \
    dataset@train=ASCADf_profiling \
    dataset@test=ASCADf_attack

python eval_DNN.py --run \
    model=MLP \
    dataset@test=ASCADf_attack

#####################
# ASCAD_sim dataset #
#####################
# KAN
python train_KAN.py --run \
    model=KANsmall \
    dataset@train=ASCAD_sim \
    train.dataset.sigma=0.0 \
    train.dataset.n_traces=20000 \
    dataset@test=ASCAD_sim \
    test.dataset.sigma=0.0 \
    test.dataset.n_traces=500 \
    label_transforms=hw \
    model.train_params.steps=3000

python eval_KAN.py --run \
    model=KANsmall \
    dataset@train=ASCAD_sim \
    dataset@test=ASCAD_sim \
    test.dataset.sigma=0.0 \
    test.dataset.n_traces=500 \
    label_transforms=hw \
    n_attack_traces=100 \
    n_trials=40
