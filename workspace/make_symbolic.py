import torch
import hydra
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import seaborn
import pickle

from src import utils, sca_utils


@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def run(cfg):
    device = hydra.utils.instantiate(cfg.device)
    cpu = torch.device('cpu')

    # Prepare datasets
    print('[INFO] Loading Dataset')
    profiling_dataset = hydra.utils.instantiate(cfg.train.dataset)
    test_dataset = hydra.utils.instantiate(cfg.test.dataset)

    # Change torch-style dataset to KAN-style dataset
    train_dataloader = torch.utils.data.DataLoader(
        profiling_dataset, batch_size=cfg.train.batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.test.batch_size, shuffle=False
    )
    KANds = utils.to_KAN_dataset(
        train_dataloader, test_dataloader,
        device=device)

    # Prepare model
    model = hydra.utils.instantiate(cfg.model.model)
    model.load_ckpt(cfg.model_name+'.ckpt', cfg.save_path)
    model = model.to(device)

    # To symbolic
    _ = model(KANds['train_input'].to(device))
    model.auto_symbolic()

    # Fit
    _ = model.train(
        KANds,
        **hydra.utils.instantiate(cfg.model.train_params)
        )
    # Save
    model.to(torch.device('cpu')).save_ckpt('kan_symbolic.ckpt', '.')
    model = model.to(device)

    # Load dataset
    print('[INFO] Load dataset')
    attack_dataset = hydra.utils.instantiate(cfg.test.dataset)
    attack_dl = torch.utils.data.DataLoader(
        attack_dataset, batch_size=cfg.test.batch_size,
        shuffle=False
    )

    # Make predictions
    preds, labels, _ = utils.make_prediction(
        model, attack_dl, device,
        cfg.label_transforms.one_hot)
    if not cfg.label_transforms.one_hot:
        preds_class = np.zeros(preds.shape)
        preds_class[preds > 0.5] = 1.0
        preds_class = preds_class.astype(np.int32)
    else:
        preds_class = np.argmax(preds, axis=1)
    accuracy = np.mean(labels == preds)
    _ = utils.make_confmat(
        preds_class, labels, accuracy, cfg.save_path)

    # Evaluate
    print('[INFO] Calculate guessing entropy')
    correct_key = attack_dataset.key[0][cfg.target_byte]
    key_hyposesis = range(256)
    label_hyposesis = utils.make_label_hyposesis(
        attack_dataset, key_hyposesis, one_hot=cfg.label_transforms.one_hot)
    if not cfg.label_transforms.one_hot:
        confidence = utils.make_confidence(preds)
    else:
        confidence = preds
    ge = sca_utils.calc_guessing_entropy(
        confidence, label_hyposesis, correct_key,
        cfg.n_attack_traces, n_trial=cfg.n_trials)

    # Save results
    print(f'[INFO] GE@{cfg.n_attack_traces}:{ge[-1]}')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ge)
    ax.set_title(f'GE@{cfg.n_attack_traces}:{ge[-1]}')
    fig.savefig(Path(cfg.save_path, f'{cfg.model.name}_sym.png'), dpi=300)

    print('[INFO] Making plot with attack traces')
    _ = model(KANds['test_input'].to(device))
    model.plot(folder=cfg.save_path+'/attack')
    plt.savefig(
        Path(cfg.save_path, f'{cfg.model.name}_attack_sym.png'),
        dpi=300)
    utils.plot_KAN(cfg, model, 'attack')

    formula, variables = model.symbolic_formula()
    with open(Path(cfg.save_path, 'symbolic_fomula.pkl'), mode='wb') as f:
        pickle.dump((formula, variables), f)


if __name__ == '__main__':
    run()
