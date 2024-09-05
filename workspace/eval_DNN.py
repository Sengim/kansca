import torch
import hydra
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src import utils, sca_utils


@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def run(cfg):
    device = hydra.utils.instantiate(cfg.device)

    # Load dataset
    print('[INFO] Load dataset')
    attack_dataset = hydra.utils.instantiate(cfg.test.dataset)
    attack_dl = torch.utils.data.DataLoader(
        attack_dataset, batch_size=cfg.test.batch_size,
        shuffle=False
    )

    # Prepare model
    model = hydra.utils.instantiate(cfg.model.model)
    model.load_state_dict(torch.load(
        Path(cfg.save_path, cfg.model_name+'.pt')))
    model = model.to(device)

    # Make predictions
    preds, labels, _ = utils.make_prediction(
        model, attack_dl, device)
    preds_class = np.argmax(preds, axis=1)
    accuracy = np.mean(labels == preds_class)
    _ = utils.make_confmat(
        preds_class, labels, accuracy, cfg.save_path)

    # Evaluate
    print('[INFO] Calculate guessing entropy')
    correct_key = attack_dataset.key[0][cfg.target_byte]
    key_hyposesis = range(256)
    label_hyposesis = utils.make_label_hyposesis(
        attack_dataset, key_hyposesis)
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
    fig.savefig(Path(cfg.save_path, f'{cfg.model.name}.png'), dpi=300)


if __name__ == '__main__':
    run()
