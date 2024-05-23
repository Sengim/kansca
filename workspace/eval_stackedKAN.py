import torch
import hydra
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import seaborn

from src import utils, sca_utils


@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def run(cfg):
    device = hydra.utils.instantiate(cfg.device)
    cpu = torch.device('cpu')

    # Load dataset
    print('[INFO] Load dataset')
    attack_dataset = hydra.utils.instantiate(cfg.test_stack.dataset)
    attack_dl = torch.utils.data.DataLoader(
        attack_dataset, batch_size=cfg.test.batch_size,
        shuffle=False
    )

    # Prepare model
    kan = hydra.utils.instantiate(cfg.model.model)
    kan.load_ckpt(cfg.model_name+'.ckpt', cfg.save_path)
    dnn = hydra.utils.instantiate(cfg.stack.model)
    _ = dnn(torch.tensor(attack_dataset[0][0]))  # Initialize lazy modules
    dnn.load_state_dict(torch.load(
        Path(cfg.save_path, cfg.stack.name+'.pt')))
    model = utils.StackedModel(kan, dnn)
    model = model.to(device)

    # Make predictions
    print('[INFO] Make predictions')
    preds = []
    labels = []
    for batch in attack_dl:
        y = torch.nn.Softmax(dim=1)(model(batch[0].to(device)))
        preds.append(y)
        labels.append(batch[1])
    preds = torch.cat(preds).detach().to(cpu).numpy()

    # Calc confmat
    labels = torch.cat(labels).numpy()
    preds_class = np.argmax(preds, axis=1)
    accuracy = np.mean(labels == preds_class)
    confmat = sklearn.metrics.confusion_matrix(labels, preds_class)
    partial_sum = np.sum(confmat, axis=1)
    print(f'[INFO] Label distributioins: {partial_sum}')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _ = seaborn.heatmap(confmat/partial_sum, cmap='Blues')
    ax.set_title(f'Accuracy: {accuracy:.2f}')
    fig.savefig(Path(cfg.save_path, f'{cfg.stack.name}_confmat.png'), dpi=300)

    # Evaluate
    print('[INFO] Calculate guessing entropy')
    correct_key = attack_dataset.key[0][cfg.target_byte]
    key_hyposesis = range(256)
    label_hyposesis = utils.make_label_hyposesis(
        attack_dataset, key_hyposesis)
    ge = sca_utils.calc_guessing_entropy(
        preds, label_hyposesis, correct_key,
        cfg.n_attack_traces, n_trial=cfg.n_trials)

    # Save results
    print(f'[INFO] GE@{cfg.n_attack_traces}:{ge[-1]}')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ge)
    ax.set_title(f'GE@{cfg.n_attack_traces}:{ge[-1]}')
    fig.savefig(Path(cfg.save_path, f'{cfg.stack.name}.png'), dpi=300)


if __name__ == '__main__':
    run()
