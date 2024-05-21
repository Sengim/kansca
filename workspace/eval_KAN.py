import torch
import hydra
from pathlib import Path
import matplotlib.pyplot as plt

from src import utils, sca_utils


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def run(cfg):
    device = hydra.utils.instantiate(cfg.device)
    cpu = torch.device('cpu')

    # Load dataset
    attack_dataset = hydra.utils.instantiate(cfg.test.dataset)
    attack_dl = torch.utils.data.DataLoader(
        attack_dataset, batch_size=cfg.test.batch_size,
        shuffle=False
    )

    # Prepare model
    model = hydra.utils.instantiate(cfg.model.model)
    model.load_ckpt(cfg.model_name+'.ckpt', cfg.save_path)
    model = model.to(device)

    # Make predictions
    preds = []
    for batch in attack_dl:
        y = torch.nn.Softmax(dim=1)(model(batch[0].to(device)))
        preds.append(y)
    preds = torch.cat(preds).detach().to(cpu).numpy()

    # Evaluate
    correct_key = attack_dataset.key[0][cfg.target_byte]
    key_hyposesis = range(256)
    label_hyposesis = utils.make_label_hyposesis(
        attack_dataset, key_hyposesis)
    ge = sca_utils.calc_guessing_entropy(
        preds, label_hyposesis, correct_key,
        cfg.n_attack_traces, n_trial=cfg.n_trials)

    # Save results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ge)
    ax.set_title(f'GE@{cfg.n_attack_traces}:{ge[-1]}')
    fig.savefig(Path(cfg.save_path, f'{cfg.model.name}.png'), dpi=300)


if __name__ == '__main__':
    run()
