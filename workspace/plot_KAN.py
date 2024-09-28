import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import hydra
from src.utils import plot_KAN


@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def run(cfg):
    device = hydra.utils.instantiate(cfg.device)

    test_dataset = hydra.utils.instantiate(cfg.test.dataset)
    idx = np.arange(len(test_dataset))[:10000]
    test_inputs = torch.tensor(
        np.array([test_dataset[v][0] for v in idx])).to(device)

    model = hydra.utils.instantiate(cfg.model.model)
    model.load_ckpt(cfg.model_name+'.ckpt', cfg.save_path)

    _ = model(test_inputs)
    model.plot(folder=cfg.save_path+'/graph')
    plt.savefig(
        Path(cfg.save_path, 'graph', f'{cfg.model.name}.png'),
        dpi=300)
    plot_KAN(cfg, model, 'graph')


if __name__ == '__main__':
    run()
