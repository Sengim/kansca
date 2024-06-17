import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import hydra
from src.utils import plot_KAN


@hydra.main(config_path='conf', config_name='config', version_base='1.3')
def run(cfg):
    device = hydra.utils.instantiate(cfg.device)

    profiling_dataset = hydra.utils.instantiate(cfg.train.dataset)
    test_dataset = hydra.utils.instantiate(cfg.test.dataset)
    n_samples = cfg.n_attack_traces

    idx = np.random.choice(
        np.arange(len(profiling_dataset)), size=n_samples, replace=False)
    prof_inputs = torch.tensor(
        np.array([profiling_dataset[v][0] for v in idx])).to(device)

    idx = np.random.choice(
        np.arange(len(test_dataset)), size=n_samples, replace=False)
    test_inputs = torch.tensor(
        np.array([test_dataset[v][0] for v in idx])).to(device)

    model = hydra.utils.instantiate(cfg.model.model)
    model.load_ckpt(cfg.model_name+'.ckpt', cfg.save_path)

    print('[INFO] Making plot with profiling traces')
    _ = model(prof_inputs)
    model.plot(folder=cfg.save_path+'/profiling/figures')
    plt.savefig(
        Path(cfg.save_path, f'{cfg.model.name}_profiling.png'),
        dpi=300)
    plot_KAN(cfg, model, 'profiling')

    print('[INFO] Making plot with attack traces')
    _ = model(test_inputs)
    model.plot(folder=cfg.save_path+'/attack')
    plt.savefig(
        Path(cfg.save_path, f'{cfg.model.name}_attack.png'),
        dpi=300)
    plot_KAN(cfg, model, 'attack')


if __name__ == '__main__':
    run()
