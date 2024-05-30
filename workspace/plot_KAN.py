import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import hydra


def plot(cfg, model, folder_name):
    Path(cfg.save_path, folder_name).mkdir(exist_ok=True, parents=True)
    for idx, _ in enumerate(cfg.model.model.width[:-1]):
        for i in range(cfg.model.model.width[idx]):
            for j in range(cfg.model.model.width[idx+1]):
                inputs = model.spline_preacts[idx][:, j, i]
                inputs = inputs.to(torch.device('cpu'))
                outputs = model.spline_postacts[idx][:, j, i]
                outputs = outputs.to(torch.device('cpu'))
                rank = np.argsort(inputs)
                inputs = inputs[rank]
                outputs = outputs[rank]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(inputs, outputs, marker="o")
                fig.savefig(
                    Path(cfg.save_path, folder_name, f'{idx}.{i}.{j}.png'),
                    dpi=300)
                plt.close()


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

    _ = model(prof_inputs)
    model.plot(folder=cfg.save_path+'/profiling/figures')
    plt.savefig(
        Path(cfg.save_path, 'profiling', f'{cfg.model.name}_plot.png'),
        dpi=300)
    plot(cfg, model, 'profiling')

    _ = model(test_inputs)
    model.plot(folder=cfg.save_path+'/attack')
    plt.savefig(
        Path(cfg.save_path, 'attack', f'{cfg.model.name}_plot.png'),
        dpi=300)
    plot(cfg, model, 'attack')


if __name__ == '__main__':
    run()
