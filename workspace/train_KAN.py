import torch
import hydra
from pathlib import Path
import matplotlib.pyplot as plt

from src import utils


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def run(cfg):
    device = hydra.utils.instantiate(cfg.device)
    cpu = torch.device('cpu')

    # Prepare datasets
    profiling_dataset = hydra.utils.instantiate(cfg.train.dataset)
    test_dataset = hydra.utils.instantiate(cfg.test.dataset)

    # Change torch-style dataset to KAN-style dataset
    train_dataloader = torch.utils.data.DataLoader(
        profiling_dataset, batch_size=cfg.train.batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.test.batch_size, shuffle=False
    )
    KANds = utils.to_KAN_dataset(train_dataloader, test_dataloader)

    # Prepare model
    model = hydra.utils.instantiate(cfg.model.model)
    model = model.to(device)

    # Profiling phase (Train KAN model)
    _ = model.train(
        KANds,
        **hydra.utils.instantiate(cfg.model.train_params)
        )
    _ = model(KANds['test_input'].to(device))
    model = model.to(cpu)
    model.plot(
        folder=Path(cfg.save_path, f'{cfg.model.name}_plot'), scale=10)
    plt.savefig(Path(cfg.save_path, f'{cfg.model.name}_plot.png', dpi=300))

    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    model.save_ckpt(cfg.model_name+'.ckpt', cfg.save_path)


if __name__ == '__main__':
    run()
