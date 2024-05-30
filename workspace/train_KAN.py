import torch
import hydra
from pathlib import Path
from src import utils


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
    print('[INFO] Start training process')
    model = hydra.utils.instantiate(cfg.model.model)
    model = model.to(device)

    # Profiling phase (Train KAN model)
    _ = model.train(
        KANds,
        **hydra.utils.instantiate(cfg.model.train_params)
        )

    print(
        f'[INFO] Save trained model to {cfg.save_path}/{cfg.model_name}.ckpt')
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    model.to(cpu).save_ckpt(cfg.model_name+'.ckpt', cfg.save_path)


if __name__ == '__main__':
    run()
