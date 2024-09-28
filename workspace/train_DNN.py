import torch
import hydra
from pathlib import Path
from src import trainDNN


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
        profiling_dataset,
        batch_size=cfg.model.train_params.batch, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.model.train_params.batch, shuffle=False
    )

    # Prepare model
    print('[INFO] Start training process')
    model = hydra.utils.instantiate(cfg.model.model)
    model = model.to(device)

    # Profiling phase (Train KAN model)
    train_kwargs = hydra.utils.instantiate(cfg.model.train_params)
    train_kwargs.opt = train_kwargs.opt(model.parameters())
    _ = trainDNN.train(
        model,
        train_dataloader,
        test_dataloader,
        **train_kwargs
        )
    model = model.to(cpu)

    print(
        f'[INFO] Save trained model to {cfg.save_path}/{cfg.model_name}.pt')
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), Path(cfg.save_path, cfg.model_name+'.pt'))


if __name__ == '__main__':
    run()
