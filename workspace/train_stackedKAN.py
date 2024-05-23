import torch
import hydra
from pathlib import Path
from src import trainDNN, utils


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def run(cfg):
    device = hydra.utils.instantiate(cfg.device)
    cpu = torch.device('cpu')

    # Prepare datasets
    profiling_dataset = hydra.utils.instantiate(cfg.train_stack.dataset)
    test_dataset = hydra.utils.instantiate(cfg.test_stack.dataset)

    # Change torch-style dataset to KAN-style dataset
    train_dataloader = torch.utils.data.DataLoader(
        profiling_dataset,
        batch_size=cfg.model.train_params.batch, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg.model.train_params.batch, shuffle=False
    )

    # Prepare model
    kan = hydra.utils.instantiate(cfg.model.model)
    kan.load_ckpt(cfg.model_name+'.ckpt', cfg.save_path)
    dnn = hydra.utils.instantiate(cfg.stack.model)
    _ = dnn(torch.tensor(profiling_dataset[0][0]))  # Initialize lazy modules
    model = utils.StackedModel(kan, dnn)
    model = model.to(device)

    # Profiling phase (Train KAN model)
    train_kwargs = hydra.utils.instantiate(cfg.stack.train_params)
    train_kwargs.opt = train_kwargs.opt(model.dnn.parameters())
    _ = trainDNN.train(
        model,
        train_dataloader,
        test_dataloader,
        **train_kwargs
        )
    model = model.to(cpu)

    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    torch.save(model.dnn.state_dict(), Path(cfg.save_path, cfg.stack.name+'.pt'))


if __name__ == '__main__':
    run()
