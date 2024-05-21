import torch
import hydra
from pathlib import Path
from tqdm import tqdm


def train_step(model, batch, optim, loss_fn, device):
    x, t = batch
    x = x.to(device)
    t = t.to(device)

    optim.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, t)
    loss.backward()
    optim.step()

    return loss.detach().to(torch.device('cpu')).numpy()


def test_step(model, batch, loss_fn, device):
    x, t = batch
    x = x.to(device)
    t = t.to(device)

    pred = model(x)
    loss = loss_fn(pred, t)

    return loss.detach().to(torch.device('cpu')).numpy()


def train(model, train_dl, test_dl, opt, loss_fn, steps, device, **kwargs):
    optim = opt(model.parameters())
    n = 0
    log = {
        'train/loss': [],
        'test/loss': []
    }

    pbar = tqdm(total=steps)
    while n < steps:
        for batch in train_dl:
            loss = train_step(model, batch, optim, loss_fn, device)
            log['train/loss'].append(loss)

            n += 1
            pbar.update(1)
            if n >= steps:
                break

        for batch in test_dl:
            loss = test_step(model, batch, loss_fn, device)
            log['test/loss'].append(loss)

    return log


@hydra.main(config_path='conf', config_name='config', version_base='1.1')
def run(cfg):
    device = hydra.utils.instantiate(cfg.device)
    cpu = torch.device('cpu')

    # Prepare datasets
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
    model = hydra.utils.instantiate(cfg.model.model)
    model = model.to(device)

    # Profiling phase (Train KAN model)
    _ = train(
        model,
        train_dataloader,
        test_dataloader,
        **hydra.utils.instantiate(cfg.model.train_params)
        )
    model = model.to(cpu)

    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), Path(cfg.save_path, cfg.model_name+'.pt'))


if __name__ == '__main__':
    run()
