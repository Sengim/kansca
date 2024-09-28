import torch
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
    n = 0
    log = {
        'train/loss': [],
        'test/loss': []
    }

    pbar = tqdm(total=steps)
    while n < steps:
        for batch in train_dl:
            loss = train_step(model, batch, opt, loss_fn, device)
            log['train/loss'].append(loss)

            n += 1
            pbar.update(1)
            if n >= steps:
                break

        for batch in test_dl:
            loss = test_step(model, batch, loss_fn, device)
            log['test/loss'].append(loss)

    return log
