import torch


def model(width):
    layers = []
    for n in width:
        layers.append(
            torch.nn.LazyLinear(n)
        )
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers[:-1])
