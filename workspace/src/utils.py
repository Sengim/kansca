import torch
import numpy as np


def to_KAN_dataset(
        train_dataloader, test_dataloader, device=torch.device('cpu')):
    """ Make dataset for KAN training from torch.utils.data.DataLoader

        Args:
            train_dataloader: torch.utils.data.Dataloader
                Dataloader for training dataset
            test_dataloader: torch.utils.data.Dataloader
                Dataloader for test dataset
            device: torch.device
                Device

        Returns: dict
            Dictionary of dataset for KAN training
    """
    def extract_samples(dataloader):
        x = []
        t = []
        for batch in dataloader:
            x.append(batch[0])
            t.append(batch[1])
        return torch.cat(x), torch.cat(t)

    KANds = {}
    # Training data
    x, t = extract_samples(train_dataloader)
    KANds['train_input'] = x.to(torch.float64).to(device)
    KANds['train_label'] = t.to(torch.int64).to(device)

    # Test (attack) data
    x, t = extract_samples(test_dataloader)
    KANds['test_input'] = x.to(torch.float64).to(device)
    KANds['test_label'] = t.to(torch.int64).to(device)

    return KANds


def make_label_hyposesis(attack_dataset, key_hyposesis):
    label_hyposesis = []
    for k in key_hyposesis:
        attack_dataset.set_key_hyposesis(k)
        label_hyposesis.append([v[1] for v in attack_dataset])
    label_hyposesis = np.array(label_hyposesis, dtype=np.int32)

    return label_hyposesis


class StackedModel(torch.nn.Module):
    def __init__(self, kan, dnn):
        super().__init__()
        self.dnn = dnn
        self.kan = kan

    def forward(self, x):
        y = self.dnn(x)
        return self.kan(y)
