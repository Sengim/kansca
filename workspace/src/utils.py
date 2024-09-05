import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sklearn.metrics
import seaborn


def to_torch(v):
    if isinstance(v, torch.Tensor):
        return v
    else:
        return torch.tensor(v)


def to_float(v):
    return v.to(torch.float32)


def to_double(v):
    return v.to(torch.float64)


def to_long(v):
    return v.to(torch.int64)


def to_KAN_dataset(
        train_dataloader, test_dataloader,
        device=torch.device('cpu')):
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
    KANds['train_input'] = x.to(device)
    KANds['train_label'] = t.to(device)

    # Test (attack) data
    x, t = extract_samples(test_dataloader)
    KANds['test_input'] = x.to(device)
    KANds['test_label'] = t.to(device)

    return KANds


def make_label_hyposesis(attack_dataset, key_hyposesis):
    label_hyposesis = []
    for k in tqdm(key_hyposesis,
                  desc="Making label hyposesis [/key hyposesis]"):
        attack_dataset.set_key_hyposesis(k)
        tmp = np.array([attack_dataset.get_label(v)
                        for v in range(len(attack_dataset))])
        tmp = tmp.argmax(axis=1)
        label_hyposesis.append(tmp)
    label_hyposesis = np.array(label_hyposesis, dtype=np.int32)

    return label_hyposesis


def plot_KAN(cfg, model, folder_name):
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


def make_prediction(model, test_dl, device):
    cpu = torch.device('cpu')
    model = model.to(device)
    inputs = []
    preds = []
    labels = []
    for batch in test_dl:
        x = batch[0].to(device)
        y = model(x)
        y = torch.nn.functional.softmax(y, dim=1)
        inputs.append(x.to(cpu))
        preds.append(y.to(cpu))
        labels.append(batch[1].to(cpu))
    inputs = torch.cat(inputs).detach().numpy()
    preds = torch.cat(preds).detach().numpy()
    labels = torch.cat(labels, dim=0).numpy()
    labels = np.argmax(labels, axis=1)
    th = None
    labels = labels.astype(np.int32)
    return preds, labels, th


def make_confmat(preds, labels, accuracy, save_path):
    confmat = sklearn.metrics.confusion_matrix(labels, preds)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _ = seaborn.heatmap(
        confmat, cmap='Blues', annot=True, fmt='.0f')
    ax.set_title(f'Accuracy: {accuracy:.4f}')
    fig.savefig(Path(save_path, 'confmat.png'), dpi=300)
    plt.close()
