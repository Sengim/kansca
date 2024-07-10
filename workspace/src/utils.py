import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import sklearn
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


def make_label_hyposesis(attack_dataset, key_hyposesis, one_hot=False):
    label_hyposesis = []
    for k in tqdm(key_hyposesis,
                  desc="Making label hyposesis [/key hyposesis]"):
        attack_dataset.set_key_hyposesis(k)
        tmp = np.array([attack_dataset.get_label(v)
                        for v in range(len(attack_dataset))])
        if one_hot:
            tmp = tmp.argmax(axis=1)
        label_hyposesis.append(tmp)
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


def Youdens_threshold(label, pred):
    """ Determine threshold for binary classifier by Youden's index.
    """
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(label, pred, pos_label=1)
    index = np.argmax(tpr - fpr)  # Youden's index
    return thresholds[index]


def make_prediction(model, test_dl, device, one_hot, yodens_threshold=False):
    cpu = torch.device('cpu')
    model = model.to(device)
    inputs = []
    preds = []
    labels = []
    for batch in test_dl:
        x = batch[0].to(device)
        y = model(x)
        if one_hot:
            y = torch.nn.functional.softmax(y, dim=1)
        else:
            y = torch.nn.functional.sigmoid(y)
        inputs.append(x.to(cpu))
        preds.append(y.to(cpu))
        labels.append(batch[1])
    inputs = torch.cat(inputs).detach().numpy()
    preds = torch.cat(preds).detach().numpy()
    labels = torch.cat(labels, dim=0).numpy()
    if one_hot:
        labels = np.argmax(labels, axis=1)
        th = None
    else:
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        if yodens_threshold:
            th = Youdens_threshold(labels, preds)
            preds += 0.5-th
            preds = np.clip(preds, 0.0, 1.0)
            print(f'Detection threshold: {th:.3f}')
        else:
            th = None
    labels = labels.astype(np.int32)
    return preds, labels, th


def make_confidence(preds):
    confidence = np.zeros((preds.shape[0], 2), dtype=np.float64)
    confidence[:, 0] = 1 - preds
    confidence[:, 1] = preds
    return confidence


def make_confmat(preds, labels, accuracy, save_path):
    confmat = sklearn.metrics.confusion_matrix(labels, preds)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _ = seaborn.heatmap(
        confmat, cmap='Blues', annot=True, fmt='.0f')
    ax.set_title(f'Accuracy: {accuracy:.2f}')
    fig.savefig(Path(save_path, 'confmat.png'), dpi=300)
    plt.close()
