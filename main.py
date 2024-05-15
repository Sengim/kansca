from pykan.kan import KAN
from utils import load_dataset, guessing_entropy
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.datasets.simulate_higher_order import *

dataset_id = 'ASCAD'
resample_window=80
traces_dim = 15
n_prof= 50000
lm = "HW"
CLASSES=9 if lm == "HW" else 256

class HackyDatasetObject(object):
    pass
def create_torch_dataset(dataset, device="cuda:0"):
    tmp = {}
    tmp["train_input"] = torch.from_numpy(dataset.x_profiling).to(device).double()
    print(tmp['train_input'].dtype)
    tmp["test_input"] = torch.from_numpy(dataset.x_attack).to(device).double()

    tmp['train_label'] = torch.from_numpy(to_categorical(np.array(dataset.profiling_labels))).to(device)
    tmp['test_label'] = torch.from_numpy(to_categorical(np.array(dataset.attack_labels))).to(device)
    return tmp

def to_categorical(y, num_classes=CLASSES):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='double')[y]

target_byte = 2
path = "/mnt/d/Datasets"

#dataset = load_dataset(dataset_id, path, target_byte, traces_dim, leakage_model=lm)
dataset = SimulateHigherOrder(1, n_prof, 10000, 1, traces_dim, leakage_model=lm)

model = KAN(width=[traces_dim, 2, CLASSES], grid=3, k=3, device='cuda:0', seed=0, symbolic_enabled=True)
model.double()
# model.to('cuda:0')
new_dataset = create_torch_dataset(dataset, device=model.device)
print(new_dataset['train_label'].shape)
#model.plot()
model.train(new_dataset,opt="Adam", steps=2000, device=model.device,lamb=0.01, lamb_l1=0.2,lamb_entropy=0.2,  lr=1e-3,batch=200, loss_fn=torch.nn.CrossEntropyLoss())
model.prune()
#To SHow the plot you need ot add plt.show() at end of plot function
model.plot(title="fa")
#model.train(new_dataset,opt="Adam", steps=8000, device=model.device,lamb=0.001, lamb_l1=0.2,lamb_entropy=0.2,  lr=1e-3,batch=200, loss_fn=torch.nn.CrossEntropyLoss())
# y_pred= model(torch.from_numpy(dataset.x_attack[:2000]).to('cuda:0')).cpu().detach().numpy()
# for i in range(1, 5):
#     y_pred = np.append(y_pred,model(torch.from_numpy(dataset.x_attack[i*2000:(i+1)*2000]).to('cuda:0')).cpu().detach().numpy(), axis=0)
#     print(y_pred.shape)
# print(y_pred)
# y_pred = np.clip(y_pred, 0, None)
# sums = np.sum(y_pred, axis=1).reshape(-1,1)
# y_pred = y_pred/sums
# ge, ge_v, nt= guessing_entropy(y_pred, dataset.labels_key_hypothesis_attack, dataset.correct_key_attack, 5000)
# plt.plot(ge_v)
# plt.show()

