import os
import numpy as np
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepproblog.dataset import Dataset, DataLoader
from deepproblog.query import Query
from problog.logic import Term, Constant

from deepproblog.network import Network
from deepproblog.model import Model
from deepproblog.engines import ExactEngine
from deepproblog.train import train_model


class ComponentTensorSource:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.x = torch.from_numpy(d["patches"]).float()  # (N,1,P,P)

    def __len__(self):
        return self.x.shape[0]

    @staticmethod
    def _to_int(idx):
        if isinstance(idx, (tuple, list)):
            if len(idx) != 1:
                raise TypeError(f"Unexpected tensor index tuple: {idx}")
            idx = idx[0]
        if hasattr(idx, "value"):
            return int(idx.value)
        return int(str(idx))

    def __getitem__(self, i):
        i = self._to_int(i)
        return self.x[i]


class ComponentQueries(Dataset):
    def __init__(self, npz_path, subset_name="train"):
        d = np.load(npz_path)
        self.areas = d["areas"].tolist()
        self.labels = d["labels"].tolist()
        self.subset = subset_name

    def __len__(self):
        return len(self.labels)

    def to_query(self, i):
        area = int(self.areas[i])
        y = int(self.labels[i])
        sub = {Term("Img"): Term("tensor", Term(self.subset, Constant(i)))}
        return Query(Term("keep_label", Term("Img"), Constant(area), Constant(y)), sub)



class SeedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        logits = self.net(x)
        # 转成概率并避免0/1导致log(0)
        probs = F.softmax(logits, dim=-1)
        probs = probs.clamp(1e-6, 1 - 1e-6)
        return probs


def main():
    npz_path = r"D:\Desktop\Neural Symbolic\Code\GITCode\EddyNN-DeepProblog\data\pretain\components_train.npz"
    pl_path = "eddy_dp.pl"

    print("[CWD]", os.getcwd())
    print("[FILES] npz:", os.path.exists(npz_path), "pl:", os.path.exists(pl_path))

    d = np.load(npz_path)
    print("[DATA] patches:", d["patches"].shape, "labels_sum:", int(d["labels"].sum()))
    if d["patches"].shape[0] == 0:
        raise RuntimeError("components_train.npz has N=0 samples. Re-run make_components_npz.py and check paths.")

    net_module = SeedNet()
    net = Network(net_module, "seednet", batching=True)
    net.optimizer = torch.optim.Adam(net_module.parameters(), lr=1e-3)

    model = Model(pl_path, [net])
    model.set_engine(ExactEngine(model))
    model.add_tensor_source("train", ComponentTensorSource(npz_path))

    dataset = ComponentQueries(npz_path, subset_name="train")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 训练回调：每次 optimizer step 后做梯度裁剪，防止爆炸
    def clip_grad():
        torch.nn.utils.clip_grad_norm_(net_module.parameters(), max_norm=5.0)

    # DeepProbLog 的 train_model 没有直接 hook，我们用一个更稳的策略：
    # 先把 lr 调低 + log 更频繁，如果还 nan 再进一步诊断
    print("[TRAIN] start")
    train_model(model, loader, stop_condition=5, log_iter=20)

    model.save_state("seednet_deepproblog.mdl")
    print("[SAVE] seednet_deepproblog.mdl")


if __name__ == "__main__":
    main()
