# train_eddy_cnn.py
import os
import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Config
# ---------------------------
@dataclass
class CFG:
    # data
    ssh_npy: str = "data/trainAVISO-SSH_2000-2010.npy"
    facts_pos_seed: str = "data/facts/facts_pos_seed_2000-2010.npy"
    facts_pos_grow: str = "data/facts/facts_pos_grow_2000-2010.npy"
    facts_neg_seed: str = "data/facts/facts_neg_seed_2000-2010.npy"
    facts_neg_grow: str = "data/facts/facts_neg_grow_2000-2010.npy"
    facts_ocean: str = "data/facts/facts_ocean_2000-2010.npy"

    # training
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 30
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4

    # Windows 下如果遇到多进程 DataLoader 问题，改成 0
    num_workers: int = 0
    val_ratio: float = 0.1

    # loss weights
    w_bce: float = 1.0
    w_dice: float = 1.0
    w_consistency: float = 0.2  # seed <= grow soft constraint

    # model
    base_ch: int = 32  # U-Net width

    # output
    out_dir: str = "Output/model"
    ckpt_name: str = "best.pt"

cfg = CFG()

# ---------------------------
# ROI crop (align with your baseline)
# ---------------------------
Y0 = 25
BOX = 168
W_FULL = 280

def crop_168_box(arr: np.ndarray, y0: int = Y0, box: int = BOX, w_full: int = W_FULL) -> np.ndarray:
    """
    Your baseline crop:
      y: [y0, y0+box)
      x: [w_full-box, w_full)
    """
    x0 = w_full - box  # 112 when w_full=280, box=168
    if arr.ndim == 3:
        return arr[:, y0:y0 + box, x0:x0 + box]
    elif arr.ndim == 2:
        return arr[y0:y0 + box, x0:x0 + box]
    else:
        raise ValueError(f"Unsupported ndim={arr.ndim}, shape={arr.shape}")

def maybe_crop_to_roi(arr: np.ndarray, name: str) -> np.ndarray:
    """
    Accept:
      - already cropped (T,168,168)
      - original with width 280 and height >= y0+168, then crop to (T,168,168)
    """
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3D (T,H,W). Got shape={arr.shape}")
    T, H, W = arr.shape
    if (H, W) == (BOX, BOX):
        return arr
    if W == W_FULL and H >= (Y0 + BOX) and W >= BOX:
        return crop_168_box(arr)
    raise ValueError(
        f"{name} spatial shape not supported: (H,W)=({H},{W}). "
        f"Expected ({BOX},{BOX}) or original with W={W_FULL}."
    )

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------
# Dataset
# ---------------------------
class EddyFactsDataset(Dataset):
    """
    单时刻：(ssh2d)-> 4通道事实标签
    """
    def __init__(self, ssh: np.ndarray, y4: np.ndarray, ocean: np.ndarray, indices: np.ndarray):
        """
        ssh:   (T,H,W) float32
        y4 :   (T,4,H,W) float32 0/1
        ocean: (T,H,W) float32 0/1
        """
        self.ssh = ssh
        self.y4 = y4
        self.ocean = ocean
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k: int) -> Dict[str, torch.Tensor]:
        t = int(self.indices[k])
        x = self.ssh[t].astype(np.float32)         # (H,W)
        oc = self.ocean[t].astype(np.float32)      # (H,W)
        y = self.y4[t].astype(np.float32)          # (4,H,W)

        # 把非海洋区域置0，避免网络学到“陆地=涡”
        x = np.where(oc > 0, x, 0.0)

        # per-sample ocean normalization（只在海洋上做统计）
        ocean_vals = x[oc > 0]
        if ocean_vals.size > 10:
            mu = float(ocean_vals.mean())
            sd = float(ocean_vals.std() + 1e-6)
            x = (x - mu) / sd

        # contiguous to avoid negative stride issues after any ops
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        oc = np.ascontiguousarray(oc, dtype=np.float32)

        x = torch.from_numpy(x).unsqueeze(0)      # (1,H,W)
        y = torch.from_numpy(y)                   # (4,H,W)
        oc = torch.from_numpy(oc).unsqueeze(0)    # (1,H,W)

        return {"x": x, "y": y, "ocean": oc}

# ---------------------------
# Model: small U-Net
# ---------------------------
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=4, base=32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)

        self.bott = conv_block(base*4, base*8)

        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = conv_block(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_block(base*2, base)

        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bott(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)  # logits (B,4,H,W)

# ---------------------------
# Losses
# ---------------------------
def dice_loss_with_logits(logits, targets, mask=None, eps=1e-6):
    """
    logits/targets: (B,C,H,W)
    mask: (B,1,H,W) ocean mask, 1=valid
    """
    probs = torch.sigmoid(logits)
    if mask is not None:
        probs = probs * mask
        targets = targets * mask

    num = 2.0 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    dice = 1.0 - (num / den)  # (B,C)
    return dice.mean()

def bce_loss_with_logits(logits, targets, mask=None):
    if mask is None:
        return F.binary_cross_entropy_with_logits(logits, targets)
    w = mask.expand_as(targets)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = (loss * w).sum() / (w.sum() + 1e-6)
    return loss

def consistency_loss(logits):
    """
    soft constraint: seed <= grow
    channel order:
      0 pos_seed, 1 pos_grow, 2 neg_seed, 3 neg_grow
    """
    p = torch.sigmoid(logits)
    pos_seed, pos_grow = p[:, 0], p[:, 1]
    neg_seed, neg_grow = p[:, 2], p[:, 3]
    return (F.relu(pos_seed - pos_grow).mean() + F.relu(neg_seed - neg_grow).mean())

# ---------------------------
# Train / Eval
# ---------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot = 0.0
    n = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        oc = batch["ocean"].to(device, non_blocking=True)
        logits = model(x)
        loss = bce_loss_with_logits(logits, y, oc) + dice_loss_with_logits(logits, y, oc)
        tot += float(loss.item())
        n += 1
    return tot / max(n, 1)

def train():
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # ---- load ----
    ssh_raw = np.asarray(np.load(cfg.ssh_npy, mmap_mode="r"), dtype=np.float32)
    pos_seed_raw = np.asarray(np.load(cfg.facts_pos_seed, mmap_mode="r"), dtype=np.float32)
    pos_grow_raw = np.asarray(np.load(cfg.facts_pos_grow, mmap_mode="r"), dtype=np.float32)
    neg_seed_raw = np.asarray(np.load(cfg.facts_neg_seed, mmap_mode="r"), dtype=np.float32)
    neg_grow_raw = np.asarray(np.load(cfg.facts_neg_grow, mmap_mode="r"), dtype=np.float32)
    ocean_raw    = np.asarray(np.load(cfg.facts_ocean, mmap_mode="r"), dtype=np.float32)

    # ---- align to ROI (T,168,168) ----
    ssh      = maybe_crop_to_roi(ssh_raw,      "SSH")
    pos_seed = maybe_crop_to_roi(pos_seed_raw, "pos_seed")
    pos_grow = maybe_crop_to_roi(pos_grow_raw, "pos_grow")
    neg_seed = maybe_crop_to_roi(neg_seed_raw, "neg_seed")
    neg_grow = maybe_crop_to_roi(neg_grow_raw, "neg_grow")
    ocean    = maybe_crop_to_roi(ocean_raw,    "ocean")

    assert ssh.shape == ocean.shape == pos_seed.shape == pos_grow.shape == neg_seed.shape == neg_grow.shape, \
        f"shape mismatch after crop: ssh={ssh.shape}, ocean={ocean.shape}, pos_seed={pos_seed.shape}"

    T, H, W = ssh.shape
    y4 = np.stack([pos_seed, pos_grow, neg_seed, neg_grow], axis=1)  # (T,4,H,W)

    # ---- split ----
    idx = np.arange(T)
    np.random.shuffle(idx)
    n_val = int(T * cfg.val_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    ds_tr = EddyFactsDataset(ssh, y4, ocean, tr_idx)
    ds_va = EddyFactsDataset(ssh, y4, ocean, val_idx)

    pin = (cfg.device == "cuda")
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=pin, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=pin)

    model = UNet(in_ch=1, out_ch=4, base=cfg.base_ch).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # cosine schedule
    steps_per_epoch = len(dl_tr)
    total_steps = cfg.epochs * steps_per_epoch

    def lr_lambda(step):
        warm = max(1, int(0.03 * total_steps))
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    use_amp = (cfg.device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = 1e9

    for ep in range(1, cfg.epochs + 1):
        model.train()
        run = 0.0

        for batch in dl_tr:
            x = batch["x"].to(cfg.device, non_blocking=True)
            y = batch["y"].to(cfg.device, non_blocking=True)
            oc = batch["ocean"].to(cfg.device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss_bce = bce_loss_with_logits(logits, y, oc)
                loss_dice = dice_loss_with_logits(logits, y, oc)
                loss_cons = consistency_loss(logits)
                loss = cfg.w_bce * loss_bce + cfg.w_dice * loss_dice + cfg.w_consistency * loss_cons

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()

            run += float(loss.item())

        val = evaluate(model, dl_va, cfg.device)
        trn = run / max(len(dl_tr), 1)
        print(f"epoch {ep:03d} | train={trn:.4f} val={val:.4f} lr={opt.param_groups[0]['lr']:.2e}")

        if val < best:
            best = val
            ckpt_path = os.path.join(cfg.out_dir, cfg.ckpt_name)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "crop": dict(y0=Y0, box=BOX, w_full=W_FULL)}, ckpt_path)
            print(f"  saved -> {ckpt_path}")

if __name__ == "__main__":
    train()
