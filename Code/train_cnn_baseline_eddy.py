#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ===================== 路径/设置（改这里） =====================
SSH_NPY   = r"data/trainAVISO-SSH_2000-2010.npy"     # (T,200,280) 或 (T,168,168)
EDDY_NPY  = r"data/eddy_mask_2000-2010.npy"          # (T,168,168) or (T,200,280)
Y0 = 25
BOX = 168
W_FULL = 280

# 涡识别（诊断）：当天 SSH -> 当天 mask
IN_LEN = 1
OUT_OFFSET = 0

EPOCHS = 5
BATCH = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
BASE = 16
NUM_WORKERS = 0

# 处理类别不平衡：pos_weight = neg/pos（可先用 auto）
AUTO_POS_WEIGHT = True
MANUAL_POS_WEIGHT = 10.0  # AUTO_POS_WEIGHT=False 时用这个

# 训练加速/稳定
USE_AMP = True  # 有GPU建议 True
CUDNN_BENCHMARK = True

SAVE_DIR = r"Output/model"
SAVE_CKPT = os.path.join(SAVE_DIR, "nn_baseline_eddy_binary_crop168.pt")
# =============================================================


def crop_168_box(arr: np.ndarray, y0: int = Y0, box: int = BOX, w_full: int = W_FULL) -> np.ndarray:
    x0 = w_full - box
    if arr.ndim == 3:
        return arr[:, y0:y0 + box, x0:x0 + box]
    elif arr.ndim == 2:
        return arr[y0:y0 + box, x0:x0 + box]
    else:
        raise ValueError(f"Unsupported ndim={arr.ndim}, shape={arr.shape}")


def maybe_crop_to_roi(arr: np.ndarray, name: str) -> np.ndarray:
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


class EddyBinaryDataset(Dataset):
    """
    输入：过去 IN_LEN 天 SSH ROI -> (IN_LEN,168,168)
    标签：当天涡二分类 -> (168,168) in {0,1}, 1 表示 (mask==1 or 2)
    """
    def __init__(self, ssh_npy: str, eddy_npy: str, t_start: int, t_end: int, augment: bool = False):
        ssh_raw = np.asarray(np.load(ssh_npy, mmap_mode="r"), dtype=np.float32)
        eddy_raw = np.asarray(np.load(eddy_npy, mmap_mode="r"), dtype=np.int64)

        self.ssh = maybe_crop_to_roi(ssh_raw, "SSH")       # (T,168,168)
        self.eddy = maybe_crop_to_roi(eddy_raw, "EDDY")    # (T,168,168)

        if self.ssh.shape[0] != self.eddy.shape[0]:
            raise ValueError(f"T mismatch: SSH T={self.ssh.shape[0]} vs EDDY T={self.eddy.shape[0]}")

        self.augment = augment
        T = self.ssh.shape[0]

        self.valid_t = list(range(t_start + (IN_LEN - 1), min(t_end, T) - OUT_OFFSET))
        if not self.valid_t:
            raise ValueError("No valid samples. Check t_start/t_end/IN_LEN/OUT_OFFSET")

    def __len__(self):
        return len(self.valid_t)

    @staticmethod
    def _simple_aug(x: np.ndarray, y: np.ndarray):
        # x: (C,H,W), y: (H,W)
        r = np.random.randint(0, 4)
        if r:
            x = np.rot90(x, r, axes=(1, 2))
            y = np.rot90(y, r, axes=(0, 1))
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1].copy()
            y = y[:, ::-1].copy()
        if np.random.rand() < 0.5:
            x = x[:, ::-1, :].copy()
            y = y[::-1, :].copy()
        return x, y

    def __getitem__(self, idx):
        t = self.valid_t[idx]
        x = self.ssh[t - IN_LEN + 1: t + 1]  # (IN_LEN,168,168)

        # per-sample 标准化
        m = x.mean()
        s = x.std() + 1e-6
        x = (x - m) / s

        mask3 = self.eddy[t + OUT_OFFSET]    # (168,168) 0/1/2
        y = (mask3 > 0).astype(np.float32)   # 二分类：eddy=1 else 0

        if self.augment:
            x, y = self._simple_aug(x, y)

        # 防止 negative stride / 非连续内存导致 torch.from_numpy 报错
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)

        return torch.from_numpy(x), torch.from_numpy(y)


def conv_block(in_ch, out_ch, p_drop=0.2):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p_drop),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetBinary(nn.Module):
    """小一点的 U-Net，输出 1 通道 logits"""
    def __init__(self, in_channels: int, base: int = BASE):
        super().__init__()
        self.enc1 = conv_block(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1).squeeze(1)  # (B,H,W) logits


def dice_from_probs(p: torch.Tensor, y: torch.Tensor, eps=1e-6) -> torch.Tensor:
    inter = (p * y).sum(dim=(1, 2))
    denom = p.sum(dim=(1, 2)) + y.sum(dim=(1, 2))
    return ((2 * inter + eps) / (denom + eps)).mean()


@torch.no_grad()
def compute_metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, thr=0.5, eps=1e-6):
    p = torch.sigmoid(logits)
    pred = (p >= thr).float()

    tp = (pred * y).sum().item()
    fp = (pred * (1 - y)).sum().item()
    fn = ((1 - pred) * y).sum().item()
    tn = ((1 - pred) * (1 - y)).sum().item()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)
    dice      = 2 * tp / (2 * tp + fp + fn + eps)
    acc       = (tp + tn) / (tp + tn + fp + fn + eps)
    bal_acc   = 0.5 * (tp / (tp + fn + eps) + tn / (tn + fp + eps))

    return dict(acc=acc, bal_acc=bal_acc, precision=precision, recall=recall, f1=f1, iou=iou, dice=dice)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)
    if device == "cuda":
        print("GPU =", torch.cuda.get_device_name(0))

    ssh_shape = np.load(SSH_NPY, mmap_mode="r").shape
    eddy_shape = np.load(EDDY_NPY, mmap_mode="r").shape
    print("raw SSH shape =", ssh_shape)
    print("raw EDDY shape =", eddy_shape)

    T = ssh_shape[0]
    split = int(T * 0.8)
    print("T =", T, "split =", split)

    train_ds = EddyBinaryDataset(SSH_NPY, EDDY_NPY, t_start=0,     t_end=split, augment=True)
    val_ds   = EddyBinaryDataset(SSH_NPY, EDDY_NPY, t_start=split, t_end=T,     augment=False)
    print("train/val lens:", len(train_ds), len(val_ds))

    pin = (device == "cuda")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=pin, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=pin, drop_last=False)

    xb, yb = next(iter(train_loader))
    print("first batch x:", xb.shape, xb.dtype, "y:", yb.shape, yb.dtype, "y mean:", float(yb.mean()))

    # 估计 pos_weight（用于 BCEWithLogitsLoss），缓解不平衡
    if AUTO_POS_WEIGHT:
        pos = 0.0
        tot = 0.0
        for i, (_, y) in enumerate(train_loader):
            pos += y.sum().item()
            tot += y.numel()
            if i >= 20:
                break
        neg = tot - pos
        pos_weight_value = (neg / (pos + 1e-6))
    else:
        pos_weight_value = MANUAL_POS_WEIGHT

    print(f"pos_weight (neg/pos) ≈ {pos_weight_value:.3f}")

    model = UNetBinary(in_channels=IN_LEN).to(device)

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ===== 新 AMP API（消除 FutureWarning）=====
    use_amp = USE_AMP and (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_dice = -1.0

    for ep in range(1, EPOCHS + 1):
        # ===== train =====
        model.train()
        tr_loss = 0.0
        tr_dice = 0.0
        n_tr = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)                     # (B,H,W)
                loss_bce = bce(logits, y)
                p = torch.sigmoid(logits)
                loss_dice = 1.0 - dice_from_probs(p, y)
                loss = 0.7 * loss_bce + 0.3 * loss_dice

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item()
            tr_dice += (1.0 - loss_dice).item()
            n_tr += 1

        tr_loss /= max(1, n_tr)
        tr_dice /= max(1, n_tr)

        # ===== val =====
        model.eval()
        va_loss = 0.0
        n_va = 0
        agg = dict(acc=0, bal_acc=0, precision=0, recall=0, f1=0, iou=0, dice=0)

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                loss_bce = bce(logits, y)
                p = torch.sigmoid(logits)
                loss_dice = 1.0 - dice_from_probs(p, y)
                loss = 0.7 * loss_bce + 0.3 * loss_dice

                m = compute_metrics_from_logits(logits, y, thr=0.5)
                for k in agg:
                    agg[k] += m[k]

                va_loss += loss.item()
                n_va += 1

        va_loss /= max(1, n_va)
        for k in agg:
            agg[k] /= max(1, n_va)

        print(
            f"Epoch {ep:03d} | "
            f"train_loss {tr_loss:.4f} train_dice {tr_dice:.4f} | "
            f"val_loss {va_loss:.4f} val_dice {agg['dice']:.4f} val_iou {agg['iou']:.4f} "
            f"P {agg['precision']:.4f} R {agg['recall']:.4f} F1 {agg['f1']:.4f} "
            f"Acc {agg['acc']:.4f} BalAcc {agg['bal_acc']:.4f}"
        )

        if agg["dice"] > best_val_dice:
            best_val_dice = agg["dice"]
            torch.save({
                "model": model.state_dict(),
                "in_len": IN_LEN,
                "out_offset": OUT_OFFSET,
                "crop": dict(y0=Y0, box=BOX, w_full=W_FULL),
                "pos_weight": float(pos_weight_value),
                "best_val_dice": float(best_val_dice),
            }, SAVE_CKPT)
            print(f"  Saved best (val_dice={best_val_dice:.4f}) -> {SAVE_CKPT}")

    print("Done.")


if __name__ == "__main__":
    main()
