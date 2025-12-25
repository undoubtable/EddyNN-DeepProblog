#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ============ 你只需要改这里 ============
CKPT_PATH = r"Output/model/nn_baseline_eddy_binary_crop168.pt"

SSH_NPY  = r"data/testAVISO-SSH_2011.npy"         # (T,H,W) 或 (T,168,168)
EDDY_NPY = r"data/eddy_mask_2011.npy"       # (T,168,168) 或 (T,H,W)

T_IDX = 0           # 想看的“第几天”（按你 test npy 的索引）
THR = 0.5           # 概率阈值
OUT_DIR = r"Output/Figure"

# ROI（必须与训练一致）
Y0 = 25
BOX = 168
W_FULL = 280
# ======================================


def crop_168_box(arr, y0=Y0, box=BOX, w_full=W_FULL):
    x0 = w_full - box
    if arr.ndim == 3:
        return arr[:, y0:y0+box, x0:x0+box]
    elif arr.ndim == 2:
        return arr[y0:y0+box, x0:x0+box]
    else:
        raise ValueError(arr.shape)

def maybe_crop(arr, name="arr"):
    if arr.ndim != 3:
        raise ValueError(f"{name} must be (T,H,W), got {arr.shape}")
    T, H, W = arr.shape
    if (H, W) == (BOX, BOX):
        return arr
    if W == W_FULL and H >= (Y0 + BOX):
        return crop_168_box(arr)
    raise ValueError(f"{name} unsupported shape {arr.shape}")

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.2),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNetBinary(nn.Module):
    def __init__(self, in_channels=1, base=16):
        super().__init__()
        self.enc1 = conv_block(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base*2, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_block(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_block(base*2, base)

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

        return self.head(d1).squeeze(1)  # (B,H,W)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # load data
    ssh = maybe_crop(np.load(SSH_NPY), "SSH")
    eddy = maybe_crop(np.load(EDDY_NPY), "EDDY")
    assert 0 <= T_IDX < ssh.shape[0], f"T_IDX out of range: {T_IDX} vs T={ssh.shape[0]}"

    ssh_day = ssh[T_IDX].astype(np.float32)           # (168,168)
    gt3 = eddy[T_IDX].astype(np.int64)                # (168,168)
    gt = (gt3 > 0).astype(np.float32)                 # 二分类 GT

    # normalize like training (per-sample)
    x = ssh_day[None, None, :, :]                     # (1,1,H,W)
    m, s = x.mean(), x.std() + 1e-6
    x = (x - m) / s
    x = np.ascontiguousarray(x, dtype=np.float32)

    # load model
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = UNetBinary(in_channels=1, base=16).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        prob = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (168,168)

    pred = (prob >= THR).astype(np.float32)

    # ---- plot ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(ssh_day)
    axes[0, 0].set_title(f"SSH (ROI) t={T_IDX}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt)
    axes[0, 1].set_title("GT eddy mask (binary)")
    axes[0, 1].axis("off")

    im = axes[1, 0].imshow(prob)
    axes[1, 0].set_title(f"Pred prob p_eddy (thr={THR})")
    axes[1, 0].axis("off")
    # fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(pred)
    axes[1, 1].set_title("Pred mask (binary)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"day_{T_IDX:04d}.png")
    plt.savefig(out_path, dpi=200)
    print("Saved figure:", out_path)
    plt.show()

if __name__ == "__main__":
    main()
