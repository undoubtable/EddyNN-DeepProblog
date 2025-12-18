#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ===================== 你只需要改这里 =====================
SSH_NPY = r"data/testAVISO-SSH_2011.npy"
NEW_MASK_NPY = r"data/eddy_mask_2011.npy"
OLD_MASK_NPY = r"data/testSegmentation_2011.npy"   # 不需要对比就设 None

DAY = 10                 # 要画哪一天（0-based）
NEW_MASK_T0 = 0          # 如果 new mask 是子集（t_start>0），填 t_start

# ROI crop（⚠️必须与你前面所有代码一致）
Y0 = 25
BOX = 168
W_FULL = 280

SIGMA_LARGE = 20.0
SIGMA_SMALL = 1.0

OUT_PNG = rf"Output/eddy_compare_day_{DAY}.png"
# =========================================================


def crop_168_box(arr: np.ndarray, y0: int = Y0, box: int = BOX, w_full: int = W_FULL) -> np.ndarray:
    """(H,W)->(168,168) or (T,H,W)->(T,168,168)"""
    x0 = w_full - box
    if arr.ndim == 2:
        return arr[y0:y0 + box, x0:x0 + box]
    elif arr.ndim == 3:
        return arr[:, y0:y0 + box, x0:x0 + box]
    else:
        raise ValueError(arr.shape)


def compute_bg_sla(ssh2d, sigma_large, sigma_small):
    bg = gaussian_filter(ssh2d, sigma=sigma_large, mode="nearest")
    sla = ssh2d - bg
    if sigma_small and sigma_small > 0:
        sla = gaussian_filter(sla, sigma=sigma_small, mode="nearest")
    return bg, sla


def main():
    # ---------- load ----------
    if not os.path.exists(SSH_NPY):
        raise FileNotFoundError(SSH_NPY)
    if not os.path.exists(NEW_MASK_NPY):
        raise FileNotFoundError(NEW_MASK_NPY)
    if OLD_MASK_NPY is not None and not os.path.exists(OLD_MASK_NPY):
        raise FileNotFoundError(OLD_MASK_NPY)

    ssh = np.load(SSH_NPY)
    new_mask = np.load(NEW_MASK_NPY)

    ssh2d_full = ssh[DAY]
    ssh2d = crop_168_box(ssh2d_full)        # ✅ crop SSH

    idx_new = DAY - NEW_MASK_T0
    if idx_new < 0 or idx_new >= new_mask.shape[0]:
        raise ValueError(
            f"DAY={DAY} not in new_mask range: idx={idx_new}, "
            f"new_mask.shape={new_mask.shape}, NEW_MASK_T0={NEW_MASK_T0}"
        )

    newm = new_mask[idx_new]
    if newm.shape != (BOX, BOX):
        raise ValueError(f"new mask shape must be (168,168), got {newm.shape}")

    oldm = None
    if OLD_MASK_NPY is not None:
        oldm_full = np.load(OLD_MASK_NPY)[DAY]
        oldm = crop_168_box(oldm_full)       # ✅ crop old mask

    # ---------- compute ----------
    bg, sla = compute_bg_sla(ssh2d, SIGMA_LARGE, SIGMA_SMALL)

    # ---------- plot ----------
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))

    ax = axs[0, 0]
    im = ax.imshow(ssh2d)
    ax.set_title("SSH (raw, ROI)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axs[0, 1]
    im = ax.imshow(bg)
    ax.set_title("Background (ROI)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axs[0, 2]
    im = ax.imshow(sla)
    ax.set_title("SLA (ROI)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axs[1, 0]
    if oldm is not None:
        im = ax.imshow(oldm, vmin=0, vmax=2)
        ax.set_title("Mask BEFORE (old, ROI)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.axis("off")
        ax.set_title("Mask BEFORE (not provided)")

    ax = axs[1, 1]
    im = ax.imshow(newm, vmin=0, vmax=2)
    ax.set_title("Mask AFTER (new, ROI)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axs[1, 2]
    if oldm is not None:
        diff = (oldm.astype(np.int16) != newm.astype(np.int16)).astype(np.int8)
        im = ax.imshow(diff, vmin=0, vmax=1)
        ax.set_title("DIFF (ROI, 1=different)")
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        ax.axis("off")
        ax.set_title("DIFF (need old mask)")

    for a in axs.ravel():
        if a.has_data():
            a.set_xticks([])
            a.set_yticks([])

    plt.suptitle(f"Day index = {DAY} (ROI 168×168)", y=0.98)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print(f"[OK] Saved figure -> {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
