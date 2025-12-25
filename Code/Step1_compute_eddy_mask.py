#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.ndimage import gaussian_filter, label, find_objects

# ===================== 改这里 =====================
SSH_NPY = r"data/trainAVISO-SSH_2000-2010.npy"
OUT_DIR = r"data"

Y0 = 25
BOX = 168
W_FULL = 280

SIGMA_LARGE = 10.0
SIGMA_SMALL = 1.0
THRESH_SEED = 0.08
THRESH_GROW = 0.05

# 这里决定你 mask 的生成逻辑：hysteresis + CC
INCLUDE_CC = True
MIN_AREA = 15
MAX_AREA = 400

# 如果你的 eddy_mask 还有别的过滤（比如 compactness/hole filling），也要加在这里，否则不可能对齐
PRINT_EVERY = 10
# ================================================


def crop_168_box(arr: np.ndarray, y0: int = Y0, box: int = BOX, w_full: int = W_FULL) -> np.ndarray:
    x0 = w_full - box
    if arr.ndim == 3:
        return arr[:, y0:y0 + box, x0:x0 + box]
    elif arr.ndim == 2:
        return arr[y0:y0 + box, x0:x0 + box]
    else:
        raise ValueError(arr.shape)


def ocean_mask_from_ssh(ssh2d: np.ndarray) -> np.ndarray:
    return (np.isfinite(ssh2d) & (ssh2d != 0.0)).astype(np.float32)


def weighted_gaussian(field: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    num = gaussian_filter(field * weight, sigma=sigma, mode="nearest")
    den = gaussian_filter(weight, sigma=sigma, mode="nearest")
    return num / (den + 1e-6)


def compute_sla(ssh2d: np.ndarray):
    ocean = ocean_mask_from_ssh(ssh2d)
    bg = weighted_gaussian(ssh2d.astype(np.float32), ocean, SIGMA_LARGE)
    sla = ssh2d - bg
    sla = np.where(ocean > 0, sla, np.nan)

    if SIGMA_SMALL and SIGMA_SMALL > 0:
        sla = weighted_gaussian(np.nan_to_num(sla, nan=0.0), ocean, SIGMA_SMALL)
        sla = np.where(ocean > 0, sla, np.nan)

    return sla.astype(np.float32), (ocean > 0)


def cc_keep_mask(binary_mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    structure = np.ones((3, 3), dtype=int)
    lab, n = label(binary_mask, structure=structure)
    out = np.zeros_like(binary_mask, dtype=np.uint8)
    if n == 0:
        return out
    slices = find_objects(lab)
    for comp_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        region = (lab[sl] == comp_id)
        area = int(region.sum())
        if area < min_area or area > max_area:
            continue
        out[sl][region] = 1
    return out


def build_threshold_facts(ssh2d: np.ndarray):
    """返回阈值事实：pos_seed,pos_grow,neg_seed,neg_grow,ocean"""
    sla, ocean = compute_sla(ssh2d)
    pos_seed = ((sla >= +THRESH_SEED) & ocean).astype(np.uint8)
    pos_grow = ((sla >= +THRESH_GROW) & ocean).astype(np.uint8)
    neg_seed = ((sla <= -THRESH_SEED) & ocean).astype(np.uint8)
    neg_grow = ((sla <= -THRESH_GROW) & ocean).astype(np.uint8)
    return pos_seed, pos_grow, neg_seed, neg_grow, ocean.astype(np.uint8)


def hysteresis_mask(grow: np.ndarray, seed: np.ndarray, ocean: np.ndarray):
    """grow 的连通域中，仅保留包含 seed 的连通域（0/1输出）"""
    structure = np.ones((3, 3), dtype=int)
    lab, n = label((grow.astype(bool) & ocean.astype(bool)), structure=structure)
    out = np.zeros_like(grow, dtype=np.uint8)
    if n == 0:
        return out

    slices = find_objects(lab)
    for comp_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        region = (lab[sl] == comp_id)
        # 必须含 seed
        if not np.any(seed[sl].astype(bool) & region):
            continue
        out[sl][region] = 1
    return out


def build_eddy_mask_from_facts(pos_seed, pos_grow, neg_seed, neg_grow, ocean):
    """用 facts 生成最终 eddy_mask（0/1/2）"""
    # hysteresis：grow 连通域里必须包含 seed
    pos = hysteresis_mask(pos_grow, pos_seed, ocean)
    neg = hysteresis_mask(neg_grow, neg_seed, ocean)

    if INCLUDE_CC:
        pos = cc_keep_mask(pos.astype(bool), MIN_AREA, MAX_AREA)
        neg = cc_keep_mask(neg.astype(bool), MIN_AREA, MAX_AREA)

    m = np.zeros_like(pos, dtype=np.int8)
    m[neg == 1] = 1
    m[pos == 1] = 2
    return m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ssh = crop_168_box(np.load(SSH_NPY).astype(np.float32))
    T, H, W = ssh.shape

    facts_pos_seed = np.zeros((T, H, W), np.uint8)
    facts_pos_grow = np.zeros((T, H, W), np.uint8)
    facts_neg_seed = np.zeros((T, H, W), np.uint8)
    facts_neg_grow = np.zeros((T, H, W), np.uint8)
    facts_ocean    = np.zeros((T, H, W), np.uint8)
    eddy_mask      = np.zeros((T, H, W), np.int8)

    print(f"SSH crop shape={ssh.shape}")
    print(f"Params: sigma_large={SIGMA_LARGE}, sigma_small={SIGMA_SMALL}, seed={THRESH_SEED}, grow={THRESH_GROW}")
    print(f"Mask logic: hysteresis + CC={INCLUDE_CC} (min_area={MIN_AREA}, max_area={MAX_AREA})")

    for t in range(T):
        ps, pg, ns, ng, oc = build_threshold_facts(ssh[t])
        m = build_eddy_mask_from_facts(ps, pg, ns, ng, oc)

        facts_pos_seed[t] = ps
        facts_pos_grow[t] = pg
        facts_neg_seed[t] = ns
        facts_neg_grow[t] = ng
        facts_ocean[t]    = oc
        eddy_mask[t]      = m

        # --- 一致性核对（关键）---
        mask_pos = (m == 2)
        mask_neg = (m == 1)

        # 1) mask 必须是 grow 的子集（否则说明你 eddy_mask 的生成逻辑不是基于 grow）
        bad_pos_outside_grow = int((mask_pos & (pg == 0)).sum())
        bad_neg_outside_grow = int((mask_neg & (ng == 0)).sum())

        # 2) hysteresis 约束：每个保留连通域里必须含 seed（用像素级近似统计：mask 内 seed 像素数）
        seed_in_pos = int((mask_pos & (ps == 1)).sum())
        seed_in_neg = int((mask_neg & (ns == 1)).sum())

        if (t + 1) % PRINT_EVERY == 0 or t == T - 1:
            print(f"[{t+1:4d}/{T}] "
                  f"bad_pos_outside_grow={bad_pos_outside_grow} bad_neg_outside_grow={bad_neg_outside_grow} | "
                  f"seed_in_pos={seed_in_pos} seed_in_neg={seed_in_neg} | "
                  f"mask_nz={(m!=0).sum()}")

    # 保存
    np.save(os.path.join(OUT_DIR, "facts_pos_seed_2000-2010.npy"), facts_pos_seed.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "facts_pos_grow_2000-2010.npy"), facts_pos_grow.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "facts_neg_seed_2000-2010.npy"), facts_neg_seed.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "facts_neg_grow_2000-2010.npy"), facts_neg_grow.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "facts_ocean_2000-2010.npy"), facts_ocean.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "eddy_mask_2000-2010.npy"), eddy_mask)

    print("\nSaved:")
    print(" - facts_pos_seed_2000-2010.npy / facts_pos_grow_2000-2010.npy / facts_neg_seed_2000-2010.npy / facts_neg_grow_2000-2010.npy / facts_ocean_2000-2010.npy")
    print(" - eddy_mask_2000-2010.npy")
if __name__ == "__main__":
    main()
