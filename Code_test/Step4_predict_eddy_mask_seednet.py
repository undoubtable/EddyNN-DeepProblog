import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label, find_objects

# ====== 你需要改的路径 ======
SSH_2011_NPY = r"data/testAVISO-SSH_2011.npy"              # (T,H,W) or (T,168,168)
POS_GROW_2011 = r"data/facts_pos_grow_2011.npy"
NEG_GROW_2011 = r"data/facts_neg_grow_2011.npy"
OCEAN_2011    = r"data/facts_ocean_2011.npy"

SEEDNET_PT = r"Output/model_seednet/seednet.pt"           # 建议你从训练脚本另存一个 torch state_dict
OUT_PRED = r"Output/eddy_mask_pred_2011.npy"

# ====== 参数（要和你的 rule 一致）======
MIN_AREA = 15
MAX_AREA = 400
PATCH = 32
TAU_KEEP = 0.5              # 候选 keep 的阈值（先用0.5）
CONNECTIVITY8 = True

# ROI crop：和你其他脚本一致（168x168, y0=25, x0=280-168）
Y0 = 25
BOX = 168
W_FULL = 280

def crop_168_box(arr: np.ndarray, y0: int = Y0, box: int = BOX, w_full: int = W_FULL) -> np.ndarray:
    x0 = w_full - box
    if arr.ndim == 3:
        return arr[:, y0:y0 + box, x0:x0 + box]
    elif arr.ndim == 2:
        return arr[y0:y0 + box, x0:x0 + box]
    else:
        raise ValueError(arr.shape)

def maybe_crop_to_roi(arr: np.ndarray) -> np.ndarray:
    # 若已经是 (T,168,168) 就不裁；否则按 (T, H>=193, W=280) 裁
    if arr.ndim != 3:
        raise ValueError(arr.shape)
    if arr.shape[1] == BOX and arr.shape[2] == BOX:
        return arr
    if arr.shape[2] == W_FULL and arr.shape[1] >= Y0 + BOX:
        return crop_168_box(arr)
    raise ValueError(f"Unexpected shape: {arr.shape}")

def normalize_patch(patch_img: np.ndarray, eps=1e-6, clip=5.0) -> np.ndarray:
    patch_img = patch_img.astype(np.float32)
    m = patch_img.mean()
    s = patch_img.std()
    if s < eps:
        out = np.zeros_like(patch_img, dtype=np.float32)
    else:
        out = (patch_img - m) / (s + eps)
    out = np.clip(out, -clip, clip)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)

def crop_centered_patch(img2d: np.ndarray, sl, patch=32) -> np.ndarray:
    # sl 是 find_objects 给的 slice，代表该连通域的 bbox
    y0, y1 = sl[0].start, sl[0].stop
    x0, x1 = sl[1].start, sl[1].stop
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    half = patch // 2

    ys = max(0, cy - half); ye = min(img2d.shape[0], cy + half)
    xs = max(0, cx - half); xe = min(img2d.shape[1], cx + half)

    out = np.zeros((patch, patch), dtype=np.float32)
    out[(half-(cy-ys)):(half+(ye-cy)), (half-(cx-xs)):(half+(xe-cx))] = img2d[ys:ye, xs:xe]
    return out

def structure(connectivity8=True):
    if connectivity8:
        return np.ones((3, 3), dtype=int)
    return np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)

# 你的 seednet 结构要与训练时一致
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
        probs = F.softmax(logits, dim=-1).clamp(1e-6, 1 - 1e-6)
        return probs  # (B,2)

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ssh_raw = np.load(SSH_2011_NPY).astype(np.float32)
    pos_grow_raw = np.load(POS_GROW_2011).astype(np.uint8)
    neg_grow_raw = np.load(NEG_GROW_2011).astype(np.uint8)
    ocean_raw    = np.load(OCEAN_2011).astype(np.uint8)

    ssh = maybe_crop_to_roi(ssh_raw)
    pos_grow = maybe_crop_to_roi(pos_grow_raw)
    neg_grow = maybe_crop_to_roi(neg_grow_raw)
    ocean    = maybe_crop_to_roi(ocean_raw)

    T, H, W = ssh.shape
    print("ROI shape:", ssh.shape)

    model = SeedNet().to(device)
    sd = torch.load(SEEDNET_PT, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    S = structure(CONNECTIVITY8)

    pred = np.zeros((T, H, W), dtype=np.int8)

    # 为了处理 pos/neg 冲突，先记概率图
    prob_pos = np.zeros((T, H, W), dtype=np.float32)
    prob_neg = np.zeros((T, H, W), dtype=np.float32)

    for t in range(T):
        img = np.nan_to_num(ssh[t], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        oc = (ocean[t] > 0)

        for sign, grow2d, prob_map, label_val in [
            ("pos", pos_grow[t], prob_pos[t], 2),
            ("neg", neg_grow[t], prob_neg[t], 1),
        ]:
            grow = (grow2d > 0) & oc
            lab, n = label(grow.astype(np.uint8), structure=S)
            if n == 0:
                continue

            areas = np.bincount(lab.ravel())
            slices = find_objects(lab)

            # 收集需要跑网络的候选
            patches = []
            metas = []  # (sl, region_mask, pmap_ref)

            for cid in range(1, n + 1):
                area = int(areas[cid])
                if area < MIN_AREA or area > MAX_AREA:
                    continue  # area_ok 不满足，keep 直接为0
                sl = slices[cid - 1]
                if sl is None:
                    continue
                region = (lab[sl] == cid)
                if region.sum() == 0:
                    continue

                patch = crop_centered_patch(img, sl, patch=PATCH)
                patch = normalize_patch(patch)
                patches.append(patch[None, None, ...])  # (1,1,P,P)
                metas.append((sl, region))

            if not patches:
                continue

            batch = torch.from_numpy(np.concatenate(patches, axis=0)).to(device)  # (B,1,P,P)
            probs = model(batch).detach().cpu().numpy()  # (B,2)
            p_keep = probs[:, 1]  # P(seed_present=1) == P(keep)

            # 回填到像素图（连通域内赋同一个 keep 概率）
            for (sl, region), pk in zip(metas, p_keep):
                if pk >= TAU_KEEP:
                    # 注意：sl 局部赋值时要用 np.maximum 叠加
                    block = prob_map[sl]
                    block[region] = np.maximum(block[region], float(pk))
                    prob_map[sl] = block

        if (t + 1) % 20 == 0:
            print(f"[{t+1}/{T}] nonzero pos={int((prob_pos[t]>0).sum())} neg={int((prob_neg[t]>0).sum())}")

    # 合成 0/1/2，冲突时取更大概率
    pred[(prob_neg >= TAU_KEEP) & (prob_neg > prob_pos)] = 1
    pred[(prob_pos >= TAU_KEEP) & (prob_pos >= prob_neg)] = 2

    os.makedirs(os.path.dirname(OUT_PRED) or ".", exist_ok=True)
    np.save(OUT_PRED, pred)
    print("Saved:", OUT_PRED, pred.shape)

if __name__ == "__main__":
    main()
