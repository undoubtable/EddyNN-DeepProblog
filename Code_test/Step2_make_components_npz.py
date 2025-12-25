import numpy as np
from scipy.ndimage import label, find_objects

# ===== 可调参数 =====
MIN_AREA = 15
MAX_AREA = 400
PATCH = 32
OVERLAP_THR = 0.5
CONNECTIVITY8 = True

def crop_centered_patch(img2d, bbox, patch=32):
    (y0, y1, x0, x1) = bbox
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    half = patch // 2

    ys = max(0, cy - half); ye = min(img2d.shape[0], cy + half)
    xs = max(0, cx - half); xe = min(img2d.shape[1], cx + half)

    out = np.zeros((patch, patch), dtype=np.float32)
    out[(half-(cy-ys)):(half+(ye-cy)), (half-(cx-xs)):(half+(xe-cx))] = img2d[ys:ye, xs:xe]
    return out

def normalize_patch(patch_img, eps=1e-6, clip=5.0):
    """简单标准化 + 截断，避免极值导致训练发散"""
    patch_img = patch_img.astype(np.float32)
    m = patch_img.mean()
    s = patch_img.std()
    if s < eps:
        return np.zeros_like(patch_img, dtype=np.float32)
    patch_img = (patch_img - m) / (s + eps)
    patch_img = np.clip(patch_img, -clip, clip)
    return patch_img.astype(np.float32)

def build(
    ssh_npy,
    pos_grow_npy,
    neg_grow_npy,
    ocean_npy,
    eddy_mask_npy,
    out_npz="components_train.npz",
):
    ssh = np.load(ssh_npy).astype(np.float32)          # (T,H,W)
    pos_grow = np.load(pos_grow_npy).astype(np.uint8)  # (T,H,W)
    neg_grow = np.load(neg_grow_npy).astype(np.uint8)
    ocean = np.load(ocean_npy).astype(np.uint8)
    m = np.load(eddy_mask_npy).astype(np.uint8)        # 0/1/2 (1=neg, 2=pos)

    patches, areas, labels = [], [], []

    if CONNECTIVITY8:
        structure = np.ones((3,3), dtype=int)
    else:
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=int)

    T = ssh.shape[0]
    dropped_nan = 0
    dropped_contradiction = 0
    dropped_empty = 0

    for t in range(T):
        # 关键：先把 NaN/Inf 清掉，否则 patch 里会传染到 loss
        img = np.nan_to_num(ssh[t], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        oc = ocean[t].astype(bool)

        for grow, gt_val in [(pos_grow[t], 2), (neg_grow[t], 1)]:
            grow = (grow.astype(bool) & oc)

            lab, n = label(grow.astype(np.uint8), structure=structure)
            if n == 0:
                continue
            sls = find_objects(lab)

            for cid, sl in enumerate(sls, start=1):
                if sl is None:
                    continue
                region = (lab[sl] == cid)
                area = int(region.sum())
                if area == 0:
                    dropped_empty += 1
                    continue

                # 用 eddy_mask 给样本打标签（对象级 keep / drop）
                gt_region = ((m[t][sl] == gt_val) & region)
                overlap = gt_region.sum() / (area + 1e-9)
                y = 1 if overlap >= OVERLAP_THR else 0

                # 关键：过滤“逻辑上不可能为真”的正样本
                # 如果你的规则里 keep 必须 area_ok，但这个样本 area 不在范围内，
                # 那么 label=1 的 query 概率永远是 0，会导致 log(0)->inf->nan。
                if y == 1 and not (MIN_AREA <= area <= MAX_AREA):
                    dropped_contradiction += 1
                    continue

                # 计算 bbox
                ys, xs = np.where(region)
                y0, y1 = ys.min()+sl[0].start, ys.max()+sl[0].start+1
                x0, x1 = xs.min()+sl[1].start, xs.max()+sl[1].start+1

                patch_img = crop_centered_patch(img, (y0,y1,x0,x1), patch=PATCH)
                patch_img = np.nan_to_num(patch_img, nan=0.0, posinf=0.0, neginf=0.0)

                # 可选：做简单标准化/截断，让训练稳定
                patch_img = normalize_patch(patch_img)

                # 最后再确认 patch 没有 nan/inf
                if (not np.isfinite(patch_img).all()):
                    dropped_nan += 1
                    continue

                patches.append(patch_img[None, ...])  # (1,P,P)
                areas.append(area)
                labels.append(y)

    patches = np.stack(patches, axis=0).astype(np.float32) if len(patches) else np.zeros((0,1,PATCH,PATCH), dtype=np.float32)
    areas = np.array(areas, dtype=np.int32)
    labels = np.array(labels, dtype=np.int64)

    np.savez_compressed(out_npz, patches=patches, areas=areas, labels=labels)
    print("saved:", out_npz, patches.shape, areas.shape, labels.shape)
    print("dropped_nan:", dropped_nan, "dropped_contradiction:", dropped_contradiction, "dropped_empty:", dropped_empty)
    print("pos labels:", int(labels.sum()))

if __name__ == "__main__":
    # 按你自己的路径改
    build(
        ssh_npy=r"D:\Desktop\Neural Symbolic\Code\GITCode\EddyNN-DeepProblog\data\trainAVISO-SSH_2000-2010.npy",
        pos_grow_npy=r"D:\Desktop\Neural Symbolic\Code\GITCode\EddyNN-DeepProblog\data\facts\facts_pos_grow_2000-2010.npy",
        neg_grow_npy=r"D:\Desktop\Neural Symbolic\Code\GITCode\EddyNN-DeepProblog\data\facts\facts_neg_grow_2000-2010.npy",
        ocean_npy=r"D:\Desktop\Neural Symbolic\Code\GITCode\EddyNN-DeepProblog\data\facts\facts_ocean_2000-2010.npy",
        eddy_mask_npy=r"D:\Desktop\Neural Symbolic\Code\GITCode\EddyNN-DeepProblog\data\eddy_mask_2000-2010.npy",
        out_npz=r"D:\Desktop\Neural Symbolic\Code\GITCode\EddyNN-DeepProblog\data\pretain\components_train.npz",
    )
