import os
import glob
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from scipy.ndimage import label
from problog.program import PrologString
from problog import get_evaluatable


# =========================
# Config
# =========================
@dataclass
class CFG:
    probs4_dir: str = "Output/probs4"
    ocean_npy: str = "data/facts/facts_ocean_2000-2010.npy"
    out_dir: str = "Output/eddy_prob"

    tau_pos_grow: float = 0.5
    tau_neg_grow: float = 0.5

    min_area: int = 10
    max_area: int = 1000

    connectivity8: bool = True


cfg = CFG()


# =========================
# Utils
# =========================
def structure(connectivity8=True):
    if connectivity8:
        return np.ones((3, 3), dtype=np.int32)
    return np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.int32)


def p_has_seed(p_seed: np.ndarray, mask: np.ndarray) -> float:
    vals = np.clip(p_seed[mask].astype(np.float64), 0.0, 1.0)
    if vals.size == 0:
        return 0.0
    return float(1.0 - np.exp(np.log1p(-vals).sum()))


def problog_keep_single(p_has_seed: float, area_ok: bool) -> float:
    """
    Extremely small ProbLog program.
    This NEVER hangs.
    """
    if not area_ok or p_has_seed <= 0.0:
        return 0.0

    program = PrologString(f"""
        {p_has_seed:.8f}::has_seed.
        area_ok.
        keep :- has_seed, area_ok.
        query(keep).
    """)

    result = get_evaluatable().create_from(program).evaluate()
    return float(next(iter(result.values()), 0.0))


# =========================
# Core inference
# =========================
def infer_one_frame(probs4: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    ps, pg, ns, ng = probs4
    H, W = ps.shape
    out = np.zeros((H, W), dtype=np.float32)

    S = structure(cfg.connectivity8)

    # ---- POSITIVE EDDIES ----
    labels, n = label(pg >= cfg.tau_pos_grow, structure=S)
    areas = np.bincount(labels.ravel())

    for cid in range(1, n + 1):
        area = areas[cid]
        ok = cfg.min_area <= area <= cfg.max_area
        mask = labels == cid
        p = p_has_seed(ps, mask)
        keep_p = problog_keep_single(p, ok)
        if keep_p > 0:
            out[mask] = np.maximum(out[mask], keep_p)

    # ---- NEGATIVE EDDIES ----
    labels, n = label(ng >= cfg.tau_neg_grow, structure=S)
    areas = np.bincount(labels.ravel())

    for cid in range(1, n + 1):
        area = areas[cid]
        ok = cfg.min_area <= area <= cfg.max_area
        mask = labels == cid
        p = p_has_seed(ns, mask)
        keep_p = problog_keep_single(p, ok)
        if keep_p > 0:
            out[mask] = np.maximum(out[mask], keep_p)

    # land mask
    out *= (ocean > 0).astype(np.float32)
    return out


# =========================
# Main
# =========================
def main():
    os.makedirs(cfg.out_dir, exist_ok=True)

    ocean = np.load(cfg.ocean_npy).astype(np.float32)
    files = sorted(glob.glob(os.path.join(cfg.probs4_dir, "probs4_t*.npy")))

    for i, fp in enumerate(files):
        t = int(os.path.basename(fp).split("t")[1].split(".")[0])
        probs4 = np.load(fp).astype(np.float32)
        ocean2d = ocean[t]

        if i == 0:
            print("[debug]")
            print("probs4 min/max:", probs4.min(), probs4.max())
            print("ocean sum:", ocean2d.sum())
            print("pg>=tau:", (probs4[1] >= cfg.tau_pos_grow).sum())

        eddy_prob = infer_one_frame(probs4, ocean2d)
        np.save(os.path.join(cfg.out_dir, f"eddy_prob_t{t:05d}.npy"), eddy_prob)

        if i == 0:
            print("eddy_prob nonzero:", (eddy_prob > 0).sum(),
                  "max:", eddy_prob.max())

    print("Done.")


if __name__ == "__main__":
    main()
