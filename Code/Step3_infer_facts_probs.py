import os
import numpy as np
import torch

# ====== 跟训练保持一致的 ROI ======
Y0, BOX, W_FULL = 25, 168, 280
def crop_168_box(arr):
    x0 = W_FULL - BOX
    if arr.ndim == 3:
        return arr[:, Y0:Y0+BOX, x0:x0+BOX]
    elif arr.ndim == 2:
        return arr[Y0:Y0+BOX, x0:x0+BOX]
    raise ValueError(arr.shape)

def maybe_crop_to_roi(arr):
    if arr.ndim != 3: raise ValueError(arr.shape)
    T,H,W = arr.shape
    if (H,W) == (BOX,BOX): return arr
    if W == W_FULL and H >= (Y0+BOX): return crop_168_box(arr)
    raise ValueError(arr.shape)

# ====== 你的 UNet 定义要和 train_eddy_cnn.py 一致 ======
import torch.nn as nn
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
        d3 = self.up3(b); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.up2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.up1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.head(d1)

@torch.no_grad()
def main():
    ssh_npy   = "data/trainAVISO-SSH_2000-2010.npy"
    ocean_npy = "data/facts/facts_ocean_2000-2010.npy"
    ckpt_path = "Output/model/best.pt"
    out_dir   = "Output/probs4"
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ssh_raw = np.load(ssh_npy).astype(np.float32)
    oc_raw  = np.load(ocean_npy).astype(np.float32)

    ssh = maybe_crop_to_roi(ssh_raw)   # (T,168,168)
    oc  = maybe_crop_to_roi(oc_raw)    # (T,168,168)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    base_ch = ckpt.get("cfg", {}).get("base_ch", 32) if isinstance(ckpt, dict) else 32

    model = UNet(in_ch=1, out_ch=4, base=base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 逐时刻输出 4 张概率图
    T = ssh.shape[0]
    for t in range(T):
        x = ssh[t].copy()
        ocean = oc[t].copy()

        # 非海洋置0 + 海洋标准化（跟训练一致）
        x = np.where(ocean > 0, x, 0.0)
        vals = x[ocean > 0]
        if vals.size > 10:
            mu, sd = float(vals.mean()), float(vals.std() + 1e-6)
            x = (x - mu) / sd

        xt = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
        logits = model(xt)                                              # (1,4,H,W)
        probs4 = torch.sigmoid(logits).squeeze(0).cpu().numpy()         # (4,H,W)

        np.save(os.path.join(out_dir, f"probs4_t{t:05d}.npy"), probs4)

    print("Done. Saved probs4 to", out_dir)

if __name__ == "__main__":
    main()
