"""
Detecting AI-Edited Parts of an Image Using Invisible Watermarks
COMPLETE PIPELINE: train -> evaluate -> robustness -> plots -> saliency

Quick test (no data):   python maskmark_full.py --synthetic --epochs 2
Real run (CASIA 2.0):   python maskmark_full.py --images /path/CASIA2 --masks /path/CASIA2 --epochs 15

Outputs (in --out): maskmark.pt, results.json, roc_pr.png, confusion.png,
                    robustness.png, examples.png (watermark, map, saliency)
"""
import argparse, os, io, json, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from skimage.metrics import structural_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NBITS = 32
W_MSG = 1.0
dev = "cuda" if torch.cuda.is_available() else "cpu"
EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


# =============================== DATA ===============================
def random_mask(size, rng):
    m = np.zeros((size, size), np.float32)
    h, w = rng.randint(size // 6, size // 2 + 1), rng.randint(size // 6, size // 2 + 1)
    y, x = rng.randint(0, size - h + 1), rng.randint(0, size - w + 1)
    if rng.rand() < 0.5:
        m[y:y + h, x:x + w] = 1
    else:
        yy, xx = np.ogrid[:size, :size]
        m[((yy - y - h / 2) / (h / 2)) ** 2 + ((xx - x - w / 2) / (w / 2)) ** 2 <= 1] = 1
    return m


def list_images(root):
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            s, e = os.path.splitext(f)
            if e.lower() in EXTS and not s.endswith("_gt"):
                out.append(os.path.join(d, f))
    return sorted(out)


def index_masks(root):
    idx = {}
    if root:
        for d, _, fs in os.walk(root):
            for f in fs:
                s, e = os.path.splitext(f)
                if e.lower() in EXTS:
                    idx[s] = os.path.join(d, f)
    return idx


class Data(Dataset):
    def __init__(self, files, masks, size, train, synth=0):
        self.files, self.masks, self.size, self.train, self.synth = files, masks, size, train, synth

    def __len__(self):
        return self.synth or len(self.files)

    def __getitem__(self, i):
        rng = np.random.RandomState(None if self.train else i)
        S = self.size
        if self.synth:
            low = torch.from_numpy(rng.rand(1, 3, 8, 8).astype(np.float32))
            img = F.interpolate(low, size=(S, S), mode="bilinear")[0]
            return img, torch.from_numpy(random_mask(S, rng))[None]
        p = self.files[i]
        img = Image.open(p).convert("RGB").resize((S, S))
        img = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)
        stem = os.path.splitext(os.path.basename(p))[0]
        gt = self.masks.get(stem + "_gt") or self.masks.get(stem)
        mask = None
        if gt and gt != p and (not self.train or rng.rand() < 0.5):
            m = Image.open(gt).convert("L").resize((S, S))
            m = (np.asarray(m) > 127).astype(np.float32)
            if m.sum() >= 16:
                mask = m
        if mask is None:
            mask = random_mask(S, rng)
        return img, torch.from_numpy(mask)[None]


# =============================== MODEL ===============================
def cbr(i, o, d=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=d, dilation=d), nn.BatchNorm2d(o), nn.ReLU(True))


class Encoder(nn.Module):
    """watermarked = image + strength * residual * mask  (mask-bounded)"""
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(cbr(3 + 1 + NBITS, ch), cbr(ch, ch), cbr(ch, ch), cbr(ch, ch))
        self.out = nn.Conv2d(ch, 3, 1)
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, img, msg, mask):
        B, _, H, W = img.shape
        m = msg.view(B, NBITS, 1, 1).expand(-1, -1, H, W)
        res = torch.tanh(self.out(self.net(torch.cat([img, mask, m], 1))))
        s = self.strength.clamp(0.02, 0.2)
        return (img + s * res * mask).clamp(0, 1)


class Extractor(nn.Module):
    """Returns bit logits (provenance) and per-pixel watermark logits (localization)."""
    def __init__(self, ch=64):
        super().__init__()
        self.enc = nn.Sequential(cbr(3, ch), cbr(ch, ch, 2), cbr(ch, ch, 4), cbr(ch, ch, 2), cbr(ch, ch))
        self.loc = nn.Conv2d(ch, 1, 1)
        self.bits = nn.Linear(ch, NBITS)

    def forward(self, x):
        f = self.enc(x)
        loc = self.loc(f)
        w = torch.sigmoid(loc)
        pooled = (f * w).sum((2, 3)) / (w.sum((2, 3)) + 1e-6)
        return self.bits(pooled), loc


# =============================== ATTACKS ===============================
def _jpeg(x, q):
    out = []
    for im in x.detach().cpu():
        arr = (im.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "JPEG", quality=q)
        buf.seek(0)
        out.append(torch.from_numpy(np.asarray(Image.open(buf).convert("RGB"), np.float32) / 255).permute(2, 0, 1))
    return torch.stack(out).to(x.device)


def attack(x, m, name):
    """Returns (attacked image, correspondingly transformed mask)."""
    H = x.shape[-1]
    if name == "none":
        return x, m
    if name == "quant8":
        return torch.round(x * 255) / 255, m
    if name.startswith("jpeg"):
        return _jpeg(x, int(name[4:])), m
    if name == "gauss":
        return (x + torch.randn_like(x) * 0.03).clamp(0, 1), m
    if name == "blur":
        k = torch.tensor([1., 2., 1.], device=x.device)
        k = (k[:, None] * k[None]) / 16
        k = k.expand(3, 1, 3, 3).contiguous()
        return F.conv2d(x, k, padding=1, groups=3), m
    if name == "resize":
        y = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        return F.interpolate(y, size=(H, H), mode="bilinear", align_corners=False), m
    if name == "crop":
        c = int(H * 0.8); o = (H - c) // 2
        y = F.interpolate(x[:, :, o:o + c, o:o + c], size=(H, H), mode="bilinear", align_corners=False)
        mm = F.interpolate(m[:, :, o:o + c, o:o + c], size=(H, H), mode="nearest")
        return y, mm
    raise ValueError(name)


class Noise(nn.Module):
    """Differentiable training-time attack layer (straight-through for JPEG / rounding)."""
    def forward(self, x):
        a = random.choice(["none", "gauss", "blur", "resize", "quant8", "jpeg", "jpeg"])
        if a == "jpeg":
            y, _ = attack(x, None, f"jpeg{random.choice([50, 60, 75, 90])}")
            return x + (y - x).detach()
        if a == "gauss":
            return (x + torch.randn_like(x) * random.uniform(0.005, 0.04)).clamp(0, 1)
        y, _ = attack(x, None, a)
        return x + (y - x).detach() if a == "quant8" else y


# =============================== TRAINING ===============================
def step(enc, ext, noise, img, mask, w_img, p_keep=0.7):
    img, mask = img.to(dev), mask.to(dev)
    B = img.size(0)
    keep = (torch.rand(B, 1, 1, 1, device=dev) < p_keep).float()   # some samples are clean negatives
    mask_in = mask * keep
    msg = torch.randint(0, 2, (B, NBITS), device=dev).float()
    wm = enc(img, msg, mask_in)
    logits, loc = ext(noise(wm))
    l_img = ((wm - img) ** 2).sum() / (mask_in.sum() * 3 + 1)
    k = keep.view(B)
    bce = F.binary_cross_entropy_with_logits(logits, msg, reduction="none").mean(1)
    l_msg = (bce * k).sum() / (k.sum() + 1e-6)
    l_loc = F.binary_cross_entropy_with_logits(loc, mask_in)
    loss = w_img * l_img + W_MSG * l_msg + l_loc
    bit_acc = (((logits > 0).float() == msg).float().mean(1) * k).sum() / (k.sum() + 1e-6)
    return loss, dict(img=l_img.item(), msg=l_msg.item(), loc=l_loc.item(), bit_acc=bit_acc.item())


def train(enc, ext, tr, va, epochs, w_img, out, warmup=3):
    """Invisibility weight ramps 0 -> w_img over `warmup` epochs to avoid encoder collapse."""
    noise = Noise().to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(ext.parameters()), 1e-3)
    best, saved = 1e9, False
    path = os.path.join(out, "maskmark.pt")
    for ep in range(epochs):
        w_ep = w_img * min(1.0, ep / max(1, warmup))
        enc.train(); ext.train()
        for img, mask in tr:
            loss, st = step(enc, ext, noise, img, mask, w_ep)
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval(); ext.eval()
        vl = []
        with torch.no_grad():
            for img, mask in va:
                l, s_ = step(enc, ext, noise, img, mask, w_img)
                vl.append(l.item())
        v = float(np.mean(vl))
        print(f"epoch {ep+1}/{epochs} w_img={w_ep:.1f} train: " + " ".join(f"{k}={x:.4f}" for k, x in st.items())
              + f" | val_loss={v:.4f}")
        if ep >= warmup and v < best:
            best, saved = v, True
            torch.save({"enc": enc.state_dict(), "ext": ext.state_dict()}, path)
    if not saved:
        torch.save({"enc": enc.state_dict(), "ext": ext.state_dict()}, path)
    ck = torch.load(path, map_location=dev)
    enc.load_state_dict(ck["enc"]); ext.load_state_dict(ck["ext"])


# =============================== EVALUATION ===============================
def bin_metrics(y, s, thr=0.5):
    y, s = np.asarray(y), np.asarray(s)
    p = (s >= thr).astype(int)
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(accuracy=accuracy_score(y, p),
                precision=precision_score(y, p, zero_division=0),
                recall=recall_score(y, p, zero_division=0),
                specificity=tn / (tn + fp + 1e-9),
                f1=f1_score(y, p, zero_division=0),
                roc_auc=roc_auc_score(y, s) if len(set(y)) > 1 else float("nan"),
                pr_auc=average_precision_score(y, s) if len(set(y)) > 1 else float("nan"),
                confusion_matrix=cm.tolist())


@torch.no_grad()
def evaluate(enc, ext, loader, name, keep_raw=False):
    enc.eval(); ext.eval()
    bit_ok = bit_n = 0
    psnr, ssim, iou = [], [], []
    px_y, px_s, im_y, im_s = [], [], [], []
    for img, mask in loader:
        img, mask = img.to(dev), mask.to(dev)
        B, _, H, W = img.shape
        msg = torch.randint(0, 2, (B, NBITS), device=dev).float()
        wm = enc(img, msg, mask)
        wa, ma = attack(wm, mask, name)
        ca, _ = attack(img, mask, name)                     # clean (never watermarked) image, same attack
        lg, loc = ext(wa)
        _, loc0 = ext(ca)
        p, p0 = torch.sigmoid(loc), torch.sigmoid(loc0)
        bit_ok += ((lg > 0).float() == msg).float().sum().item(); bit_n += msg.numel()
        mse = ((wm - img) ** 2).flatten(1).mean(1)
        psnr += (10 * torch.log10(1 / (mse + 1e-10))).tolist()
        pred = (p > 0.5).float()
        inter = (pred * ma).flatten(1).sum(1); uni = ((pred + ma) > 0).float().flatten(1).sum(1)
        iou += (inter / (uni + 1e-9)).tolist()
        if name == "none":
            for a, b in zip(wm.cpu().numpy(), img.cpu().numpy()):
                ssim.append(structural_similarity(a.transpose(1, 2, 0), b.transpose(1, 2, 0),
                                                  channel_axis=2, data_range=1.0))
        # pixel-level samples (watermarked image -> mask, clean image -> zeros)
        n = 2000
        idx = torch.randint(0, H * W, (B, n), device=dev)
        px_s += [torch.gather(p.flatten(1), 1, idx).cpu().numpy().ravel(),
                 torch.gather(p0.flatten(1), 1, idx).cpu().numpy().ravel()]
        px_y += [torch.gather(ma.flatten(1), 1, idx).cpu().numpy().ravel(),
                 np.zeros(B * n)]
        # image-level score = mean of top 1% pixel probabilities
        k = max(1, int(0.01 * H * W))
        im_s += [p.flatten(1).topk(k, 1).values.mean(1).cpu().numpy(),
                 p0.flatten(1).topk(k, 1).values.mean(1).cpu().numpy()]
        im_y += [np.ones(B), np.zeros(B)]
    px_y, px_s = np.concatenate(px_y), np.concatenate(px_s)
    im_y, im_s = np.concatenate(im_y), np.concatenate(im_s)
    res = dict(attack=name, bit_accuracy=bit_ok / bit_n, localization_iou=float(np.mean(iou)),
               psnr=float(np.mean(psnr)), pixel_level=bin_metrics(px_y, px_s),
               image_level=bin_metrics(im_y, im_s))
    if ssim:
        res["ssim"] = float(np.mean(ssim))
    if keep_raw:
        res["_raw"] = (px_y, px_s, im_y, im_s)
    return res


# =============================== PLOTS ===============================
def plot_curves(raw, out):
    px_y, px_s, im_y, im_s = raw
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    for r, (y, s, t) in enumerate([(px_y, px_s, "Pixel-level"), (im_y, im_s, "Image-level")]):
        fpr, tpr, _ = roc_curve(y, s); pr, rc, _ = precision_recall_curve(y, s)
        ax[r, 0].plot(fpr, tpr); ax[r, 0].plot([0, 1], [0, 1], "--", c="gray")
        ax[r, 0].set(title=f"{t} ROC (AUC={roc_auc_score(y, s):.3f})", xlabel="FPR", ylabel="TPR")
        ax[r, 1].plot(rc, pr)
        ax[r, 1].set(title=f"{t} PR (AP={average_precision_score(y, s):.3f})", xlabel="Recall", ylabel="Precision")
    plt.tight_layout(); plt.savefig(os.path.join(out, "roc_pr.png"), dpi=150); plt.close()


def plot_confusion(res, out):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for a, k in zip(ax, ["pixel_level", "image_level"]):
        cm = np.array(res[k]["confusion_matrix"])
        a.imshow(cm, cmap="Blues"); a.set(title=f"{k} confusion", xticks=[0, 1], yticks=[0, 1],
                                          xticklabels=["clean", "marked"], yticklabels=["clean", "marked"],
                                          xlabel="Predicted", ylabel="True")
        for i in range(2):
            for j in range(2):
                a.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(os.path.join(out, "confusion.png"), dpi=150); plt.close()


def plot_robustness(allres, out):
    names = [r["attack"] for r in allres]
    fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(len(names)); w = 0.27
    ax.bar(x - w, [r["bit_accuracy"] for r in allres], w, label="Bit accuracy")
    ax.bar(x, [r["localization_iou"] for r in allres], w, label="Localization IoU")
    ax.bar(x + w, [r["image_level"]["roc_auc"] for r in allres], w, label="Image ROC-AUC")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30); ax.set_ylim(0, 1.05); ax.legend()
    ax.set_title("Robustness to attacks"); plt.tight_layout()
    plt.savefig(os.path.join(out, "robustness.png"), dpi=150); plt.close()


def saliency_examples(enc, ext, loader, out, n=4):
    enc.eval(); ext.eval()
    img, mask = next(iter(loader))
    img, mask = img[:n].to(dev), mask[:n].to(dev)
    msg = torch.randint(0, 2, (img.size(0), NBITS), device=dev).float()
    with torch.no_grad():
        wm = enc(img, msg, mask)
    x = wm.clone().requires_grad_(True)
    _, loc = ext(x)
    torch.sigmoid(loc).sum().backward()                       # gradient saliency
    sal = x.grad.abs().max(1)[0].cpu().numpy()
    prob = torch.sigmoid(ext(wm)[1]).detach().cpu().numpy()[:, 0]
    diff = ((wm - img).abs() * 10).clamp(0, 1)
    titles = ["Original", "Watermarked", "Residual x10", "True mask", "Predicted map", "Saliency"]
    fig, ax = plt.subplots(n, 6, figsize=(15, 2.5 * n))
    for i in range(n):
        ims = [img[i].permute(1, 2, 0).cpu(), wm[i].permute(1, 2, 0).cpu(), diff[i].permute(1, 2, 0).cpu(),
               mask[i, 0].cpu(), prob[i], sal[i] / (sal[i].max() + 1e-9)]
        for j, im in enumerate(ims):
            ax[i, j].imshow(im, cmap=None if j < 3 else "magma"); ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(titles[j])
    plt.tight_layout(); plt.savefig(os.path.join(out, "examples.png"), dpi=130); plt.close()


# =============================== MAIN ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images"); ap.add_argument("--masks")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--w_img", type=float, default=20.0, help="weight of invisibility loss")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--nbits", type=int, default=16)
    ap.add_argument("--w_msg", type=float, default=5.0, help="weight of message-bit loss")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    global NBITS, W_MSG
    NBITS, W_MSG = a.nbits, a.w_msg
    os.makedirs(a.out, exist_ok=True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    if a.synthetic:
        mk = lambda tr, n: Data(None, None, a.size, tr, synth=n)
        tr_ds, va_ds, te_ds = mk(True, 256), mk(False, 64), mk(False, 64)
    else:
        files = list_images(a.images)
        if not files:
            raise SystemExit(f"No images found in {a.images}")
        random.Random(0).shuffle(files)
        if a.max_images:
            files = files[:a.max_images]
        n = len(files); i1, i2 = int(0.8 * n), int(0.9 * n)
        masks = index_masks(a.masks)
        print(f"{n} images, {len(masks)} mask files indexed")
        tr_ds = Data(files[:i1], masks, a.size, True)
        va_ds = Data(files[i1:i2], masks, a.size, False)
        te_ds = Data(files[i2:], masks, a.size, False)

    tr = DataLoader(tr_ds, a.bs, shuffle=True, drop_last=True, num_workers=2)
    va = DataLoader(va_ds, a.bs, num_workers=2)
    te = DataLoader(te_ds, a.bs, num_workers=2)

    enc, ext = Encoder().to(dev), Extractor().to(dev)
    train(enc, ext, tr, va, a.epochs, a.w_img, a.out, a.warmup)

    allres = []
    for name in ["none", "quant8", "jpeg90", "jpeg75", "jpeg50", "blur", "gauss", "resize", "crop"]:
        r = evaluate(enc, ext, te, name, keep_raw=(name == "none"))
        raw = r.pop("_raw", None)
        if raw is not None:
            plot_curves(raw, a.out); plot_confusion(r, a.out)
        allres.append(r)
        print(f"[{name:7s}] bit_acc={r['bit_accuracy']:.3f} IoU={r['localization_iou']:.3f} "
              f"img_AUC={r['image_level']['roc_auc']:.3f} px_F1={r['pixel_level']['f1']:.3f} PSNR={r['psnr']:.1f}")
    plot_robustness(allres, a.out)
    saliency_examples(enc, ext, te, a.out)

    json.dump(allres, open(os.path.join(a.out, "results.json"), "w"), indent=2)
    print("\n=== CLEAN (no attack) FULL METRICS ===")
    c = allres[0]
    print(f"Bit acc {c['bit_accuracy']:.4f} | IoU {c['localization_iou']:.4f} | PSNR {c['psnr']:.2f} dB | SSIM {c.get('ssim', float('nan')):.4f}")
    for k in ["pixel_level", "image_level"]:
        m = c[k]
        print(f"{k}: acc={m['accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
              f"spec={m['specificity']:.4f} f1={m['f1']:.4f} roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f}")
    print(f"All outputs saved in: {a.out}/")


if __name__ == "__main__":
    main()
