"""12-frame stride panel: overlay the frozen certain-sky mask on frames, one grid image.
Frames chosen by fixed rule BEFORE any model prediction: indices round(linspace(0, n-1, 12))
over the sorted frame-id list of the image list. Run by the parent on the registered frames only.
usage: make_panel.py --images list.txt --masks masks.npz --out panel.png
Dropped frames (absent from the mask file) are shown with a grey 'DROPPED' tint."""
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sky_mask import read_mask_file

N = 12


def pick(ids, n=N):
    ids = sorted(ids)
    k = min(n, len(ids))
    return [ids[i] for i in np.round(np.linspace(0, len(ids) - 1, k)).astype(int)]


def overlay(img, mask):
    o = img.astype(np.float32).copy()
    if mask is None:
        o = o * 0.4 + 128 * 0.6
    else:
        o[mask] = o[mask] * 0.4 + np.array([255, 0, 0], np.float32) * 0.6
    return o.clip(0, 255).astype(np.uint8)


def panel(frames, masks, cols=4, tile=(480, 270)):
    """frames: {id: uint8 HxWx3}; masks: {id: bool}. Returns uint8 grid."""
    ids = pick(frames)
    rows = (len(ids) + cols - 1) // cols
    g = Image.new("RGB", (cols * tile[0], rows * tile[1]))
    for i, fid in enumerate(ids):
        t = Image.fromarray(overlay(frames[fid], masks.get(fid))).resize(tile, Image.BILINEAR)
        d = ImageDraw.Draw(t)
        frac = masks[fid].mean() if fid in masks else 0
        d.text((4, 4), f"{fid} sky={100*frac:.1f}%" + ("" if fid in masks else " DROPPED"), fill=(255, 255, 0))
        g.paste(t, ((i % cols) * tile[0], (i // cols) * tile[1]))
    return np.asarray(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = {p.stem: p for p in map(Path, Path(a.images).read_text().split("\n")) if str(p).strip()}
    masks, meta = read_mask_file(a.masks)
    sel = pick(paths)
    frames = {f: np.asarray(Image.open(paths[f]).convert("RGB")) for f in sel}
    Image.fromarray(panel(frames, masks)).save(a.out)
    print("panel", a.out, "frames", sel)


if __name__ == "__main__":
    main()
