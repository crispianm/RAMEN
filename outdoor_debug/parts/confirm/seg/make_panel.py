"""12-frame stride panel: overlay the frozen masks (the scorer's MaskStore format, written by av2/build_av2_masks.py) on frames, one grid image.
Frames chosen by a fixed rule BEFORE any model prediction: indices round(linspace(0, n-1, 12)) over the sorted frame-id list ('<log>:<cam_ts>') of the registered frames tsv.
Overlay: certain-sky = red, any-sky-but-not-certain = blue; frames listed in the mask meta `dropped` (out of the PRIMARY P only; masks stay in the store) are tagged 'P-DROPPED'.
usage: make_panel.py --root <av2_val data dir> --frames av2_frames.tsv --masks sky_masks.npz --out panel.png"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "av2"))
import av2_dataset as A

N = 12


def pick(ids, n=N):
    ids = sorted(ids)
    k = min(n, len(ids))
    return [ids[i] for i in np.round(np.linspace(0, len(ids) - 1, k)).astype(int)]


def overlay(img, certain, any_sky):
    o = img.astype(np.float32).copy()
    only_any = any_sky & ~certain
    o[only_any] = o[only_any] * 0.5 + np.array([0, 80, 255], np.float32) * 0.5
    o[certain] = o[certain] * 0.4 + np.array([255, 0, 0], np.float32) * 0.6
    return o.clip(0, 255).astype(np.uint8)


def panel(frames, masks, dropped=(), cols=4, tile=(300, 400)):
    """frames: {id: uint8 HxWx3}; masks: {id: (certain, any) bool}; dropped: ids tagged P-DROPPED. Returns the uint8 grid."""
    ids = pick(frames)
    rows = (len(ids) + cols - 1) // cols
    g = Image.new("RGB", (cols * tile[0], rows * tile[1]))
    for i, fid in enumerate(ids):
        c, a = masks[fid]
        t = Image.fromarray(overlay(frames[fid], c, a)).resize(tile, Image.BILINEAR)
        ImageDraw.Draw(t).text((4, 4), f"{fid[-22:]} sky={100 * c.mean():.1f}%" + (" P-DROPPED" if fid in dropped else ""), fill=(255, 255, 0))
        g.paste(t, ((i % cols) * tile[0], (i // cols) * tile[1]))
    return np.asarray(g)


def main(argv=None):
    ap = argparse.ArgumentParser()
    for k in ("root", "frames", "masks", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args(argv)
    rows = A.read_frames_tsv(a.frames)
    by_id = {A.frame_id(r[0], r[1]): r for r in rows}
    sel = pick(by_id)
    store = A.MaskStore(a.masks)
    meta = json.load(open(Path(a.masks).with_suffix(".meta.json")))
    frames = {f: A.load_rgb(A.Log.image_path_of(a.root, by_id[f][0], by_id[f][1])) for f in sel}
    masks = {f: store.get(f) for f in sel}
    Image.fromarray(panel(frames, masks, set(meta["dropped"]))).save(a.out)
    print("panel", a.out, "frames", sel)


if __name__ == "__main__":
    main()
