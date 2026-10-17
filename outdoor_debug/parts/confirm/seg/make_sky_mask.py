"""Frozen 'certain-sky' mask builder for the strict confirmatory run (Track 1).

DECLARED RULE (thresholds fixed here, BEFORE any use on real data):
  * two segmenters must AGREE: SegFormer ADE20K (class 'sky', id 2) and a Cityscapes
    OneFormer semantic model (class 'sky', trainId 10). Ids are asserted BY NAME from the
    model config id2label (assert_sky_id), never trusted by number alone.
  * working resolution: short side 512, long side rounded to a multiple of 32 (aspect kept),
    image resized bilinear (PIL) for both models; ImageNet mean/std normalisation.
  * sky PROBABILITY maps (float32, working res) are upsampled BILINEARLY (align_corners=False)
    to the original image size; hard labels are never resized. Then threshold >= 0.9 in BOTH.
  * boundary band: a pixel is kept only if its Euclidean distance to the nearest
    non-certain pixel is > r, r = round(8 * short_side / 900) px, min 1 (8 px at 1600x900 =
    the proposal's rule). Image borders are NOT treated as non-sky.
  * LiDAR exclusion (applied by the scorer, function provided): drop pixels within
    r_l = round(5 * short_side / 900) px (min 1) of any projected LiDAR return.
  * dropped frames: a frame with < 1% of its pixels certain-sky (after erosion and, if
    given, LiDAR exclusion) is DROPPED and counted.
  * Determinism: fp32, eval(), no augmentation, torch seed 0, no sampling.
  * Cityscapes OneFormer (shi-labs/oneformer_cityscapes_swin_large, task token 'semantic') sky
    score = the transformers post_process_semantic_segmentation score
    S[c,h,w] = sum_q softmax(class_queries_logits)[q,c] (null class dropped, [..., :-1])
               * sigmoid(masks_queries_logits)[q,h,w],
    NORMALISED here (declared) by sum_c S[c,h,w] so that per-pixel class scores sum to 1;
    sky prob = S[sky]/sum_c S, computed at the model output resolution, then upsampled bilinearly.
  * SegFormer sky prob = softmax over the 150 classes of the logits.
File format: ONE .npz (uncompressed, keys sorted): 'm|<frame_id>' packed bits (np.packbits of the
HxW bool, row-major), 's|<frame_id>' int32 [H,W], '__meta__' uint8 json (rule constants, model
pins, dropped list). The sha256 of the file is written next to it (<file>.sha256).
"""
import argparse, hashlib, io, json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

PROB_THRESH = 0.9
ERODE_REF_PX, ERODE_REF_SHORT = 8, 900
LIDAR_REF_PX = 5
DROP_FRAC = 0.01
WORK_SHORT = 512
WORK_MULT = 32
SEED = 0
ADE_EXPECT = ("sky", 2)
CITY_EXPECT = ("sky", 10)
RULE = dict(prob_thresh=PROB_THRESH, erode_ref_px=ERODE_REF_PX, erode_ref_short=ERODE_REF_SHORT,
            lidar_ref_px=LIDAR_REF_PX, drop_frac=DROP_FRAC, work_short=WORK_SHORT,
            work_mult=WORK_MULT, seed=SEED, ade_sky=list(ADE_EXPECT), city_sky=list(CITY_EXPECT))


# ---------------------------------------------------------------- id2label by name
def assert_sky_id(id2label, name, expected_id):
    """Return the class id whose label is `name`; it must be unique AND equal expected_id."""
    norm = {int(k): str(v).strip().lower() for k, v in id2label.items()}
    hits = [k for k, v in norm.items() if v == name]
    if len(hits) != 1:
        raise AssertionError(f"id2label: expected exactly one class named {name!r}, found {hits}")
    if hits[0] != expected_id:
        raise AssertionError(f"id2label: class {name!r} has id {hits[0]}, expected {expected_id}")
    return hits[0]


# ---------------------------------------------------------------- pure mask logic
def erosion_radius(short_side):
    return max(1, int(round(ERODE_REF_PX * short_side / ERODE_REF_SHORT)))


def lidar_radius(short_side):
    return max(1, int(round(LIDAR_REF_PX * short_side / ERODE_REF_SHORT)))


def certain_sky(p_a, p_b, thresh=PROB_THRESH, radius=None, require_both=True):
    """p_a, p_b: float32 [H,W] sky probabilities at IMAGE size. Returns bool [H,W].
    `radius`: erosion radius px (None -> declared rule from the image short side; 0 -> none)."""
    h, w = p_a.shape
    if radius is None:
        radius = erosion_radius(min(h, w))
    m = (p_a >= thresh) & (p_b >= thresh) if require_both else (p_a >= thresh)
    if radius > 0 and m.any():
        pad = np.pad(m, 1, mode="edge")   # border is not a boundary
        d = ndimage.distance_transform_edt(pad)[1:-1, 1:-1]
        m = m & (d > radius)
    return m


def lidar_exclusion(shape, uv, radius=None):
    """Bool [H,W] True where a projected LiDAR return is within `radius` px (Euclidean).
    uv: [N,2] float (x=col, y=row); out-of-image points are ignored."""
    h, w = shape
    if radius is None:
        radius = lidar_radius(min(h, w))
    seed = np.ones((h, w), bool)
    uv = np.asarray(uv, float).reshape(-1, 2)
    x = np.round(uv[:, 0]).astype(int); y = np.round(uv[:, 1]).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if not ok.any():
        return np.zeros((h, w), bool)
    seed[y[ok], x[ok]] = False
    return ndimage.distance_transform_edt(seed) <= radius


def apply_lidar(mask, uv, radius=None):
    return mask & ~lidar_exclusion(mask.shape, uv, radius)


def is_dropped(mask, drop_frac=DROP_FRAC):
    return mask.sum() < drop_frac * mask.size


def build_masks(prob_pairs, lidar=None, **kw):
    """prob_pairs: {frame_id: (p_a, p_b)} image-size maps. lidar: {frame_id: uv} optional.
    Returns (masks{id: bool}, dropped[list of ids]). Dropped frames are not in masks."""
    masks, dropped = {}, []
    for fid in sorted(prob_pairs):
        pa, pb = prob_pairs[fid]
        m = certain_sky(pa, pb, **kw)
        if lidar is not None and fid in lidar:
            m = apply_lidar(m, lidar[fid])
        if is_dropped(m):
            dropped.append(fid)
        else:
            masks[fid] = m
    return masks, dropped


# ---------------------------------------------------------------- file format
def write_mask_file(path, masks, dropped, pins=None):
    meta = dict(rule=RULE, pins=pins or {}, dropped=sorted(dropped), n_kept=len(masks))
    arrs = {}
    for fid in sorted(masks):
        m = masks[fid]
        arrs["m|" + fid] = np.packbits(m.reshape(-1))
        arrs["s|" + fid] = np.array(m.shape, np.int32)
    arrs["__meta__"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), np.uint8)
    path = Path(path)
    np.savez(path, **dict(sorted(arrs.items())))
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".sha256").write_text(f"{h}  {path.name}\n")
    return h


def read_mask_file(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["__meta__"]).decode())
    masks = {}
    for k in z.files:
        if k.startswith("m|"):
            fid = k[2:]; h, w = (int(v) for v in z["s|" + fid])
            masks[fid] = np.unpackbits(z[k])[: h * w].reshape(h, w).astype(bool)
    return masks, meta


# ---------------------------------------------------------------- model wrappers (lazy)
def work_size(h, w):
    s = WORK_SHORT / min(h, w)
    r = lambda v: max(WORK_MULT, int(round(v * s / WORK_MULT)) * WORK_MULT)
    return r(h), r(w)


def oneformer_semantic_scores(class_logits, mask_logits):
    """transformers post_process_semantic_segmentation score, normalised.
    class_logits [B,Q,C+1] (last = null class), mask_logits [B,Q,h,w] -> [B,C,h,w], sums to 1 over C."""
    import torch
    cl = class_logits.softmax(dim=-1)[..., :-1]
    mk = mask_logits.sigmoid()
    seg = torch.einsum("bqc,bqhw->bchw", cl, mk)
    return seg / seg.sum(1, keepdim=True).clamp_min(1e-6)


def oneformer_sky_prob(class_logits, mask_logits, sky_id):
    return oneformer_semantic_scores(class_logits, mask_logits)[:, sky_id:sky_id + 1]


class _Seg:
    def __init__(self, repo, revision, expect, kind, device="cpu"):
        import torch
        from transformers import AutoConfig
        self.torch = torch
        torch.manual_seed(SEED)
        cfg = AutoConfig.from_pretrained(repo, revision=revision)
        self.sky = assert_sky_id(cfg.id2label, *expect)
        self.kind, self.device = kind, device
        if kind == "segformer":
            from transformers import SegformerForSemanticSegmentation as M
        else:
            from transformers import OneFormerForUniversalSegmentation as M, OneFormerProcessor
            self.processor = OneFormerProcessor.from_pretrained(repo, revision=revision)
        self.model = M.from_pretrained(repo, revision=revision, torch_dtype=torch.float32).to(device).eval()

    def sky_prob(self, img):
        """img uint8 [H,W,3] -> float32 [H,W] sky prob upsampled bilinear to image size."""
        torch = self.torch
        from PIL import Image
        import torch.nn.functional as F
        H, W = img.shape[:2]
        wh, ww = work_size(H, W)
        x = np.asarray(Image.fromarray(img).resize((ww, wh), Image.BILINEAR), np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.kind == "segformer":
                o = self.model(pixel_values=x)
                p = F.softmax(o.logits.float(), 1)[:, self.sky:self.sky + 1]      # [1,1,h/4,w/4]
            else:
                # the processor is used ONLY for the task-token ids ('the task is semantic');
                # pixel_values stay our declared working-res tensor (no processor resize).
                ti = self.processor(images=Image.fromarray(img).resize((ww, wh), Image.BILINEAR),
                                    task_inputs=["semantic"], return_tensors="pt")["task_inputs"].to(self.device)
                o = self.model(pixel_values=x, task_inputs=ti)
                p = oneformer_sky_prob(o.class_queries_logits.float(), o.masks_queries_logits.float(), self.sky)
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
        return p[0, 0].cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="text file, one image path per line (frame id = file stem)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ade-repo", default="nvidia/segformer-b5-finetuned-ade-640-640")
    ap.add_argument("--city-repo", default="shi-labs/oneformer_cityscapes_swin_large")
    ap.add_argument("--ade-rev", required=True, help="pinned HF commit hash")
    ap.add_argument("--city-rev", required=True, help="pinned HF commit hash")
    ap.add_argument("--lidar-npz", default=None, help="optional npz: key=frame id, value [N,2] uv")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from PIL import Image
    paths = [Path(p) for p in Path(a.images).read_text().split("\n") if p.strip()]
    ids = [p.stem for p in paths]
    assert len(set(ids)) == len(ids), "frame ids (file stems) must be unique"
    A = _Seg(a.ade_repo, a.ade_rev, ADE_EXPECT, "segformer", a.device)
    B = _Seg(a.city_repo, a.city_rev, CITY_EXPECT, "oneformer", a.device)
    lid = dict(np.load(a.lidar_npz)) if a.lidar_npz else None
    pairs = {}
    for p, fid in zip(paths, ids):
        img = np.asarray(Image.open(p).convert("RGB"))
        pairs[fid] = (A.sky_prob(img), B.sky_prob(img))
    masks, dropped = build_masks(pairs, lid)
    pins = dict(ade=[a.ade_repo, a.ade_rev], city=[a.city_repo, a.city_rev])
    h = write_mask_file(a.out, masks, dropped, pins)
    print(f"kept {len(masks)} dropped {len(dropped)} sha256 {h}")


if __name__ == "__main__":
    main()
