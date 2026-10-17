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
  * LiDAR exclusion (applied by av2/build_av2_masks.py): drop pixels within
    r_l = round(5 * short_side / 900) px (min 1; 9 px at 1550) of ANY projected LiDAR return in front of the camera.
  * dropped frames: a frame with < 1% of its pixels certain-sky (after erosion and LiDAR exclusion)
    is listed as dropped = out of the PRIMARY P only (R1 M4); its masks stay in the store.
  * Determinism: fp32, eval(), no augmentation, torch seed 0, no sampling.
  * Cityscapes OneFormer (shi-labs/oneformer_cityscapes_swin_large, task token 'semantic') sky
    score = the transformers post_process_semantic_segmentation score
    S[c,h,w] = sum_q softmax(class_queries_logits)[q,c] (null class dropped, [..., :-1])
               * sigmoid(masks_queries_logits)[q,h,w],
    NORMALISED here (declared) by sum_c S[c,h,w] so that per-pixel class scores sum to 1;
    sky prob = S[sky]/sum_c S, computed at the model output resolution, then upsampled bilinearly.
  * SegFormer sky prob = softmax over the 150 classes of the logits.
Mask files for AV2 are written ONLY by av2/build_av2_masks.py (the scorer's MaskStore format + sidecar meta). The older build_masks/write_mask_file path (which removed
dropped frames from the store, contradicting R1 M4) was DELETED.
"""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage

PROB_THRESH = 0.9
ERODE_REF_PX, ERODE_REF_SHORT = 8, 900
LIDAR_REF_PX = 5
DROP_FRAC = 0.01
ANY_P = 0.5            # `any` layer per segmenter: argmax is sky OR sky prob >= ANY_P (so certain, which needs p >= 0.9 in both, is a subset BY CONSTRUCTION)
IOU_MIN = 0.85         # pooled IoU of the two segmenters' `any` masks over all candidate frames; below = mask set REJECTED (stop, NOT CONFIRMED)
WORK_SHORT = 512
WORK_MULT = 32
SEED = 0
ADE_EXPECT = ("sky", 2)
CITY_EXPECT = ("sky", 10)
RULE = dict(prob_thresh=PROB_THRESH, erode_ref_px=ERODE_REF_PX, erode_ref_short=ERODE_REF_SHORT,
            lidar_ref_px=LIDAR_REF_PX, drop_frac=DROP_FRAC, any_p=ANY_P, iou_min=IOU_MIN, work_short=WORK_SHORT,
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
    uv: [N,2] float (x=col, y=row). A return outside the image still excludes the image pixels within `radius` of it (canvas padded by ceil(radius))."""
    h, w = shape
    if radius is None:
        radius = lidar_radius(min(h, w))
    pad = int(np.ceil(radius))
    uv = np.asarray(uv, float).reshape(-1, 2)
    uv = uv[np.isfinite(uv).all(1)]
    x = np.round(uv[:, 0]).astype(np.int64) + pad; y = np.round(uv[:, 1]).astype(np.int64) + pad
    ok = (x >= 0) & (x < w + 2 * pad) & (y >= 0) & (y < h + 2 * pad)
    if not ok.any():
        return np.zeros((h, w), bool)
    seed = np.ones((h + 2 * pad, w + 2 * pad), bool)
    seed[y[ok], x[ok]] = False
    return (ndimage.distance_transform_edt(seed) <= radius)[pad:pad + h, pad:pad + w]


def apply_lidar(mask, uv, radius=None):
    return mask & ~lidar_exclusion(mask.shape, uv, radius)


def any_sky(p, margin, any_p=ANY_P):
    """Per-segmenter `any` layer at IMAGE size. p: sky prob; margin: sky prob minus the largest other-class prob (both bilinearly upsampled).
    True where sky is the argmax (margin >= 0) or p >= any_p."""
    return (margin >= 0) | (p >= any_p)


def any_layer(pa, ma, pb, mb):
    """`any` = sky by EITHER segmenter (see any_sky). Returns bool [H,W]."""
    return any_sky(pa, ma) | any_sky(pb, mb)


def iou_counts(pa, ma, pb, mb):
    """(intersection, union) pixel counts of the two segmenters' own `any_sky` masks on one frame."""
    a, b = any_sky(pa, ma), any_sky(pb, mb)
    return int((a & b).sum()), int((a | b).sum())


def pooled_iou(counts):
    """counts: iterable of (inter, union) over ALL candidate frames (kept and dropped). Returns pooled IoU; nan if the union is empty."""
    i = sum(c[0] for c in counts); u = sum(c[1] for c in counts)
    return float("nan") if u == 0 else i / u          # empty union = UNEVALUABLE (check_agreement rejects it)


def check_agreement(counts, iou_min=IOU_MIN):
    """Raise MaskRejected if the pooled IoU is below iou_min (declared stop rule). Returns the IoU."""
    v = pooled_iou(counts)
    if not v == v:
        raise MaskRejected("pooled segmenter IoU unevaluable (no sky in either segmenter on any frame): mask set rejected (NOT CONFIRMED)")
    if v < iou_min:
        raise MaskRejected(f"pooled segmenter IoU {v:.4f} < {iou_min}: mask set rejected (NOT CONFIRMED, stop before any prediction)")
    return v


class MaskRejected(Exception):
    pass


def is_dropped(mask, drop_frac=DROP_FRAC):
    return mask.sum() < drop_frac * mask.size


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


def split_sky_margin(P, sky_id):
    """P [B,C,h,w] class probs -> (p_sky [B,1,h,w], margin [B,1,h,w] = p_sky - max over the other classes)."""
    import torch
    p = P[:, sky_id:sky_id + 1]
    other = torch.cat([P[:, :sky_id], P[:, sky_id + 1:]], 1).amax(1, keepdim=True)
    return p, p - other


class _Seg:
    def __init__(self, repo, revision, expect, kind, device="cpu", use_safetensors=None):
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
        kw = {} if use_safetensors is None else dict(use_safetensors=use_safetensors)     # D9: the verified pytorch_model.bin is loaded, never a model.safetensors
        self.model = M.from_pretrained(repo, revision=revision, torch_dtype=torch.float32, **kw).to(device).eval()

    def sky_maps(self, img):
        """img uint8 [H,W,3] -> (p, margin) float32 [H,W]: sky prob and (sky prob - max other-class prob), both upsampled bilinear to image size."""
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
                P = F.softmax(o.logits.float(), 1)                                   # [1,C,h/4,w/4]
            else:
                # the processor is used ONLY for the task-token ids ('the task is semantic');
                # pixel_values stay our declared working-res tensor (no processor resize).
                ti = self.processor(images=Image.fromarray(img).resize((ww, wh), Image.BILINEAR),
                                    task_inputs=["semantic"], return_tensors="pt")["task_inputs"].to(self.device)
                o = self.model(pixel_values=x, task_inputs=ti)
                P = oneformer_semantic_scores(o.class_queries_logits.float(), o.masks_queries_logits.float())
            p, margin = split_sky_margin(P, self.sky)
            p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
            margin = F.interpolate(margin, size=(H, W), mode="bilinear", align_corners=False)
        return p[0, 0].cpu().numpy().astype(np.float32), margin[0, 0].cpu().numpy().astype(np.float32)

    def sky_prob(self, img):
        return self.sky_maps(img)[0]


def main():
    raise SystemExit("make_sky_mask.py has no CLI any more: use outdoor_debug/parts/confirm/av2/build_av2_masks.py")


if __name__ == "__main__":
    main()
