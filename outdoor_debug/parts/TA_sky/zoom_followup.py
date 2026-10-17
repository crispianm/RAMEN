"""Full-resolution zoom of one fixed frame: RGB | seg-sky overlay (green; yellow = seg-sky with finite d<9000, red = saturated seg-sky). args: depth_root seg_root env traj frame out.png"""
import sys, json, zipfile, numpy as np, cv2
R, S, env, t, f, out = sys.argv[1:7]
d = np.squeeze(np.ascontiguousarray(cv2.imdecode(np.fromfile(f"{R}/{env}/Data_easy/{t}/depth_lcam_front/{f}", np.uint8), cv2.IMREAD_UNCHANGED)).view("<f4"), -1)
sg = cv2.imdecode(np.frombuffer(zipfile.ZipFile(f"{S}/{env}/Data_easy/seg_lcam_front.zip").read(f"{env}/Data_easy/{t}/seg_lcam_front/" + f.replace("_depth", "_seg")), np.uint8), cv2.IMREAD_UNCHANGED)
ids = json.load(open("/projects/b5dh/scratch_ddad_20261017/ta_audit/fu/code/sky_ids_74.json"))["envs"][env]["sky_ids"]
rgb = cv2.imread(f"{R}/{env}/Data_easy/{t}/image_lcam_front/" + f.replace("_depth", ""))
sk = np.isin(sg, ids); sat = ~np.isfinite(d) | (d >= 1e4)
o = rgb.astype(np.float32); 
for m, c in ((sat & ~sk, (255, 0, 255)), (sk & sat, (0, 0, 255)), (sk & ~sat & (d >= 9000), (0, 255, 0)), (sk & ~sat & (d < 9000), (0, 255, 255))): o[m] = 0.4 * o[m] + 0.6 * np.array(c, np.float32)
cv2.imwrite(out, np.hstack([rgb, o.astype(np.uint8)]))
print(env, f, "seg-sky", int(sk.sum()), "sat", int((sk & sat).sum()), "finite<9000", int((sk & ~sat & (d < 9000)).sum()))
