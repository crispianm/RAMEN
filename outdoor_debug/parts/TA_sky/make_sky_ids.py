"""Per-env sky ids from the official tartanairpy seg_label_map.json (74 envs). args: <segfiles_x_dir> <segfiles.zip> <out.json>"""
import sys, json, os, hashlib
src, zp, out = sys.argv[1:4]
SKYN = ("sky", "skysphere", "sky0", "cloud")
envs = sorted(e for e in os.listdir(src) if os.path.isfile(f"{src}/{e}/seg_label_map.json"))
res = {}
for e in envs:
    nm = json.load(open(f"{src}/{e}/seg_label_map.json"))["name_map"]
    sky = {k: v for k, v in nm.items() if k.lower() in SKYN}
    fuzzy = {k: v for k, v in nm.items() if ("sky" in k.lower() or "cloud" in k.lower() or "sun" == k.lower()) and k not in sky}
    res[e] = dict(sky_names=sky, sky_ids=sorted(set(sky.values())), has_sky_name=bool(sky), zero_is_sky_id=0 in sky.values(),
                  other_sky_like_names_not_used=fuzzy, n_classes=len(nm))
sha = hashlib.sha256(open(zp, "rb").read()).hexdigest()
json.dump(dict(tartanairpy_repo="https://github.com/castacks/tartanairpy", tartanairpy_commit="158a6844d782942110967325ca3082f50ab2bfc7",
               segfiles_zip="tartanairpy/segfiles.zip", segfiles_zip_sha256=sha, sky_name_rule="class name in {sky, skysphere, sky0, cloud} (case-insensitive)",
               n_envs=len(res), n_without_sky_name=sum(not v["has_sky_name"] for v in res.values()), envs=res), open(out, "w"), indent=1)
print(len(res), sha, [e for e, v in res.items() if not v["has_sky_name"]])
