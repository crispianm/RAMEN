import json, copy, subprocess, sys, tempfile, os
B="/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts"
CS=B+"/submit/score/check_scores.py"
base=json.load(open(B+"/scorer/regression_c3_n46_step12000_20261017T182513Z/scores.json"))
def run(d,args=()):
    p=tempfile.mktemp(suffix=".json",dir="/tmp"); json.dump(d,open(p,"w"))
    r=subprocess.run([sys.executable,"-I",CS,p,*args],capture_output=True,text=True); os.remove(p)
    return r.returncode,[l for l in r.stdout.splitlines() if l.startswith("FAIL")]+([r.stderr.strip().splitlines()[-1]] if r.stderr.strip() else [])
def mut(f):
    d=copy.deepcopy(base); f(d); return d
G=("--expect-step","12000","--expect-ckpt-sha",base["meta"]["ckpt"]["sha256"],"--expect-host-contains","4090")
cases=[
("baseline (4090 host, 12000)",lambda d:None,G),
("1 pins_asserted False",lambda d:d["meta"].__setitem__("pins_asserted",False),G),
("1b pins_asserted missing",lambda d:d["meta"].pop("pins_asserted"),G),
("2 scorer sha",lambda d:d["meta"].__setitem__("scorer_sha256","fbd2c999"+"0"*56),G),
("3 bj sha",lambda d:d["meta"].__setitem__("bench_joint_sha256","0"*64),G),
("3b bj pinned field only",lambda d:d["meta"].__setitem__("bench_joint_pinned","0"*64),G),
("4 combined sha",lambda d:d["meta"].__setitem__("combined_clip_sha256","0"*64),G),
("4b combined_pinned only",lambda d:d["meta"].__setitem__("combined_pinned","0"*64),G),
("5 weights_key trainable",lambda d:d["meta"]["ckpt"].__setitem__("weights_key","trainable"),G),
("6 step 14000 vs expect 12000",lambda d:d["meta"]["ckpt"].__setitem__("step",14000),G),
("6b step missing, no --expect-step",lambda d:d["meta"]["ckpt"].pop("step"),()),
("6c step 0 no --expect-step (accepted?)",lambda d:d["meta"]["ckpt"].__setitem__("step",1),()),
("7 ckpt sha wrong w/ expect",lambda d:d["meta"]["ckpt"].__setitem__("sha256","a"*64),G),
("7b ckpt sha garbage(64 chars) no expect",lambda d:d["meta"]["ckpt"].__setitem__("sha256","z"*64),()),
("8 host: no GH200 w/ expect GH200",lambda d:None,("--expect-host-contains","GH200")),
("8b host omitted flag, host=''",lambda d:d["meta"].__setitem__("host",""),()),
("8c gpu says A100 only, expect GH200 but hostname contains GH200",lambda d:(d["meta"].__setitem__("host","GH200-fake"),d["meta"].__setitem__("gpu","A100")),("--expect-host-contains","GH200")),
("9 nyuv2_frames 500",lambda d:d["meta"].__setitem__("nyuv2_frames",500),G),
("9b nyuv2_frames 160.0 float",lambda d:d["meta"].__setitem__("nyuv2_frames",160.0),G),
("10 precision fp32",lambda d:d["meta"].__setitem__("precision","fp32"),G),
("11 limit_clips 2",lambda d:d["meta"].__setitem__("limit_clips",2),G),
("11b max_frames 5",lambda d:d["meta"].__setitem__("max_frames",5),G),
("11c both keys deleted",lambda d:(d["meta"].pop("limit_clips"),d["meta"].pop("max_frames")),G),
("12 datasets missing vkitti",lambda d:d["meta"].__setitem__("datasets",["kitti","nyuv2","scannetpp"]),G),
("13 kitti n_clips 6",lambda d:d["guards"]["kitti"].__setitem__("n_clips",6),G),
("13b kitti per_video drop one",lambda d:d["guards"]["kitti"]["per_video"].pop(next(iter(d["guards"]["kitti"]["per_video"]))),G),
("13c kitti clips renamed (wrong keys, right count)",lambda d:d["guards"]["kitti"].__setitem__("per_video",{f"x{i}":v for i,v in enumerate(d["guards"]["kitti"]["per_video"].values())}),G),
("13d clip_lists.kitti.keys_sha256 wrong+keys swapped",lambda d:(d["meta"]["clip_lists"]["kitti"].__setitem__("keys_sha256","0"*64)),G),
("13e vkitti guards missing",lambda d:d["guards"].pop("vkitti"),G),
("14 nyuv2 abs_rel NaN",lambda d:d["guards"]["nyuv2"]["video_macro"].__setitem__("abs_rel_unaligned",float("nan")),G),
("14b nyuv2 abs_rel absurd 99",lambda d:d["guards"]["nyuv2"]["video_macro"].__setitem__("abs_rel_unaligned",99.0),G),
("15 kitti tae NaN",lambda d:d["guards"]["kitti"]["video_macro"].__setitem__("tae",float("nan")),G),
("15b scannetpp tae_gtvalid missing",lambda d:d["guards"]["scannetpp"]["video_macro"].pop("tae_gtvalid"),G),
("15c nyuv2 tae absent (not required)",lambda d:None,G),
("16 one kitti clip tae_gtvalid NaN",lambda d:d["guards"]["kitti"]["per_video"][next(iter(d["guards"]["kitti"]["per_video"]))].__setitem__("tae_gtvalid",float("nan")),G),
("16b macro tae_gtvalid altered inconsistent w/ per_video",lambda d:d["guards"]["kitti"]["video_macro"].__setitem__("tae_gtvalid",123.0),G),
("17 primary legacy missing",lambda d:d["primary"]["vkitti_far_80_655"].pop("legacy_subsample_median"),G),
("17b legacy NaN",lambda d:d["primary"]["vkitti_far_80_655"].__setitem__("legacy_subsample_median",float("nan")),G),
("17c legacy_subsample_n 5 (not 1e6)",lambda d:d["primary"]["vkitti_far_80_655"].__setitem__("legacy_subsample_n",5),G),
("17d legacy=-3 (nonsense but finite)",lambda d:d["primary"]["vkitti_far_80_655"].__setitem__("legacy_subsample_median",-3.0),G),
("17e mask string says 40<=gt (bin changed)",lambda d:d["primary"]["vkitti_far_80_655"].__setitem__("mask","40.0<=gt<655.0"),G),
("17f primary.n tiny / sentinel included? n=1",lambda d:d["primary"]["vkitti_far_80_655"].__setitem__("n",1),G),
("18 ramen_commit dirty",lambda d:None,G),
("19 scores.json is empty dict",lambda d:(d.clear()),G),
]
for name,f,args in cases:
    rc,fails=run(mut(f),args)
    print(f"[{name}] rc={rc}"); [print("    "+x) for x in fails]
