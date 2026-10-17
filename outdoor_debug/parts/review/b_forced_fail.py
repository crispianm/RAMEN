"""b: forced-fail harness for submit/check_first_run.sh items 1,2,3,5,6. Real logs/metrics fetched read-only into /tmp/rv_b (logs/, runs/). Each case mutates a COPY and runs the real checker.
usage: ~/.venvs/ramen/bin/python b_forced_fail.py"""
import os, re, json, shutil, subprocess, sys, tempfile
SRC="/tmp/rv_b"; S="/home/wg19671/.claude/jobs/a0d391cb/tmp/od/outdoor_debug/parts/submit/check_first_run.sh"
LOG={"A0":"outdoor_screen_A0_7504894.out","TS2":"outdoor_screen_TS2_7504898.out","TS2_H80":"outdoor_screen_TS2_H80_7504900.out","TS2_H300":"outdoor_screen_TS2_H300_7504902.out","TS2_VK655":"outdoor_screen_TS2_VK655_7504904.out","A0b":"outdoor_screen_A0b_7504897.out"}
def run(arm, mut_log=None, mut_metrics=None, metrics_from=None, label=""):
    d=tempfile.mkdtemp(prefix="rvb_"); os.makedirs(d+"/runs")
    for a in ("A0",arm):
        shutil.copytree(f"{SRC}/runs/outdoor_screen_{a}", f"{d}/runs/outdoor_screen_{a}", dirs_exist_ok=True)
    for a in LOG: shutil.copy(f"{SRC}/logs/{LOG[a]}", f"{d}/{LOG[a]}")
    if mut_log: 
        p=f"{d}/{LOG[arm]}"; t=open(p).read(); t2=mut_log(t); assert t2!=t, f"mutation was a no-op: {label}"; open(p,"w").write(t2)
    if mut_metrics:
        p=f"{d}/runs/outdoor_screen_{arm}/metrics.jsonl"; rows=[json.loads(l) for l in open(p)]; rows=mut_metrics(rows)
        open(p,"w").write("\n".join(json.dumps(r) for r in rows)+"\n")
    if metrics_from:
        shutil.copy(f"{SRC}/runs/outdoor_screen_{metrics_from}/metrics.jsonl", f"{d}/runs/outdoor_screen_{arm}/metrics.jsonl")
    specs=[f"A0={d}/{LOG['A0']}"] + ([f"{arm}={d}/{LOG[arm]}"] if arm!="A0" else [])
    r=subprocess.run(["bash",S,"-r",d+"/runs"]+specs,capture_output=True,text=True,env={**os.environ,"PYTHON":os.path.expanduser("~/.venvs/ramen/bin/python")})
    return r.returncode, r.stdout
def show(case, arm, item, expect, **kw):
    rc,out=run(arm,label=case,**kw)
    lines=[l for l in out.splitlines() if l.startswith(f"[{arm}] item {item}:") and not (arm=="A0" and False)]
    # for item 1 show only non-ok lines + the step of interest
    lines=[l[:210] for l in lines if (item!=1 or "ok" not in l.split(":")[1][:4] or case.startswith("1b") or case.startswith("1g"))] or [l[:210] for l in lines][:1]
    print(f"### {case}  (arm {arm}, item {item}; checker must say {expect})\n  exit={rc}  RESULT={[l for l in out.splitlines() if l.startswith('RESULT')][0]}")
    for l in lines: print("  "+l)
    print()
sub=lambda a,b,cnt=1: (lambda t: t.replace(a,b,cnt))
rx=lambda pat,rep: (lambda t: re.sub(pat,rep,t,count=1,flags=re.M))
print("######## BASELINE (unmutated, must be no FAIL)")
for arm in ("TS2_H80",): rc,out=run(arm); print(arm, "exit",rc, [l for l in out.splitlines() if l.startswith('RESULT')])
print("######## ITEM 1 (LR)")
show("1a wrong LR at step 12060 (6.00e-05 -> 7.00e-05, +17%)","TS2",1,"FAIL",mut_log=sub("step12060/14000 loss=","step12060/14000 loss=",1) and rx(r"(step12060/14000 loss=\S+ lr=)6\.00e-05",r"\g<1>7.00e-05"))
show("1b constant-LR schedule: step 12200 lr 9.93e-05 -> 1.00e-04 (no decay; should FAIL: S3 cosine vs S2 constant)","TS2",1,"FAIL",mut_log=rx(r"(step12200/14000 loss=\S+ lr=)9\.93e-05",r"\g<1>1.00e-04"))
show("1c 12200 lr 9.93e-05 -> 9.50e-05 (-4.3%)","TS2",1,"FAIL",mut_log=rx(r"(step12200/14000 loss=\S+ lr=)9\.93e-05",r"\g<1>9.50e-05"))
show("1d no-op restart bug: lr=0 at step 12300","TS2",1,"FAIL",mut_log=rx(r"(step12300/14000 loss=\S+ lr=)\S+",r"\g<1>0.00e+00"))
show("1e wrong LR at a step NOT in the table (12420: 9.32e-05 -> 5.00e-05)","TS2_H80",1,"FAIL",mut_log=rx(r"(step12420/14000 loss=\S+ lr=)9\.32e-05",r"\g<1>5.00e-05"))
show("1f ALL LR step lines deleted from the log (log truncated / never trained)","TS2",1,"FAIL-or-WAIT",mut_log=lambda t:"\n".join(l for l in t.splitlines() if not re.search(r"step\d+/\d+ loss=",l)))
print("######## ITEM 2 (resume + total)")
show("2a [resume] line missing","TS2",2,"FAIL",mut_log=lambda t:"\n".join(l for l in t.splitlines() if not l.startswith("[resume]")))
show("2b total steps printed 13000 (max_steps wrong)","TS2",2,"FAIL",mut_log=lambda t:t.replace("/14000 loss=","/13000 loss="))
show("2c resume at step=0 (fresh start, not the checkpoint)","TS2",2,"FAIL",mut_log=lambda t:t.replace("epoch=3 step=12000","epoch=0 step=0"))
show("2d resume from a DIFFERENT checkpoint path (c3_n46 -> c3_n45) at the same step","TS2",2,"FAIL",mut_log=lambda t:t.replace("dn01_c3rep_20261009_c3_n46/step12000.pt epoch","dn01_c3rep_20261009_c3_n45/step12000.pt epoch"))
show("2e no step lines AND resume line present (still starting)","TS2",2,"WAIT",mut_log=lambda t:"\n".join(l for l in t.splitlines() if not re.search(r"step\d+/\d+ loss=",l)))
print("######## ITEM 3 (hinge key in metrics.jsonl)")
def addkey(rows):
    for r in rows:
        if r.get("event")=="loss_terms" and 12000<r["step"]<=12050: r["sky_hinge_weighted"]=0.01
    return rows
def dropkey(rows):
    for r in rows: r.pop("sky_hinge_weighted",None)
    return rows
show("3a hinge key present in a NON-H arm (TS2)","TS2",3,"FAIL",mut_metrics=addkey)
show("3b hinge key MISSING in an H arm (H80)","TS2_H80",3,"FAIL",mut_metrics=dropkey)
def neg(rows):
    for r in rows:
        if r.get("event")=="loss_terms" and r["step"]==12030: r["sky_hinge_weighted"]=-0.5
    return rows
show("3c negative hinge","TS2_H80",3,"FAIL",mut_metrics=neg)
def nan(rows):
    for r in rows:
        if r.get("event")=="loss_terms" and r["step"]==12030: r["sky_hinge_weighted"]=float("nan")
    return rows
show("3d NaN hinge","TS2_H80",3,"FAIL",mut_metrics=nan)
def zero(rows):
    for r in rows:
        if r.get("event")=="loss_terms" and 12000<r["step"]<=12050: r["sky_hinge_weighted"]=0.0
    return rows
show("3e hinge exactly 0 all rows (inert hinge)","TS2_H80",3,"WARN-not-FAIL (documented)",mut_metrics=zero)
show("3f H80 arm given H300's metrics (wrong floor, key present, finite)","TS2_H80",3,"FAIL if floor checked",metrics_from="TS2_H300")
show("3g H300 arm given H80's metrics","TS2_H300",3,"FAIL if floor checked",metrics_from="TS2_H80")
def A0dummy(rows): return rows
show("3h metrics.jsonl from a DIFFERENT arm (A0 metrics under H80 name): key absent","TS2_H80",3,"FAIL",metrics_from="A0")
def late(rows):
    for r in rows:
        if r.get("event")=="loss_terms" and r["step"]>12050 : pass
    return [r for r in rows if not (r.get("event")=="loss_terms" and 12000<r["step"]<=12050 and r["step"]>12020)]
show("3i only 2 loss_terms rows in window","TS2_H80",3,"WAIT",mut_metrics=late)
print("######## ITEM 5 (non-finite)")
ins=lambda t: t.replace("e3 step12040/14000","[warn] non-finite loss at step 12035, skipping micro-batch\ne3 step12040/14000",1)
show("5a real-format warn line at step 12035","TS2",5,"FAIL",mut_log=ins)
ins2=lambda t: t.replace("e3 step12040/14000","[warn] non-finite loss at step 12035, skipping micro-batch[warn] non-finite loss at step 12036, skipping micro-batch\ne3 step12040/14000",1)
show("5b two warns interleaved on one line","TS2",5,"FAIL",mut_log=ins2)
ins3=lambda t: t.replace("e3 step12040/14000","[warn] non-finite loss at step 12400, skipping micro-batch\ne3 step12040/14000",1)
show("5c warn at step 12400 (outside first-50 window: documented as ok)","TS2",5,"ok(by design)",mut_log=ins3)
ins4=lambda t: t.replace("e3 step12040/14000","[warn] non-finite grad norm at step 12035\ne3 step12040/14000",1)
show("5d a DIFFERENT non-finite warn text ('non-finite grad') at 12035","TS2",5,"FAIL?",mut_log=ins4)
ins5=lambda t: t.replace("e3 step12040/14000","[warn] non-finite loss at step 12035, skipping micro-batch\ne3 step12040/14000",1)
show("5e warn present in A0 (reference arm)","A0",5,"FAIL",mut_log=lambda t:t.replace("e3 step12040/14000","[warn] non-finite loss at step 12035, skipping micro-batch\ne3 step12040/14000",1))
print("######## ITEM 6 (effective bound)")
show("6a vkitti bound 656 in a NON-VK655 arm (TS2)","TS2",6,"FAIL",mut_log=lambda t:t.replace("vkitti=655","vkitti=656",1))
show("6b vkitti bound 655 in the VK655 arm","TS2_VK655",6,"FAIL",mut_log=lambda t:t.replace("vkitti=656","vkitti=655",1))
show("6c bound line absent","TS2",6,"WAIT",mut_log=lambda t:"\n".join(l for l in t.splitlines() if not l.startswith("[data] effective")))
show("6d vkitti=6550 (prefix match?)","TS2",6,"FAIL",mut_log=lambda t:t.replace("vkitti=655,","vkitti=6550,",1))
show("6e tartanair bound changed 10000 -> 80 (other source wrong; item only looks at vkitti)","TS2",6,"FAIL-if-all-sources-checked",mut_log=lambda t:t.replace("tartanair=10000","tartanair=80",1))
show("6f vkitti bound 655 but TWO [data] effective lines, 1st right, 2nd wrong (rank mismatch)","TS2",6,"FAIL-if-all-lines",mut_log=lambda t:t.replace("[data] effective train depth","[data] effective train depth",1) and t.replace("\n[checkpoint]","\n[data] effective train depth upper bound per source (m, valid iff depth < bound): tartanair=10000, vkitti=656\n[checkpoint]",1))
