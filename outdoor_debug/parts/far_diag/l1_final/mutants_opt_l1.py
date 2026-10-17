"""Mutation test of check_optimizer_lr.py (second review B1 + the reviewer's O1-O10): each mutation is applied to a COPY of the probe; the package's own optimizer-probe negative-control suite is re-run on that copy.
CAUGHT = a WRONG line / non-zero exit.  usage: ~/.venvs/ramen/bin/python mutants_opt_l1.py <scratch-outdir>"""
import shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
SRC = Path(__file__).resolve().parent; OUT = Path(sys.argv[1]).resolve(); OUT.mkdir(parents=True, exist_ok=True)
PROBE0 = (SRC / "check_optimizer_lr.py").read_text()
M = {
 "real": [],
 "O1 cfg compare removed": [("cfg == want, \"differs", "True, \"differs")],
 "O2 group sizes removed": [("[len(g[\"params\"]) for g in pg] == [188, 64, 2], str(", "True, str(")],
 "O3 initial_lr check removed": [("len(pg) == 3 and all(abs(g.get(\"initial_lr\", -1) / w - 1) < 1e-9 for g, w in zip(pg, want_init))", "True")],
 "O4 saved-lr vs closed form removed": [("len(pg) == 3 and all(abs(g[\"lr\"] - g[\"initial_lr\"] * sc) <= 1e-6 * max(g[\"initial_lr\"] * sc, 1e-30) + 1e-18 for g in pg)", "True")],
 "O4b saved-lr tol 1e-6 -> 0.5": [("<= 1e-6 * max(g[\"initial_lr\"] * sc", "<= 0.5 * max(g[\"initial_lr\"] * sc")],
 "O5 ratio check removed": [("len(pg) == 3 and pg[0][\"lr\"] > 0 and abs(pg[1][\"lr\"] / pg[0][\"lr\"] - head / 1e-4) < 1e-6 and abs(pg[2][\"lr\"] / pg[0][\"lr\"] - 10) < 1e-6, \"\")", "True, \"\")")],
 "O3+O5 initial_lr and ratio removed": [("len(pg) == 3 and all(abs(g.get(\"initial_lr\", -1) / w - 1) < 1e-9 for g, w in zip(pg, want_init))", "True"),
                                        ("len(pg) == 3 and pg[0][\"lr\"] > 0 and abs(pg[1][\"lr\"] / pg[0][\"lr\"] - head / 1e-4) < 1e-6 and abs(pg[2][\"lr\"] / pg[0][\"lr\"] - 10) < 1e-6, \"\")", "True, \"\")")],
 "O6 --expect-step ignored": [("chk(f\"step == {a.expect_step}\", step == a.expect_step", "chk(f\"step == {a.expect_step}\", True")],
 "O7 final-snapshot step check removed": [("chk(f\"final snapshot step == {a.K}\", step == a.K", "chk(f\"final snapshot step == {a.K}\", True")],
 "O8 0<step<=K removed": [("0 < step <= a.K, f\"got", "True, f\"got")],
 "O9 -F does not turn unreadable into FAIL": [("return 1 if a.F else 0", "return 0")],
 "O10 S5 head accepted at either 1e-4 or 1e-5": [("head = 1e-5 if a.arm == \"S5\" else 1e-4; want_init", "head = 1e-5 if a.arm == \"S5\" else 1e-4; want_init")],  # placeholder replaced below
}
M["O10 S5 head accepted at either 1e-4 or 1e-5"] = [("all(abs(g.get(\"initial_lr\", -1) / w - 1) < 1e-9 for g, w in zip(pg, want_init))", "all(abs(g.get(\"initial_lr\", -1) / w - 1) < 1e-9 or (w in (1e-4, 1e-5) and g.get(\"initial_lr\") in (1e-4, 1e-5)) for g, w in zip(pg, want_init))"),
    ("abs(pg[1][\"lr\"] / pg[0][\"lr\"] - head / 1e-4) < 1e-6", "(abs(pg[1][\"lr\"] / pg[0][\"lr\"] - 0.1) < 1e-6 or abs(pg[1][\"lr\"] / pg[0][\"lr\"] - 1) < 1e-6)")]
M.update({
 "B1a --smoke: substitutions not applied (want = unmodified arm yaml)": [("    want = yaml.safe_load(txt)", "    want = yaml.safe_load(open(cfgp))")],
 "B1b --smoke: max_steps substitution removed": [("txt, n1 = re.subn(rf\"^  max_steps: {K}$\", f\"  max_steps: {N}\", txt, flags=re.M)", "n1 = 1")],
 "B1c --smoke: checkpoint_dir substitution removed": [("txt, n2 = re.subn(rf\"^  checkpoint_dir: runs/l1_{a.arm}_k{K}$\", f\"  checkpoint_dir: runs/l1_smoke_{a.arm}_k{K}\", txt, flags=re.M)", "n2 = 1")],
 "B1d --smoke: checkpoint_at_steps substitution removed": [("txt, n3 = re.subn(rf\"^(  checkpoint_at_steps:\\n)  - {K}$\", rf\"\\g<1>  - {N}\", txt, flags=re.M)", "n3 = 1")],
 "B1e --smoke: substitution-count check removed": [("(n1, n2, n3) == (1, 1, 1)", "True")],
 "B1f --smoke: cfg compare removed in smoke mode": [("cfg == want, \"differs", "(cfg == want or bool(a.smoke)), \"differs")],
})

def one(item):
    name, edits = item
    base = OUT / name.split()[0]
    if base.exists(): shutil.rmtree(base)
    d = base / "parts/far_diag/l1_final"; d.mkdir(parents=True); shutil.copytree(SRC / "configs", d / "configs"); shutil.copy(SRC / "train_template_l1.sbatch", d); shutil.copy(SRC / "make_optfixtures_and_negctl_l1.py", d)
    (base / "parts/confirm/stage1/base").mkdir(parents=True); shutil.copy(SRC.parents[1] / "confirm/stage1/base/effective_ramen_n46.yaml", base / "parts/confirm/stage1/base/")
    t = PROBE0
    for o, n in edits:
        assert t.count(o) == 1, f"{name}: pattern found {t.count(o)}x: {o!r}"
        t = t.replace(o, n)
    (d / "check_optimizer_lr.py").write_text(t)
    r = subprocess.run([sys.executable, str(d / "make_optfixtures_and_negctl_l1.py")], capture_output=True, text=True, cwd=d)
    wrong = [l for l in r.stdout.splitlines() if "WRONG" in l]
    return name, (r.returncode != 0 or bool(wrong)), wrong
with ThreadPoolExecutor(6) as ex: res = list(ex.map(one, M.items()))
for name, caught, wrong in res: print(f"{'CAUGHT ' if caught else 'MISSED '} {name}  ({len(wrong)} WRONG)" + "".join("\n      " + w[:170] for w in wrong[:2]))
print("MISSED total:", sum(1 for r in res if not r[1]), "of", len(res))
