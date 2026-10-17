"""Prints, for every source mutant of l1_readout.py in test_l1_readout.py (verdict/G0 mutants and S5 integrity mutants), the FIRST observed failing expectation. usage: ~/.venvs/ramen/bin/python mutants_readout_l1.py"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_l1_readout as T
miss = 0
for name, (old, new) in sorted(T.SRC_MUTANTS.items()):
    ns = T._load_mutant(old, new)
    bad = T.table2_mismatches(ns["verdict"]) or T.table_mismatches(ns["verdict"])
    why = f"case {bad[0]}" if bad else ("G0 boundary 0.93/0.75 fails" if not T._g0_boundary_ok(ns) else None)
    print(f"{'CAUGHT ' if why else 'MISSED '} {name}: {why}"); miss += why is None
for name, (old, new) in sorted(T.S5_MUTANTS.items()):
    ns = T._load_mutant(old, new)
    with tempfile.TemporaryDirectory() as d: held = T._s5_expectations_hold(ns, Path(d))
    print(f"{'CAUGHT ' if not held else 'MISSED '} S5 {name}: CLI expectations {'violated' if not held else 'all hold'}"); miss += held
print("MISSED total:", miss); sys.exit(1 if miss else 0)
