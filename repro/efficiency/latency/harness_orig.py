"""Runs the ORIGINAL bench_joint.py CLI (unchanged, pre-GPU-branch synchronize and all) from the pinned scorer clone, for the 3% harness cross-check. Usage: harness_orig.py latency --model ... --out ..."""
import os, sys
sys.dont_write_bytecode = True
CLONE = os.path.expanduser("~/.cache/ramen_scorer/368b5c53")
os.chdir(CLONE); sys.path.insert(0, CLONE)
import bench_joint as bj
sys.argv = ["bench_joint.py"] + sys.argv[1:]
bj.main()
