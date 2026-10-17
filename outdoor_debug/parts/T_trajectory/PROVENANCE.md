# Provenance of step6000/step8000 checkpoints (ft_n46, c3_n46)
Source (Isambard, read-only rsync pull, 2026-10-16 ~23:20Z):
  /projects/b5dh/RAMEN_train_dn01_c3rep_20261009/runs/dn01_c3rep_20261009_{ft_n46,c3_n46}/{step6000.pt,step8000.pt,metrics.jsonl}
Local: /mnt/DATA/DegradedDepth/checkpoints_v3/outdoor_debug_20261016/{ft_n46,c3_n46}/ (mtimes preserved by rsync -a: Oct 10)
Local SHA256SUMS (computed after pull, same dir):
  c3_n46/step6000.pt 79dcb968...  c3_n46/step8000.pt 0bcfc5e9...
  ft_n46/step6000.pt 8b673589...  ft_n46/step8000.pt 3a9b8c6d...
  (full hashes in the SHA256SUMS file; metrics.jsonl also listed)
Step12000 files are the pre-existing audited eval copies (checkpoints_v3/eval_20261012_ft_n46, eval_20261012_c3rep), unchanged.
GAP (stated honestly): NO remote-side sha256 was taken at pull time. Integrity evidence is rsync's size+mtime transfer check plus the local hashes, and the files load with cfg_check 0 differing keys. A remote `sha256sum` comparison has NOT been done; it needs one read-only ssh to clifton and I have not run it (asking Manager/Auditor before any Isambard contact).
