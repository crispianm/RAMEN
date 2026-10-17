# R1 selection record (names/timestamps only; no payload opened)
Run on Isambard login node 2026-10-17 (~22:45Z) with the R1 scripts, AV2_REG_SHA = sha256 of AV2_REGISTRATION.md at the R1 commit = ec4f027854d6dfe0fbb652b335413c6769ab6e8d14c6567a18b900c0b3710a55 (R1 text in commit 09780e3).
- scripts on Isambard: download_av2_val.sh sha 5cd5ec3c9a920d9845a2f4edf81750c51c109552142afbb0f9a21c5fc8a48a0e; select_av2_frames.py sha acc562aaba4146af3a5f5be119d633dbf781f2e9491a8783169bf76a454126d8.
- result: logs=50, scored frames=792 (of 800 candidates), dropped for |dt|>25 ms = 8, objects=8920, total listed bytes 3052142575.
- /projects/b5dh/data/heldout/av2_val/meta_R1/av2_frames.tsv sha 1207c93b73bcf2967643445e1a27d479e415a581b7aa08bfcce14f26db604e59
- /projects/b5dh/data/heldout/av2_val/meta_R1/objects.tsv  sha 9d577f318630b1056b08b109c079b54859971d2cbcaa4a0e21f484465e909f64
- per-log scored frames: 1 logs x 9 frames; 1 logs x 15 frames; 48 logs x 16 frames; 
- pre-R1 partial fetch (1404 files, R1_fetched_prer1_keys.txt): 731 of them are in the R1 object list (to be re-verified by listed size + sha at `verify`), 673 are "fetched pre-R1, unused, never read" and excluded from the manifest; 8189 objects are still to fetch.
- NOT yet run: fetch (waits for the Auditor's check of R1). Pre-R1 meta/ (392-frame selection) is kept, unused.
