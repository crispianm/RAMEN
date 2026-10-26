# Provenance of commit labels

This repository's `main` is an orphan reproducibility history: it does not contain the development history. The paper, its tables and its figure captions cite earlier commits of the development repository (for example the scorer commit in `configs/paper_results.yaml` and the commit labels in table captions). Those commits are **not** in this repository's history; they are preserved in an archive.

## Where the archive is

| Item | Value |
|---|---|
| Archive | bundles below (not hosted in this repository; available from the authors on request) |
| Full mirror bundle of the previous `crispianm/RAMEN` (all refs) | `ramen_full_20261016.bundle`, sha256 `026380f72a4cb56c0ca3203716997086d1f9e24b17ac42ef4903f1557c79ab59` |
| Bundle of the local-only development branches | `ramen_local_only_20261016.bundle`, sha256 `9098a62f071b264d0bc74bcf1866f08e7f9d325d22bc040c90966c99ccb7a1f2` |

To make an old commit resolvable in a clone of this repository (needed for the scorer clone used by `generate_results.py`):

```bash
git fetch <path-to>/ramen_full_20261016.bundle 'refs/*:refs/archive/*'
git cat-file -t 368b5c53b37ebd22f5d44ce0c9d514122efc88da   # -> commit
```

## What this commit is

The files in this commit are the union of two development lineages, merged without conflicts: the evaluation lineage (`paper-repro-I2`, commit `a547c75d`, which contains `8c7560be`, `a1493364`, `1a67278b`, `3a4884cc`) and the training lineage (`paper-train-pf`, commit `bd35a811`, which contains `0699ee4f` and `c1abf81d`). They differ in `train.py`, `data/scannetpp_train.py` and the training configs only. Unchanged source files are byte-identical to those commits; the removal list and the few edited files are in the archive README.

## Commit labels

| Sha | Role |
|---|---|
| `8c7560be84b71ce19c0cf24efd79fc034c1826e9` | F'' (paper-repro-F3): R1 regeneration; cited in released labels (tab:static, tab:rep, tab:dynamic*, tab:fid, ...) |
| `a1493364f71415e570131c0b76370f06afeb781b` | H: tab:rep_fnoise label ("agg commit") |
| `8a99f9b3ead99b2d708777b23a7a8af1b689a960` | I |
| `3a4884cce837405589283081d33d222b3cc032ed` | I' (S0 cascades; the running C1 resume job's code pin) |
| `1a67278bc331be59404979f4ab2e3bcd1ca69655` | I3: S3 ladder / filters (cited in the tab:noise label) |
| `a547c75d6d0fb797d71bd703beef38843bf3626d` | I2: S2 (DAv2-S FT, DA-AC FT) |
| `4aa49471854034c4d00b7d2aad1346e78254d5aa` | Ifig |
| `0ae6a8a2cce75af17e0fe0292c220e729b5c6cda` | Ifig3 |
| `e5d00ebc128d52f00c0ff4b7028ee780d8668c99` | Ifig4 |
| `abef23dea7ea3838a3e024ae3e5368df220ba206` | Ifig5 |
| `cd657f3ff6a3e67ac4b342c15e2421deea7f180b` | Ifig6 |
| `5f2e7ea39bf9c39eb29f39a1cce50294b50357a4` | Ifig7 |
| `e0f7447850f759dcf136948d9aaf6b57952740e5` | Ifig8 |
| `d8ccc5d44e5fb470a4e5ff7f590bcb5cc839d46c` | Ifig9 |
| `d75ccc3f5d616d471abdde1b882e659fb923fba3` | Ifig10 |
| `2b9f076a811643ac3a12bb90b79e66592255db1c` | L (latency harness) |
| `2ce36e8d0e185f0ca4d4bec1adf6500062fcc26c` | X' (efficiency) |
| `edd1904a416448cfb0626ec61242848d732bb1ce` | X (efficiency) |
| `2483b2f84ee1e05acce9ce46179c3f932e5a1e44` | BR (bridge) |
| `082ad924f9bd496b182d3377a4c83fa0abe1247d` | F' |
| `622eb8de9c5d9553c55c47582bd24e3d37d905fc` | F |
| `bd35a811fcfa5acb02c46e745348cafbad1ec00e` | TRAIN configs commit (pf_dav2a / pf_daacm) |
| `0699ee4fac9fb3b71839c8d7ed0b62c2874dc5aa` | TRAIN parent |
| `b6b6cd132203cad2b23ba320b59111f7ec85d00f` | novelty base |
| `3e915ea5755818e7ef3171776779e517b20b1589` | novelty compare.py (acked) |
| `de9b9c1a4f68e996ef060fcad8cbfa1057e91155` | novelty postbundle parent |
| `616d62cca0a71f6aab73d43d65c2a79f71992a88` | novelty postbundle |
| `e66a9ad3fb8e11121b615670989d62715af57257` | novelty i01 |
| `aee5a5020b3d508deeb015b96c5cec410be24d55` | novelty i02 |
| `cbf4730e8b4e1ed3aa44e1a6c81640f692822263` | novelty i03 |
| `12e295e20864a3c9fdd4fef0ec2cafaed665da29` | novelty i04 |
| `fcc52a175db010a7937360f6d57dbafc8730feab` | novelty i05 |
| `7419347548b7e97b38bf77927067468fae9e2306` | novelty i06 |
| `2b435cd2e4fc61b3330b4d9ffab179dcdc9a6b50` | novelty i07 |
| `03b83dcc36a5e8b4c7b8ce178b06dfd13dffab87` | novelty i08 |
| `9f14d1e8fedc680f4fab3fb1067434c318ef17d8` | filter-bed driver (dn01_headline.py) |
| `368b5c53b37ebd22f5d44ce0c9d514122efc88da` | scorer commit (the eval clone on Isambard) |
| `0c6d07b` | bench_overrides.py source commit (vendored wrapper `repro/vendored/bench_overrides.py`) |
| `5e331841` | common ancestor of the training lineage and the evaluation lineage (merge base) |
| `c1abf81d` | last commit that changed the training code and added the headline continuation config `dn01_v3_bfix_C3_B` |
| `d1efc449` | tip of the previous `main` (August 2026 state, before the paper-repro work) |

The novelty-sweep commits (`b6b6cd13`..`03b83dcc`) are exploratory work on branches that are not part of this release; they live only in `ramen_local_only_20261016.bundle`.
