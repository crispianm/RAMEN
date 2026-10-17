# TartanAir official-tooling task (queued behind E2_by_source; Auditor Addendum O5, light track)
Requirements: parity test on sampled frames from several envs incl. Downtown, GreatMarsh and an env with the 65504 sky value (e.g. HongKong); seg download sha256-pinned; per-env table: (a) sky class id, (b) stored sky depth range, (c) % of sky caught by d>1e4 rule vs by seg; plus trees-as-sky (#46) and GLASS_PROBLEM_ENVS.
Labelling rule: Downtown, Hospital, ForestEnv, OldBrickHouseNight are TRAINING envs; any TartanAir number on them (incl. outdoor-debug DEV) is labelled "training environments, not held out".
TS v2 = new recipe with its own regression test (M-03 8000 m dome -> sky=1, mask=0; B2-03 10000.0 edge). Released loader c1abf81 unchanged. Per-env counts may sharpen the paper O4 sentence only after Auditor verification.
Isambard 74-env seg pull: send job spec to Manager first; no ssh before go.
