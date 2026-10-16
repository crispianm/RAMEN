"""Leg table of the efficiency (b) plan (R-grouping): stack increments at the same precision, one leg = one process. Pure data, no torch.

Toggles (B_efficiency_E1E3 section 0):  T1 hookoff, T2 fp16build (EXACT)  T3 sdpa, T4 sdpa_motion, T5 bnfold, T6 cudnn, T7 chlast, T8 dense_group (NEAR)
T9 c_enc + c_head, T10 c_rest (compile, NEAR).  T11 fast TAG V3 joins as S4 (agf backend, not a patches flag).
Stacks are cumulative:  S0 pinned | X = T1+T2 | S1 = X+T3+T4 | S2 = S1+T5..T8 | S3 = S2+T9+T10 | S4 = S3 + fast TAG (RAMEN only; only the TAG rows differ).
Metric-VDA-S gets only the toggles it has: T3, T4, T6, T9 (no X, no S4)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

TOGGLE_FLAGS = {"T1": ["hookoff"], "T2": ["fp16build"], "T3": ["sdpa"], "T4": ["sdpa_motion"], "T5": ["bnfold"], "T6": ["cudnn"], "T7": ["chlast"],
                "T8": ["dense_group"], "T9": ["c_enc", "c_head"], "T10": ["c_rest"]}
STACK_TOGGLES = {
    "ramen": {"S0": [], "X": ["T1", "T2"], "S1": ["T1", "T2", "T3", "T4"], "S2": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
              "S3": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]},
    "mvda": {"S0": [], "S1": ["T3", "T4"], "S2": ["T3", "T4", "T6"], "S3": ["T3", "T4", "T6", "T9"]},
}
STACK_TOGGLES["ramen"]["S4"] = STACK_TOGGLES["ramen"]["S3"]               # + fast TAG backend
# L (Addendum 16, user Add.33): BN-fold candidates, toggles IDENTICAL to X' legs.py (2ce36e8d): C = S1+T5, AZ = S1+T5+T6+T7+T9+T10. Metric-VDA-S matching stacks: C -> S1, AZ -> S3.
_S1 = ["T1", "T2", "T3", "T4"]
STACK_TOGGLES["ramen"].update({"C": _S1 + ["T5"], "AZ": _S1 + ["T5", "T6", "T7", "T9", "T10"]})
MVDA_MATCH = {"C": "S1", "AZ": "S3"}
INCREMENT_BASE = {"X": "S0", "S1": "X", "S2": "S1", "S3": "S2", "S4": "S3"}   # leg-vs-leg comparisons (R-grouping)
INCREMENT_BASE_MVDA = {"S1": "S0", "S2": "S1", "S3": "S2"}
BISECT_ORDER = {"E2": ["T8", "T7", "T6", "T5"], "E1": ["T4", "T3"], "E3": ["T10", "T9"]}    # remove toggles in reverse declared order inside a failing increment
MODEL_SPEC = {"ramen": "ours", "mvda": "metric_vda_s"}
MODEL_TAG_VARIANTS = {"ramen": ["none", "agf0.2t0.06", "agf0.5t0.12"], "mvda": ["none"]}
CLASS = {"S0": "BASE", "X": "EXACT"}


@dataclass
class Leg:
    id: str
    model: str            # ramen | mvda
    stack: str
    precision: str        # fp32 | fp16
    repeat: int = 0       # 0 original; 1 = noise-floor / benchmark repeat of the same leg
    toggles: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    agf_backend: str = "cpu"            # cpu = numpy reference TAG; fast_triton_v3 for S4
    alone: bool = False                 # T6 (cudnn.benchmark) legs run alone on the card (17.9 GiB peak, memory-dependent algorithm choice)
    tag_variants: list = field(default_factory=list)
    purpose: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def make_leg(model: str, stack: str, prec: str, repeat: int = 0, exclude: tuple = (), purpose: str = "") -> Leg:
    tg = [t for t in STACK_TOGGLES[model][stack] if t not in exclude]
    flags = [f for t in tg for f in TOGGLE_FLAGS[t]]
    return Leg(id=f"{model}-{stack}-{prec}" + (f"-rep{repeat}" if repeat else ""), model=model, stack=stack, precision=prec, repeat=repeat, toggles=tg, flags=flags,
               agf_backend="fast_triton_v3" if stack == "S4" else "cpu", alone="T6" in tg, tag_variants=MODEL_TAG_VARIANTS[model], purpose=purpose)


def build_legs(exclude: tuple = (), models=("ramen", "mvda")) -> list[Leg]:
    """The full leg list in launch order: first the legs that may share the card, then the T6 legs (alone)."""
    legs: list[Leg] = []
    if "ramen" in models:
        legs += [make_leg("ramen", "S0", "fp32", purpose="reference, released-number path"),
                 make_leg("ramen", "S0", "fp32", repeat=1, purpose="noise floor (S0 twice under the same N)"),
                 make_leg("ramen", "S0", "fp16", purpose="fp16 reference; (c) fp16-vs-fp32 re-score")]
        for s in ("X", "S1"):
            for p in ("fp32", "fp16"):
                legs.append(make_leg("ramen", s, p, exclude=exclude, purpose=f"increment {INCREMENT_BASE[s]} -> {s}"))
    if "mvda" in models:
        legs += [make_leg("mvda", "S0", "fp32", purpose="reference"), make_leg("mvda", "S0", "fp32", repeat=1, purpose="noise floor (same N)"),
                 make_leg("mvda", "S0", "fp16", purpose="fp16 reference")]
        for p in ("fp32", "fp16"):
            legs.append(make_leg("mvda", "S1", p, exclude=exclude, purpose="increment S0 -> S1"))
    if "ramen" in models:
        for s in ("S2", "S3", "S4"):
            for p in ("fp32", "fp16"):
                legs.append(make_leg("ramen", s, p, exclude=exclude, purpose=f"increment {INCREMENT_BASE[s]} -> {s}"))
        legs.append(make_leg("ramen", "S2", "fp32", repeat=1, exclude=exclude, purpose="R-benchmark: S2 (first stack with T6) fp32 repeat = its noise floor"))
    if "mvda" in models:
        for s in ("S2", "S3"):
            for p in ("fp32", "fp16"):
                legs.append(make_leg("mvda", s, p, exclude=exclude, purpose=f"increment {INCREMENT_BASE_MVDA[s]} -> {s}"))
        legs.append(make_leg("mvda", "S2", "fp32", repeat=1, exclude=exclude, purpose="R-benchmark: S2 fp32 repeat = its noise floor"))
    seen = set()
    for l in legs:
        assert l.id not in seen, l.id
        seen.add(l.id)
    return legs


def select_legs(legs: list[Leg], spec) -> list[Leg]:
    """Leg-subset filter (scheduling only): keep the legs whose id is listed (comma string or list), in the original launch order. Unknown id = error (nothing is silently dropped)."""
    if not spec:
        return legs
    want = [x for s in ([spec] if isinstance(spec, str) else spec) for x in s.split(",") if x]
    have = {l.id for l in legs}
    bad = [w for w in want if w not in have]
    if bad:
        raise SystemExit(f"--legs: unknown leg id(s) {bad}; known: {sorted(have)}")
    return [l for l in legs if l.id in want]


def comparisons(legs: list[Leg], primary: str = "S3") -> list[dict]:
    """Which leg pairs the compare step evaluates: increments at the same precision (+ bitwise for X vs S0), noise floors, C32/C16 for the PRIMARY."""
    ids = {l.id for l in legs}
    out = []
    for m in ("ramen", "mvda"):
        base = INCREMENT_BASE if m == "ramen" else INCREMENT_BASE_MVDA
        for s, b in base.items():
            for p in ("fp32", "fp16"):
                a_, b_ = f"{m}-{s}-{p}", f"{m}-{b}-{p}"
                if a_ in ids and b_ in ids:
                    out.append({"kind": "increment", "test": a_, "ref": b_, "gate": "bitwise" if (s, b) == ("X", "S0") else p})
        for s in ("S0", "S2"):
            a_, b_ = f"{m}-{s}-fp32-rep1", f"{m}-{s}-fp32"
            if a_ in ids and b_ in ids:
                out.append({"kind": "noise_floor", "test": a_, "ref": b_, "gate": "fp32"})
        if f"{m}-{primary}-fp32" in ids and f"{m}-S0-fp32" in ids:
            out.append({"kind": "C32", "test": f"{m}-{primary}-fp32", "ref": f"{m}-S0-fp32", "gate": "fp32"})
            out.append({"kind": "C16", "test": f"{m}-{primary}-fp16", "ref": f"{m}-S0-fp32", "gate": "fp16"})
        if f"{m}-S0-fp16" in ids:
            out.append({"kind": "fp16_vs_fp32_S0_(c)", "test": f"{m}-S0-fp16", "ref": f"{m}-S0-fp32", "gate": "descriptive"})
    return out


def estimate(legs: list[Leg], n_workers: int, min_per_leg: float = 42.0, cache_saving: float = 0.30, alone_penalty: float = 1.0) -> dict:
    """Planning estimate only (D_profile_and_route.md: 0.6-1 h per full leg sequential, cache removes ~25-40%; contention not modelled). NEVER reported as a timing."""
    per = min_per_leg * (1.0 - cache_saving)
    shared = [l for l in legs if not l.alone]
    alone = [l for l in legs if l.alone]
    wall_shared = -(-len(shared) // n_workers) * per                 # greedy waves of N equal legs
    wall_alone = len(alone) * per * alone_penalty
    return {"legs": len(legs), "shared_phase_legs": len(shared), "alone_phase_legs": len(alone), "N": n_workers, "est_min_per_leg_cached": round(per, 1),
            "est_wall_h": round((wall_shared + wall_alone) / 60, 2), "est_wall_shared_h": round(wall_shared / 60, 2), "est_wall_alone_h": round(wall_alone / 60, 2),
            "assumptions": f"{min_per_leg} min/leg sequential, cache saves {int(cache_saving * 100)}%, no contention model"}
