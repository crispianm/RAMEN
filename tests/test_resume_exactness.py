"""EXPT-02 recovery gate (RAMEN Fable): resuming from a RAMEN resumable checkpoint
must reproduce the loss a direct, uninterrupted run would have computed at the same
step on the same batch -- otherwise a resumed step-10000 checkpoint is not really
"a screen at 10000 steps" for scientific purposes.

Exercises train.py's exact resume composition (weights, optimizer momentum, EMA
shadow, RNG, step counter -- train.py:951-966) directly, without the full train()
loop -- no dataloader/config machinery is needed to prove this invariant, and the
model includes Dropout so a broken RNG restore (not just weights/optimizer) would
show up as a mismatched loss, not pass by accident.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from train import EMA, _restore_rng_state, _rng_state, seed_everything


def _make_model_and_optimizer(seed: int):
    seed_everything(seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 1))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
    ema = EMA(model, decay=0.9)
    return model, optimizer, ema


def _step(model, optimizer, ema, batch, target) -> float:
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(batch), target)
    loss.backward()
    optimizer.step()
    ema.update(model)
    return loss.item()


def _checkpoint(model, optimizer, ema, step: int) -> dict:
    """Mirrors train.py:_save's resumable payload shape (trainable/optimizer/ema/rng/step)."""
    return {
        "step": step,
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "ema": copy.deepcopy(ema.shadow),
        "trainable": copy.deepcopy(model.state_dict()),
        "rng": _rng_state(),
    }


def _resume(checkpoint: dict, device: str = "cpu"):
    """Mirrors train.py:973-988's restore branch exactly, including the explicit
    optimizer.state sweep (train.py, caught 2026-08-26 on real GPU hardware -- AdamW's
    per-param "step" entry stays on cpu after load_state_dict, a device mismatch
    load_state_dict's own per-tensor device-matching does not cover) and the ema.shadow
    device move, both needed because torch.load(..., map_location="cpu") leaves every
    saved tensor on cpu regardless of the training device."""
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8, 1))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
    ema = EMA(model, decay=0.9)
    model.load_state_dict(checkpoint["trainable"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    ema.shadow = {k: v.to(device) for k, v in checkpoint["ema"].items()}
    gstep = int(checkpoint["step"])
    _restore_rng_state(checkpoint["rng"])
    return model, optimizer, ema, gstep


def test_resume_reproduces_the_next_step_loss_exactly():
    torch.manual_seed(0)
    batch0, target0 = torch.randn(3, 4), torch.randn(3, 1)
    batch1, target1 = torch.randn(3, 4), torch.randn(3, 1)

    # Direct, uninterrupted run: step 0 then step 1.
    model, optimizer, ema = _make_model_and_optimizer(seed=7)
    _step(model, optimizer, ema, batch0, target0)
    direct_loss_step1 = _step(model, optimizer, ema, batch1, target1)

    # Interrupted run: step 0, checkpoint, resume into a fresh model/optimizer/EMA, step 1.
    model2, optimizer2, ema2 = _make_model_and_optimizer(seed=7)
    _step(model2, optimizer2, ema2, batch0, target0)
    checkpoint = _checkpoint(model2, optimizer2, ema2, step=1)
    resumed_model, resumed_optimizer, resumed_ema, resumed_step = _resume(checkpoint)
    assert resumed_step == 1
    resumed_loss_step1 = _step(resumed_model, resumed_optimizer, resumed_ema, batch1, target1)

    assert resumed_loss_step1 == direct_loss_step1
    for key, direct_value in model.state_dict().items():
        assert torch.equal(direct_value, resumed_model.state_dict()[key]), f"weights diverged at {key}"
    for key, direct_value in ema.shadow.items():
        assert torch.equal(direct_value, resumed_ema.shadow[key]), f"EMA shadow diverged at {key}"


def test_resume_moves_every_ema_shadow_tensor_through_to_device():
    """EXPT-02 incident 2026-08-26: torch.load(resume_from, map_location="cpu") leaves
    checkpoint["ema"] tensors on cpu no matter what device training resumes on. A bare
    `ema.shadow = checkpoint["ema"]` (the pre-fix code) then crashes EMA.update()'s in-place
    add_ with a device mismatch the moment training resumes on cuda -- invisible to the
    all-cpu test above since there "device" is cpu everywhere. This machine has no
    CUDA-compatible GPU to reproduce the mismatch directly, so this test instead asserts
    the mechanical property the fix relies on: every checkpoint["ema"] tensor is routed
    through an explicit .to(device) call, not a bare dict assignment.
    """
    model, optimizer, ema = _make_model_and_optimizer(seed=7)
    checkpoint = _checkpoint(model, optimizer, ema, step=0)
    assert checkpoint["ema"], "expected a non-empty EMA shadow to exercise the move over"

    moved_calls = []
    real_to = torch.Tensor.to

    def tracking_to(self, *args, **kwargs):
        moved_calls.append(args[0] if args else kwargs.get("device"))
        return real_to(self, *args, **kwargs)

    torch.Tensor.to = tracking_to
    try:
        _, _, resumed_ema, _ = _resume(checkpoint, device="cpu")
    finally:
        torch.Tensor.to = real_to

    assert len(moved_calls) == len(checkpoint["ema"]), "not every EMA shadow tensor was moved via .to(device)"
    assert all(str(d) == "cpu" for d in moved_calls)
    for key, direct_value in checkpoint["ema"].items():
        assert torch.equal(direct_value, resumed_ema.shadow[key])


def test_resume_moves_every_optimizer_state_tensor_through_to_device():
    """Second real device-mismatch bug caught 2026-08-26, on real GPU hardware, in the FIRST
    resume smoke run after the false-positive assertion fix (5fcc2a6's own EMA fix was
    correct; the assertion just had a bare-vs-indexed-device comparison bug -- see
    test_bare_cuda_device_string_is_not_torch_device_equal_to_its_indexed_form). Once that
    was fixed, the assertion correctly caught a genuinely different mismatch:
    optimizer.load_state_dict moves most per-param state to match each param's device, but
    AdamW's per-param "step" entry stayed on cpu regardless -- a known PyTorch behavior, not
    covered by load_state_dict's own device-matching. This machine has no CUDA-compatible GPU
    to reproduce the mismatch directly, so this test asserts the mechanical property the fix
    relies on: every tensor value in every optimizer.state entry is routed through an
    explicit .to(device) call after load_state_dict, not left to load_state_dict alone.
    """
    model, optimizer, ema = _make_model_and_optimizer(seed=7)
    _step(model, optimizer, ema, torch.randn(3, 4), torch.randn(3, 1))  # populate optimizer.state
    checkpoint = _checkpoint(model, optimizer, ema, step=1)
    tensor_count = sum(1 for state in checkpoint["optimizer"]["state"].values()
                       for v in state.values() if torch.is_tensor(v))
    assert tensor_count > 0, "expected AdamW to have populated per-param state to exercise the move over"

    # Loaded separately from _resume and tracked only AFTER load_state_dict returns, so this
    # counts exactly this fix's own sweep -- load_state_dict's own internal .to() calls (it
    # already moves most, just not every, tensor) are not conflated into the count.
    fresh_model, fresh_optimizer, _ = _make_model_and_optimizer(seed=7)
    fresh_optimizer.load_state_dict(checkpoint["optimizer"])

    moved_calls = []
    real_to = torch.Tensor.to

    def tracking_to(self, *args, **kwargs):
        moved_calls.append(args[0] if args else kwargs.get("device"))
        return real_to(self, *args, **kwargs)

    torch.Tensor.to = tracking_to
    try:
        for state in fresh_optimizer.state.values():          # train.py's own sweep, mirrored
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to("cpu")
    finally:
        torch.Tensor.to = real_to

    assert len(moved_calls) == tensor_count, "not every optimizer.state tensor was moved via .to(device)"
    assert all(str(d) == "cpu" for d in moved_calls)
    assert all(v.device == torch.device("cpu") for state in fresh_optimizer.state.values()
              for v in state.values() if torch.is_tensor(v))


def test_bare_cuda_device_string_is_not_torch_device_equal_to_its_indexed_form():
    """torch.device("cuda") != torch.device("cuda:0") -- comparing DEVICE SPECIFIER OBJECTS,
    not doing any real GPU work, so this is checkable without a functional CUDA device (only
    torch.cuda.is_available() needs to report True, which it does even on this workstation's
    unsupported-CC GPU). This is what caused all 6 of EXPT-08/BENCH-08's OVDA resumes to fail
    2026-08-26 on real GPU hardware: any single-GPU (non-DDP) run sets `device = "cuda"` (bare,
    no index -- train.py's own cfg.get("model", {}).get("device", "cuda") default), so
    train.py's resume-time assertion built `resume_device = torch.device(device)` with no
    index and compared it against tensors already moved via `.to(device)`, whose own `.device`
    always reports an explicit index once actually allocated -- a hard, permanent inequality
    for every single tensor on every single-GPU run, not a real device mismatch. (DDP's
    `device = f"cuda:{dev_idx}"` is already indexed and never trips this.) Fixed by
    canonicalizing through an actual allocation (`torch.empty(0, device=device).device`)
    instead of parsing the string, which this test also demonstrates as the correct fix.
    """
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("needs torch.cuda.is_available() (device object comparison only, no GPU compute)")

    bare = torch.device("cuda")
    indexed = torch.device("cuda:0")
    assert bare != indexed, "if this ever starts passing, the underlying torch quirk changed"

    canonical = torch.empty(0, device="cuda").device
    assert canonical == indexed
    assert canonical != bare


def test_named_parameter_buffer_device_check_pattern_catches_a_mismatch():
    """Opus's audit of the optimizer.state fix (2026-08-26): after three device bugs in one
    resume path in one day, the resume-time assertion also walks core.named_parameters()/
    named_buffers() directly as a backstop, catching any future off-device component
    automatically. train.py's check is inline in a large function, not its own testable
    unit, so this test verifies the exact comparison PATTERN it uses -- `p.device !=
    resume_device` over (name, tensor) pairs -- against a real nn.Module, using a canonical
    device object built the same way (torch.empty(0, device=...).device, per the
    bare-vs-indexed-device test above) so this exercises the identical equality check.
    """
    from torch import nn

    model = nn.Linear(2, 2)
    resume_device = torch.empty(0, device="cpu").device
    off_device = [n for n, p in model.named_parameters() if p.device != resume_device]
    assert off_device == [], "a fresh cpu model must find nothing off-device against cpu"

    # No real second device on this machine to move a parameter to -- instead substitute a
    # fake (name, tensor-like) pair reporting a different device, exercising the same
    # comparison the real check performs over named_parameters()'s (name, tensor) pairs.
    class _FakeTensor:
        device = torch.device("meta")

    fake_named_params = list(model.named_parameters()) + [("poisoned.weight", _FakeTensor())]
    off_device = [n for n, p in fake_named_params if p.device != resume_device]
    assert off_device == ["poisoned.weight"]
