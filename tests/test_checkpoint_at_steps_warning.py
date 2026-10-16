from __future__ import annotations

from train import _warn_if_checkpoint_every_without_checkpoint_at_steps


def test_warns_when_checkpoint_every_set_without_checkpoint_at_steps(capsys):
    """val01_vda_s_relative.yaml hit exactly this: checkpoint_every_steps set,
    checkpoint_at_steps never set, nothing failed or warned, and the entire run's
    intermediate step{N}.pt curve was unrecoverable once noticed mid-run."""
    _warn_if_checkpoint_every_without_checkpoint_at_steps(
        {"training": {"checkpoint_every_steps": 500}}
    )
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "checkpoint_at_steps" in out


def test_no_warning_when_checkpoint_at_steps_is_set(capsys):
    _warn_if_checkpoint_every_without_checkpoint_at_steps(
        {"training": {"checkpoint_every_steps": 500, "checkpoint_at_steps": [2000, 4000]}}
    )
    assert capsys.readouterr().out == ""


def test_no_warning_when_checkpoint_every_steps_is_unset():
    _warn_if_checkpoint_every_without_checkpoint_at_steps({"training": {}})  # must not raise


def test_no_warning_when_checkpoint_every_steps_is_zero(capsys):
    _warn_if_checkpoint_every_without_checkpoint_at_steps(
        {"training": {"checkpoint_every_steps": 0}}
    )
    assert capsys.readouterr().out == ""
