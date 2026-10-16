"""train.py's _assert_and_log_git_state, called as train()'s first statement (Fable's
ruling, VAL-01, 2026-08-27, twice-amended after a real failure and a real loophole, both
found by actually running this): never trust submission-time commit == start-time commit
for a job that may queue for hours. Enforcement is keyed on ``under_slurm``
(SLURM_JOB_ID present), not merely ``expected_commit`` being set -- a submitter who
forgets VAL01_EXPECTED_COMMIT under Slurm must get a hard failure, not a silent bypass.
Mocks subprocess.check_output throughout so these tests never touch this repo's own real
git state (in particular, never simulate a real dirty working tree in a shared session's
actual checkout)."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from train import _assert_and_log_git_state

_HEAD = "d496a5baec34787ad0e46503fedf0a40167fcb63"


def _mock_check_output(rev_parse_output=_HEAD, porcelain_output="", rev_parse_raises=False):
    def _fake(cmd, text=True):
        if rev_parse_raises:
            raise subprocess.CalledProcessError(128, cmd)
        if "rev-parse" in cmd:
            return rev_parse_output + "\n"
        if "status" in cmd:
            return porcelain_output
        raise AssertionError(f"unexpected command {cmd}")
    return _fake


def test_not_under_slurm_logs_and_continues_regardless_of_git_state(capsys):
    """crispwork's own deployment is deliberately not a git checkout, and local dev runs
    aren't under Slurm either -- neither should ever be blocked by this check."""
    with patch("subprocess.check_output", side_effect=_mock_check_output()):
        _assert_and_log_git_state(None, under_slurm=False)   # must not raise
    out = capsys.readouterr().out
    assert "verification skipped" in out
    assert "not under Slurm" in out


def test_under_slurm_without_expected_commit_raises():
    """The loophole this closes: a submitter who forgot to set VAL01_EXPECTED_COMMIT
    under real Slurm must get a hard failure, not the permissive not-under-Slurm branch."""
    with patch("subprocess.check_output", side_effect=_mock_check_output()):
        with pytest.raises(RuntimeError, match="VAL01_EXPECTED_COMMIT"):
            _assert_and_log_git_state(None, under_slurm=True)


def test_under_slurm_matching_commit_and_clean_tree_does_not_raise():
    with patch("subprocess.check_output", side_effect=_mock_check_output(porcelain_output="")):
        _assert_and_log_git_state(_HEAD, under_slurm=True)   # must not raise


def test_under_slurm_dirty_tree_raises_even_when_commit_matches():
    with patch("subprocess.check_output", side_effect=_mock_check_output(porcelain_output=" M train.py\n")):
        with pytest.raises(RuntimeError, match="dirty"):
            _assert_and_log_git_state(_HEAD, under_slurm=True)


def test_under_slurm_commit_mismatch_raises():
    with patch("subprocess.check_output", side_effect=_mock_check_output(rev_parse_output="0" * 40)):
        with pytest.raises(RuntimeError, match="VAL01_EXPECTED_COMMIT"):
            _assert_and_log_git_state(_HEAD, under_slurm=True)


def test_under_slurm_not_a_git_checkout_raises():
    """Under Slurm, "not a git checkout" is never the permissive case -- a job that
    declared an expected commit at submission time cannot have that promise silently
    waived just because git isn't available where it ends up running."""
    with patch("subprocess.check_output", side_effect=_mock_check_output(rev_parse_raises=True)):
        with pytest.raises(RuntimeError, match="not a git checkout"):
            _assert_and_log_git_state(_HEAD, under_slurm=True)


def test_real_repo_head_is_logged_when_run_for_real_under_slurm(capsys):
    """One real (unmocked) check against this actual repo -- confirms the real git
    invocation works, not just the mocked branches above."""
    probe = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("not a git checkout")
    real_head = probe.stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if status.stdout.strip():
        pytest.skip("working tree is dirty -- this test needs a clean tree to check the happy path")
    _assert_and_log_git_state(real_head, under_slurm=True)   # must not raise
    out = capsys.readouterr().out
    assert f"[git] HEAD={real_head}" in out
