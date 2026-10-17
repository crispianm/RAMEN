# To append to tests/test_ddad_scorer.py with the A4 scorer (synthetic only, CPU).
def test_a4_default_is_registered_378():
    assert DS.resolve_short_side(None, None) == 378
    assert DS.resolve_short_side("metric_vda_s", None) == 378
    assert DS.resolve_short_side("metric_vda_s", 378) == 378


def test_a4_release_may_use_518():
    assert DS.resolve_short_side("metric_vda_s", 518) == 518
    assert "518 input" in DS.RELEASE_LABEL_518 and "518" not in DS.RELEASE_LABEL.split("registered")[0]


@pytest.mark.parametrize("spec,ss", [(None, 518), (None, 448), ("metric_vda_s", 448), ("metric_vda_s", 0), ("unknown_spec", 518)])
def test_a4_forced_fail_ours_or_unregistered_cannot_change_input(spec, ss):
    with pytest.raises(ValueError):
        DS.resolve_short_side(spec, ss)


def test_a4_model_input_short_side_param_default_unchanged():
    import inspect
    assert inspect.signature(DS.model_input).parameters["short_side"].default == 378
