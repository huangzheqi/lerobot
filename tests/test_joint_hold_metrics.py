import pytest

from isaac_so_arm101.scripts.rsl_rl.joint_hold_metrics import classify_hold, validate_hold_thresholds


def test_classify_hold_boundaries():
    assert classify_hold(5.0, 5.0, 20.0) == "stable"
    assert classify_hold(5.01, 5.0, 20.0) == "tracking_error"
    assert classify_hold(20.0, 5.0, 20.0) == "tracking_error"
    assert classify_hold(20.01, 5.0, 20.0) == "joint_collapse"


def test_validate_hold_thresholds_rejects_invalid_order():
    with pytest.raises(ValueError):
        validate_hold_thresholds(20.0, 5.0)
