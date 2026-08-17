"""Pure helpers for classifying SO101 joint-hold diagnostics."""


def validate_hold_thresholds(stable_threshold_deg: float, collapse_threshold_deg: float) -> None:
    """Validate ordered positive drift thresholds."""
    if stable_threshold_deg <= 0.0:
        raise ValueError("stable_threshold_deg must be positive")
    if collapse_threshold_deg <= stable_threshold_deg:
        raise ValueError("collapse_threshold_deg must be greater than stable_threshold_deg")


def classify_hold(
    max_arm_drift_deg: float,
    stable_threshold_deg: float,
    collapse_threshold_deg: float,
) -> str:
    """Classify a hold run from the largest absolute arm-joint drift."""
    validate_hold_thresholds(stable_threshold_deg, collapse_threshold_deg)
    if max_arm_drift_deg <= stable_threshold_deg:
        return "stable"
    if max_arm_drift_deg <= collapse_threshold_deg:
        return "tracking_error"
    return "joint_collapse"
