"""Absolute-epoch schedules for V3 objective hyperparameters."""

import math


SUPPORTED_WEIGHT_NAMES = ("recon_loss", "commit_loss")


def _normalize_knots(knots):
    if not isinstance(knots, list) or not knots:
        raise ValueError("Schedule knots must be a non-empty list.")

    normalized = []
    for knot in knots:
        if isinstance(knot, dict):
            epoch = knot.get("epoch")
            value = knot.get("value")
        elif isinstance(knot, (list, tuple)) and len(knot) == 2:
            epoch, value = knot
        else:
            raise ValueError(
                "Each schedule knot must be {epoch: ..., value: ...} or [epoch, value]."
            )
        epoch = int(epoch)
        value = float(value)
        if epoch < 0 or not math.isfinite(value):
            raise ValueError(f"Invalid schedule knot: epoch={epoch}, value={value}")
        normalized.append((epoch, value))

    normalized.sort(key=lambda item: item[0])
    if len({epoch for epoch, _ in normalized}) != len(normalized):
        raise ValueError("Schedule knot epochs must be unique.")
    return normalized


def piecewise_linear_value(knots, epoch):
    """Interpolate a schedule at an absolute epoch, clamping at both ends."""
    points = _normalize_knots(knots)
    epoch = int(epoch)
    if epoch <= points[0][0]:
        return points[0][1]
    if epoch >= points[-1][0]:
        return points[-1][1]

    for (left_epoch, left_value), (right_epoch, right_value) in zip(
        points, points[1:]
    ):
        if left_epoch <= epoch <= right_epoch:
            fraction = (epoch - left_epoch) / (right_epoch - left_epoch)
            return left_value + fraction * (right_value - left_value)
    raise RuntimeError("Failed to resolve a value inside the schedule range.")


def apply_loss_schedules(loss_config, schedules, epoch):
    """Apply supported schedules in-place and return the active objective values."""
    if schedules:
        unknown = set(schedules) - {"relativity", "weights"}
        if unknown:
            raise ValueError(f"Unsupported loss schedule targets: {sorted(unknown)}")

        if "relativity" in schedules:
            loss_config["relativity"] = piecewise_linear_value(
                schedules["relativity"], epoch
            )

        weight_schedules = schedules.get("weights", {})
        unknown_weights = set(weight_schedules) - set(SUPPORTED_WEIGHT_NAMES)
        if unknown_weights:
            raise ValueError(
                f"Unsupported loss weight schedules: {sorted(unknown_weights)}"
            )
        for name, knots in weight_schedules.items():
            loss_config["weights"][name] = piecewise_linear_value(knots, epoch)

    return {
        "relativity": float(loss_config["relativity"]),
        "recon_loss_weight": float(loss_config["weights"]["recon_loss"]),
        "commit_loss_weight": float(loss_config["weights"]["commit_loss"]),
    }
