"""Input checks for the plate calculator. Each raises ValueError on bad input."""

import math


def validate_weights(target_weight: float, bar_weight: float) -> None:
    """Check that the target and bar are finite, the bar isn't negative and the target can be lifted.

    Raises:
        ValueError: If any of those don't hold."""
    if not (math.isfinite(target_weight) and math.isfinite(bar_weight)):
        raise ValueError(f"invalid target {target_weight} or bar weight {bar_weight}")
    if bar_weight < 0:
        raise ValueError(f"bar weight {bar_weight} kg is negative")
    if target_weight < bar_weight:
        raise ValueError(
            f"target {target_weight} kg is lighter than the {bar_weight} kg bar"
        )


def validate_plate_weights(weights: list[float]) -> None:
    """Check that every plate weight is finite and positive.

    Raises:
        ValueError: On the first plate that isn't."""
    for weight in weights:
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"plate weight {weight} kg is not a positive number")
