import math
from collections.abc import Sequence
from fractions import Fraction

from validation import validate_plate_weights, validate_weights

UNLIMITED = sentinel("UNLIMITED")
type PlateCount = int | UNLIMITED 
type PlateStock = Sequence[tuple[float, PlateCount]]  ## (plate kgs)

PLATES: PlateStock = [
    (20.0, UNLIMITED),
    (15.0, UNLIMITED),
    (10.0, UNLIMITED),
    (5.0, UNLIMITED),
    (2.5, UNLIMITED),
    (1.5, UNLIMITED),
]
BARBELL_WEIGHT: float = 20.0  ## in kgs


def _find_unit(weights: list[float]) -> Fraction:
    """Largest weight every plate is a whole multiple of (the GCD of the plates).

    Uses exact fractions of the written-out decimals, so 2.5 and 1.5 give 0.5 and 3.1415 is 6283/2000.
    """
    fractions = [Fraction(str(weight)) for weight in weights]
    denominator = math.lcm(*(f.denominator for f in fractions))
    return Fraction(math.gcd(*(int(f * denominator) for f in fractions)), denominator)


def _to_units(
    available: PlateStock,
) -> tuple[Fraction, list[tuple[int, PlateCount]]]:
    """Express the usable plates (two or more, one per side) as whole multiples of their GCD.

    Returns:
        tuple[Fraction, list[tuple[int, PlateCount]]]: The unit in kgs, and
            (plate in units, count) pairs lightest first, so heavier plates win DP ties.
            The list is empty when no plate can be loaded on both sides."""
    usable = [(w, count) for w, count in available if count is UNLIMITED or count >= 2]
    if not usable:
        return Fraction(1), []
    unit = _find_unit([w for w, _ in usable])
    stock = [(round(Fraction(str(w)) / unit), count) for w, count in usable]
    return unit, sorted(stock, key=lambda plate: plate[0]) # type: ignore (Pylance doesn't understand PEP 661 sentinels in annotations yet)


def _get_plates_dp(
    target_weight: float,
    available: PlateStock = PLATES,
    bar_weight: float = BARBELL_WEIGHT,
) -> list[float]:
    """Calculates which plates to put on the barbell.
    Important: the output shows what plates to put on
    one side of the bar. The same sequence of plates
    should of course be put on the other side as well.

    Bounded-knapsack DP over plate types: fewest plates for every reachable
    per-side weight, then pick the reachable weight closest to the target
    (ties go lighter; equal plate counts prefer heavier plates).
    Greedy fails here, e.g. 3 kg per side is 1.5 + 1.5, not 2.5 + ???.

    Params:
        target_weight (float): Target weight in kgs. Rounded to nearest possible sequence.
        available (PlateStock, optional): (plate weight, count) pairs. The count is the total
            in the gym, so half of it (rounded down) can go on each side. Defaults to PLATES.
        bar_weight (float, optional): Weight of the empty bar in kgs. Defaults to BARBELL_WEIGHT.

    Returns:
        list[float]: List of plates to put on one side of the bar, heaviest first.

    Raises:
        ValueError: On bad input, see platulator.validation."""
    validate_weights(target_weight, bar_weight)
    validate_plate_weights([weight for weight, _ in available])
    unit, stock = _to_units(available)
    if not stock:
        return []

    # Kept unrounded: rounding here and again when picking the nearest sum can
    # pick the wrong side of a fake tie (27.4 kg would load 26, not 28).
    side = (target_weight - bar_weight) / 2 / unit
    # The lightest reachable sum above side minus any of its plates is <= side,
    # so the nearest reachable sum is never past side + the biggest plate.
    # ponytail: table size grows with 1/unit, so odd plates like 3.1415 kg make it slow.
    limit = math.ceil(side) + stock[-1][0]

    # min_plates[a]: fewest plates summing to a; picks[i][a]: how many of plate i that uses.
    inf = limit + 1  # more plates than any real answer can use
    min_plates = [0] + [inf] * limit
    picks: list[list[int]] = []
    for coin, count in stock:
        per_side = limit // coin if count is UNLIMITED else count // 2
        new_min = [inf] * (limit + 1)
        pick = [0] * (limit + 1)
        for amount in range(limit + 1):
            for k in reversed(range(min(per_side, amount // coin) + 1)):
                if min_plates[amount - k * coin] + k < new_min[amount]:
                    new_min[amount] = min_plates[amount - k * coin] + k
                    pick[amount] = k
        min_plates = new_min
        picks.append(pick)

    best = min(
        (a for a in range(limit + 1) if min_plates[a] < inf),
        key=lambda a: (abs(a - side), a),
    )
    plates: list[float] = []
    for (coin, _), pick in reversed(list(zip(stock, picks))):
        plates += [float(coin * unit)] * pick[best]
        best -= coin * pick[best]
    return plates


def get_plates(
    target_weight: float,
    available: PlateStock = PLATES,
    bar_weight: float = BARBELL_WEIGHT,
) -> str:
    """Calculates which plates to put on the barbell.
    Important: the output shows what plates to put on
    one side of the bar. The same sequence of plates
    should of course be put on the other side as well.

    Params:
        target_weight (float): Target weight in kgs. Rounded to nearest possible sequence.
        available (PlateStock, optional): (plate weight, total count) pairs. Defaults to PLATES.
        bar_weight (float, optional): Weight of the empty bar in kgs. Defaults to BARBELL_WEIGHT.

    Returns:
        str: List of plates to put on one side of the bar, heaviest first.

    Raises:
        ValueError: On bad input, see _get_plates_dp."""
    plates = _get_plates_dp(target_weight, available, bar_weight)
    plates_formatted: list[int | float] = [
        int(p) if p.is_integer() else p for p in plates
    ]
    return ", ".join(str(p) for p in plates_formatted)
