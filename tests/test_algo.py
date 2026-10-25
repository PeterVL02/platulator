"""Tests for the plate calculator in platulator.algo."""

import math
import unittest
from fractions import Fraction

from platulator.algo import (
    UNLIMITED,
    PlateStock,
    _find_unit,
    _get_plates_dp,
    get_plates,
)


class GetPlatesDpTest(unittest.TestCase):
    def test_plates(self) -> None:
        """Each target maps to the fewest plates for the nearest loadable weight."""
        cases: list[tuple[float, list[float]]] = [
            (20, []),  # bar only
            (26, [1.5, 1.5]),  # greedy would get stuck on 2.5
            (100, [20, 20]),
            (60, [20]),
            (85, [20, 10, 2.5]),
            (27, [1.5, 1.5]),  # 26 and 28 are tied, take the lighter one
            (52, [10, 1.5, 1.5, 1.5, 1.5]),  # exact beats closer-but-off 15 + 1.5
        ]
        for target, expected in cases:
            with self.subTest(target=target):
                self.assertEqual(_get_plates_dp(target), expected)


class AvailablePlatesTest(unittest.TestCase):
    def test_limited_plates(self) -> None:
        """Counts are gym totals, so each side gets half, rounded down."""
        cases: list[tuple[float, PlateStock, list[float]]] = [
            # 2x 20 kg = one per side; the rest has to come from 10s
            (120, [(20, 2), (10, UNLIMITED)], [20, 10, 10, 10]),
            # 3x 20 kg still only gives one per side
            (100, [(20, 3), (10, UNLIMITED)], [20, 10, 10]),
            # a single 20 kg plate can't be loaded symmetrically
            (60, [(20, 1), (10, UNLIMITED)], [10, 10]),
            # out of plates: take the heaviest load we can make
            (200, [(20, 4), (5, 2)], [20, 20, 5]),
            # unlimited 1.5s still beat the greedy dead end
            (26, [(2.5, UNLIMITED), (1.5, UNLIMITED)], [1.5, 1.5]),
            # only 1.5s, 3 kg per side is exact
            (26, [(1.5, 4)], [1.5, 1.5]),
            # no 1.5s left: 3 kg per side rounds to 2.5
            (26, [(2.5, UNLIMITED), (1.5, 0)], [2.5]),
            # nothing usable at all
            (100, [(20, 1)], []),
            (100, [], []),
        ]
        for target, available, expected in cases:
            with self.subTest(target=target, available=available):
                self.assertEqual(_get_plates_dp(target, available), expected)

    def test_formatting_with_available(self) -> None:
        """get_plates passes the available plates through."""
        self.assertEqual(get_plates(120, [(20, 2), (10, UNLIMITED)]), "20, 10, 10, 10")


class BarWeightTest(unittest.TestCase):
    def test_other_bars(self) -> None:
        """The bar's weight comes off the target before plates are picked."""
        cases: list[tuple[float, float, list[float]]] = [
            (60, 15, [20, 2.5]),  # 15 kg bar: 22.5 per side
            (60, 10, [20, 5]),  # 10 kg technique bar: 25 per side
            (60, 0, [20, 10]),  # no bar (e.g. a sled): 30 per side
            (32.5, 7.5, [10, 2.5]),  # 7.5 kg EZ bar: 12.5 per side
            (15, 15, []),  # just the 15 kg bar
            (23.1415, 3.1415, [10]),  # odd bars are fine, only plates need the 0.5 grid
        ]
        for target, bar, expected in cases:
            with self.subTest(target=target, bar=bar):
                self.assertEqual(_get_plates_dp(target, bar_weight=bar), expected)

    def test_get_plates_passes_bar_weight(self) -> None:
        """get_plates passes the bar weight through."""
        self.assertEqual(get_plates(60, bar_weight=15), "20, 2.5")


class InvalidInputTest(unittest.TestCase):
    def test_target_lighter_than_bar(self) -> None:
        """You can't lift less than the empty bar."""
        cases: list[tuple[float, float]] = [
            (10, 20),
            (19.99, 20),
            (0, 20),
            (-50, 20),
            (10, 15),
            (-1, 0),
        ]
        for target, bar in cases:
            with self.subTest(target=target, bar=bar), self.assertRaises(ValueError):
                _get_plates_dp(target, bar_weight=bar)

    def test_target_lighter_than_bar_via_get_plates(self) -> None:
        """The public function raises too, instead of printing an empty string."""
        with self.assertRaises(ValueError):
            get_plates(10)

    def test_not_a_number(self) -> None:
        """NaN and infinite targets or bars are rejected with a clear ValueError."""
        nan, inf = float("nan"), float("inf")
        cases: list[tuple[float, float]] = [
            (nan, 20),
            (inf, 20),
            (-inf, 20),
            (60, nan),
            (60, inf),
        ]
        for target, bar in cases:
            with self.subTest(target=target, bar=bar), self.assertRaises(ValueError):
                _get_plates_dp(target, bar_weight=bar)

    def test_negative_bar(self) -> None:
        """A bar can't weigh less than nothing."""
        with self.assertRaises(ValueError):
            _get_plates_dp(60, bar_weight=-20)

    def test_bad_plate_weights(self) -> None:
        """Non-positive or non-finite plates are rejected."""
        for weight in (0, -5, float("nan"), float("inf"), float("-inf")):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                _get_plates_dp(60, [(weight, UNLIMITED)])

    def test_one_bad_plate_spoils_the_stock(self) -> None:
        """A bad plate raises even when the other plates could do the job."""
        with self.assertRaises(ValueError):
            _get_plates_dp(60, [(20, UNLIMITED), (0, UNLIMITED)])


class WeirdInputTest(unittest.TestCase):
    def test_weird_targets(self) -> None:
        """Off-grid targets round to the nearest weight that can actually be loaded."""
        cases: list[tuple[float, list[float]]] = [
            (99, [20, 15, 1.5, 1.5, 1.5]),  # the README example
            (27.4, [2.5, 1.5]),  # 28 is 0.6 off, 26 is 1.4 off
            (99.99, [20, 20]),
            (20.4, []),  # rounds down to the bare bar
            (21, []),  # 0.5 per side: nothing is closer than 1.5
            (22.5, [1.5]),  # 1.25 per side: 1.5 is closer than nothing
            (100.0, [20, 20]),  # float and int targets agree
            (math.pi * 10 + 20, [10, 2.5, 1.5, 1.5]),  # 15.71 per side, 15.5 beats 16.5
        ]
        for target, expected in cases:
            with self.subTest(target=target):
                self.assertEqual(_get_plates_dp(target), expected)

    def test_heavy_target(self) -> None:
        """A silly-heavy target still loads exactly, mostly 20s, without blowing up."""
        plates = _get_plates_dp(3000)
        self.assertEqual(sum(plates) * 2 + 20, 3000)
        self.assertEqual(plates, [20] * 74 + [10])

    def test_unusual_but_valid_plates(self) -> None:
        """Any positive plate size works; the DP unit is the GCD of the plates."""
        cases: list[tuple[float, PlateStock, list[float]]] = [
            (71, [(25, UNLIMITED), (0.5, UNLIMITED)], [25, 0.5]),
            (
                37,
                [(4.5, UNLIMITED), (3.5, UNLIMITED)],
                [4.5, 3.5],
            ),  # 8 and 9 tie at 8.5
            (40, [(7, UNLIMITED), (3, UNLIMITED)], [7, 3]),
            (22.5, [(20, UNLIMITED), (1.25, UNLIMITED)], [1.25]),  # unit 1.25
            (20.6, [(5, UNLIMITED), (0.1, UNLIMITED)], [0.1, 0.1, 0.1]),  # unit 0.1
            (32.566, [(3.1415, UNLIMITED)], [3.1415, 3.1415]),  # unit 3.1415
            # unit 1/2000 kg: a big table, but still exact
            (66.283, [(20, UNLIMITED), (3.1415, UNLIMITED)], [20, 3.1415]),
        ]
        for target, available, expected in cases:
            with self.subTest(target=target, available=available):
                self.assertEqual(_get_plates_dp(target, available), expected)


class FindUnitTest(unittest.TestCase):
    def test_find_unit(self) -> None:
        """The unit is the largest weight every plate is a whole multiple of."""
        cases: list[tuple[list[float], Fraction]] = [
            ([20, 15, 10, 5, 2.5, 1.5], Fraction(1, 2)),
            ([20, 1.25], Fraction(5, 4)),
            ([7, 3], Fraction(1)),
            ([20, 10], Fraction(10)),
            ([3.1415], Fraction(6283, 2000)),
            ([20, 3.1415], Fraction(1, 2000)),
            ([0.1, 0.25], Fraction(1, 20)),  # no float noise from 0.1
        ]
        for weights, expected in cases:
            with self.subTest(weights=weights):
                self.assertEqual(_find_unit(weights), expected)


class GetPlatesTest(unittest.TestCase):
    def test_formatting(self) -> None:
        """Plates are comma-separated, heaviest first, whole numbers without .0."""
        cases: list[tuple[float, str]] = [
            (20, ""),  # bar only
            (60, "20"),  # single plate, no trailing comma
            (100, "20, 20"),
            (85, "20, 10, 2.5"),
            (26, "1.5, 1.5"),
            (52, "10, 1.5, 1.5, 1.5, 1.5"),
        ]
        for target, expected in cases:
            with self.subTest(target=target):
                self.assertEqual(get_plates(target), expected)


if __name__ == "__main__":
    unittest.main()
