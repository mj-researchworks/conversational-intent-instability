#!/usr/bin/env python3
"""Reproduce the three-factor Shapley values reported in the manuscript.

Factors:
    A = class weighting
    B = LLM-generated augmentation
    C = translated-data augmentation

The inputs are the test-set Macro-F1 point estimates reported for the eight
factorial ablation configurations. No model training is performed here.
"""

from itertools import combinations
from math import factorial


SCORES = {
    frozenset(): 0.634,
    frozenset("A"): 0.622,
    frozenset("B"): 0.646,
    frozenset("C"): 0.669,
    frozenset("AB"): 0.644,
    frozenset("AC"): 0.642,
    frozenset("BC"): 0.670,
    frozenset("ABC"): 0.663,
}

FACTOR_NAMES = {
    "A": "class weighting",
    "B": "LLM-generated augmentation",
    "C": "translated-data augmentation",
}


def shapley_value(factor: str) -> float:
    """Calculate the exact Shapley value for one factor."""
    all_factors = frozenset(FACTOR_NAMES)
    other_factors = sorted(all_factors - {factor})
    number_of_factors = len(all_factors)
    value = 0.0

    for subset_size in range(len(other_factors) + 1):
        for subset_tuple in combinations(other_factors, subset_size):
            subset = frozenset(subset_tuple)
            weight = (
                factorial(len(subset))
                * factorial(number_of_factors - len(subset) - 1)
                / factorial(number_of_factors)
            )
            value += weight * (SCORES[subset | {factor}] - SCORES[subset])

    return value


def main() -> None:
    print("Input test Macro-F1 scores")
    for label in ("none", "A", "B", "C", "AB", "AC", "BC", "ABC"):
        key = frozenset() if label == "none" else frozenset(label)
        print(f"{label:>4}: {SCORES[key]:.3f}")

    print("\nShapley values")
    values = {factor: shapley_value(factor) for factor in FACTOR_NAMES}
    for factor, description in FACTOR_NAMES.items():
        print(f"{factor} ({description}): {values[factor]:+.6f} ({values[factor]:+.3f})")

    total_gain = SCORES[frozenset("ABC")] - SCORES[frozenset()]
    assert abs(sum(values.values()) - total_gain) < 1e-12
    print(f"\nCheck: sum = {sum(values.values()):+.6f}; ABC - baseline = {total_gain:+.6f}")


if __name__ == "__main__":
    main()
