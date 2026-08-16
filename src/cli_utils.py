"""Command-line parsing helpers for dataset generation."""
import random
from .config import DEFECTS

def parse_defects(text):
    """
    Numbering:
      0 = none
      1..N = defects
      N+1 = all_combination
    """
    text = text.strip().lower()

    if text in ("none", "no", "clean", "0"):
        return ["none"]

    if text in ("all", "all_combination", "all_combinations"):
        defect_names = [
            d for d in DEFECTS if d != "none"
        ]
        selected = [
            d for d in defect_names
            if d not in ("black_tint", "white_tint")
        ]
        selected.append(
            random.choice(["black_tint", "white_tint"])
        )
        return selected

    try:
        numbers = [
            int(x.strip())
            for x in text.split(",")
            if x.strip()
        ]
    except ValueError:
        raise ValueError(
            "Enter serial number(s), separated by commas."
        )

    defect_names = [
        d for d in DEFECTS
        if d != "none"
    ]

    all_option = len(defect_names) + 1

    if 0 in numbers:
        if len(numbers) != 1:
            raise ValueError(
                "0 means none and cannot be combined with another number."
            )
        return ["none"]

    if all_option in numbers:
        if len(numbers) != 1:
            raise ValueError(
                f"{all_option} (all_combination) cannot be combined "
                "with another number."
            )

        selected = [
            d for d in defect_names
            if d not in ("black_tint", "white_tint")
        ]
        selected.append(
            random.choice(["black_tint", "white_tint"])
        )
        return selected

    invalid = [
        n for n in numbers
        if n < 1 or n > len(defect_names)
    ]

    if invalid:
        raise ValueError(
            "Invalid serial number(s): "
            + ", ".join(map(str, invalid))
            + f". Use 0 for none, 1-{len(defect_names)} for defects, "
              f"or {all_option} for all_combination."
        )

    selected = [
        defect_names[n - 1]
        for n in dict.fromkeys(numbers)
    ]

    if "black_tint" in selected and "white_tint" in selected:
        raise ValueError(
            "Black tint and white tint cannot be selected together."
        )

    if "blur" in selected and "smudge" in selected:
        raise ValueError(
            "Blur and smudge cannot be selected together."
        )

    return selected
