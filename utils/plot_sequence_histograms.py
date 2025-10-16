"""Generate histograms for each step of the zoom sequence."""

from __future__ import annotations

import argparse
import ast
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot histograms of zoom sequence positions from a CSV file."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("ground_truth_sequences.csv"),
        help="Path to the CSV file with a 'sequence' column.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset_visualizations"),
        help="Directory where the histogram figure will be saved.",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="sequence_histograms.png",
        help="Filename for the generated histogram figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the histograms interactively in addition to saving them.",
    )
    return parser.parse_args()


def load_sequences(csv_path: Path) -> List[List[int]]:
    sequences: List[List[int]] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sequence" not in reader.fieldnames:
            raise ValueError("CSV file does not contain a 'sequence' column.")

        for row in reader:
            raw_sequence = row["sequence"]
            if not raw_sequence:
                continue
            parsed = ast.literal_eval(raw_sequence)
            if not isinstance(parsed, (list, tuple)):
                raise ValueError(f"Sequence for row {row} is not a list or tuple.")
            sequence_list = list(parsed)
            if not all(isinstance(item, int) for item in sequence_list):
                raise ValueError(f"Sequence contains non-integer values: {sequence_list}")
            sequences.append(sequence_list)
    if not sequences:
        raise ValueError("No sequences were loaded from the CSV file.")
    return sequences


def group_by_position(sequences: List[List[int]]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for sequence in sequences:
        for position, value in enumerate(sequence):
            grouped[position].append(value)
    return grouped


def plot_histograms(grouped: Dict[int, List[int]], output_path: Path, show: bool) -> None:
    if not grouped:
        raise ValueError("No sequence values to plot.")

    all_values = [value for values in grouped.values() for value in values]
    min_value = min(all_values)
    max_value = max(all_values)
    bin_edges = list(range(min_value, max_value + 2))

    positions = sorted(grouped.keys())
    fig, axes = plt.subplots(1, len(positions), figsize=(5 * len(positions), 4), squeeze=False)

    for idx, position in enumerate(positions):
        axis = axes[0][idx]
        axis.hist(grouped[position], bins=bin_edges, edgecolor="black", align="left")
        axis.set_title(f"Step {position + 1}")
        axis.set_xlabel("Patch index")
        if idx == 0:
            axis.set_ylabel("Count")
        axis.set_xticks(sorted(set(grouped[position])))

    fig.suptitle("Zoom Sequence Patch Distributions")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"[plot_sequence_histograms] saved figure to {output_path}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sequences = load_sequences(args.csv_path)
    grouped = group_by_position(sequences)
    plot_histograms(grouped, args.output_dir / args.figure_name, args.show)


if __name__ == "__main__":
    main()
