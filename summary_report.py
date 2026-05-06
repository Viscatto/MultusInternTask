from collections import defaultdict
from pathlib import Path
import re


class SummaryReport:
    """Print and optionally plot grouped cell-count results."""

    _NAME_PATTERN = re.compile(r"^(?P<cell_line>.+)-t(?P<time>\d+(?:\.\d+)?)$")

    def __init__(self, results: list[dict]):
        self.results = results

    def print_summary(self) -> None:
        """Pretty-print a results table grouped by cell line."""
        print(self.to_text(), end="")

    def save_text(self, output_path: str | Path) -> Path:
        """Save the grouped summary report as a text file."""
        output_path = Path(output_path)
        output_path.write_text(self.to_text(), encoding="utf-8")
        return output_path

    def to_text(self) -> str:
        """Return the grouped summary report as a printable string."""
        if not self.results:
            return "\nNo results to summarize.\n"

        grouped = self.group_by_cell_line()
        successful_total = sum(
            result["cells"] for result in self.results if result["error"] is None
        )

        lines = [
            "",
            "CELL COUNT SUMMARY",
            "=" * 72,
        ]

        for cell_line in sorted(grouped):
            rows = grouped[cell_line]
            lines.extend(
                [
                    "",
                    f"Cell line: {cell_line}",
                    "-" * 72,
                    f"{'Image':<32} {'Time (h)':>10} {'Cells':>8}  Status",
                    "-" * 72,
                ]
            )

            for row in rows:
                result = row["result"]
                if result["error"]:
                    cells = "-"
                    status = f"ERROR - {result['error']}"
                else:
                    cells = str(result["cells"])
                    status = "OK"

                time = "-" if row["time_hours"] is None else f"{row['time_hours']:g}"
                lines.append(
                    f"{result['file']:<32} {time:>10} {cells:>8}  {status}"
                )

        lines.extend(
            [
                "",
                "=" * 72,
                f"{'TOTAL':<44} {successful_total:>8}",
                "",
            ]
        )
        return "\n".join(lines)

    def group_by_cell_line(self) -> dict[str, list[dict]]:
        """Return results grouped by parsed cell line, sorted by time."""
        grouped = defaultdict(list)

        for result in self.results:
            cell_line, time_hours = self._parse_image_name(result["file"])
            grouped[cell_line].append(
                {
                    "time_hours": time_hours,
                    "result": result,
                }
            )

        return {
            cell_line: sorted(
                rows,
                key=lambda row: (
                    row["time_hours"] is None,
                    row["time_hours"] if row["time_hours"] is not None else 0,
                    row["result"]["file"],
                ),
            )
            for cell_line, rows in grouped.items()
        }

    def plot_counts_over_time(self, output_path: str | Path) -> Path:
        """
        Save a line plot of cell count over time for each cell line.
        Matplotlib is imported only when plotting is enabled.
        """
        import matplotlib.pyplot as plt

        grouped = self.group_by_cell_line()
        output_path = Path(output_path)

        plt.figure(figsize=(8, 5))
        has_series = False

        for cell_line, rows in sorted(grouped.items()):
            points = [
                (row["time_hours"], row["result"]["cells"])
                for row in rows
                if row["time_hours"] is not None and row["result"]["error"] is None
            ]
            if not points:
                continue

            points.sort()
            times, counts = zip(*points)
            plt.plot(times, counts, marker="o", label=cell_line)
            has_series = True

        if not has_series:
            raise ValueError("No valid time-series data available to plot.")

        plt.xlabel("Time (hours)")
        plt.ylabel("Cell count")
        plt.title("Cell count over time")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def _parse_image_name(self, filename: str) -> tuple[str, float | None]:
        """
        Parse filenames like `CellLine-Variant-t24.png`.
        Everything before `-t` is treated as the cell line.
        """
        stem = Path(filename).stem
        match = self._NAME_PATTERN.match(stem)
        if not match:
            return stem, None

        return match.group("cell_line"), float(match.group("time"))
