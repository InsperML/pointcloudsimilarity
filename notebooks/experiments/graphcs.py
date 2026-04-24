from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


FIGURE_PATTERN = re.compile(
    r"^(?P<kind>training_(?:similarity|loss))_"
    r"(?P<n_samples>\d+)_"
    r"(?P<n_features>\d+)_"
    r"(?P<n_classes>\d+)_"
    r"(?P<hidden_dim>\d+)_"
    r"(?P<epochs>\d+)_"
    r"(?P<cluster_std>\d+(?:\.\d+)?)\.pdf$"
)


@dataclass(frozen=True)
class FigureMetadata:
    kind: str
    n_samples: int
    n_features: int
    n_classes: int
    hidden_dim: int
    epochs: int
    cluster_std: float

    @classmethod
    def from_path(cls, path: Path) -> "FigureMetadata | None":
        match = FIGURE_PATTERN.match(path.name)
        if match is None:
            return None
        groups = match.groupdict()
        return cls(
            kind=groups["kind"],
            n_samples=int(groups["n_samples"]),
            n_features=int(groups["n_features"]),
            n_classes=int(groups["n_classes"]),
            hidden_dim=int(groups["hidden_dim"]),
            epochs=int(groups["epochs"]),
            cluster_std=float(groups["cluster_std"]),
        )


def parse_figure_arguments(figures_dir: Path, kind: str) -> dict[FigureMetadata, Path]:
    figures: dict[FigureMetadata, Path] = {}
    for path in sorted(figures_dir.glob(f"{kind}_*.pdf")):
        metadata = FigureMetadata.from_path(path)
        if metadata is not None:
            figures[metadata] = path
    return figures


def _pdf_to_image_array(pdf_path: Path, dpi: int = 180):
    with tempfile.TemporaryDirectory() as tmpdir:
        output_prefix = Path(tmpdir) / pdf_path.stem
        subprocess.run(
            [
                "pdftoppm",
                "-singlefile",
                "-png",
                "-r",
                str(dpi),
                str(pdf_path),
                str(output_prefix),
            ],
            check=True,
            capture_output=True,
        )
        return mpimg.imread(f"{output_prefix}.png")


def _sorted_unique(values: set[Any]) -> list[Any]:
    return sorted(values, key=lambda value: (float(value), str(value)))


def build_custom_matrix(
    figures_dir: Path,
    *,
    kind: str = "training_similarity",
    row_key: str = "n_features",
    col_key: str = "cluster_std",
    fixed_filters: dict[str, Any] | None = None,
    output_path: Path | None = None,
    dpi: int = 180,
    output_dir: Path | None = None,
) -> Path:
    fixed_filters = fixed_filters or {}
    figures = parse_figure_arguments(figures_dir, kind)

    filtered: dict[FigureMetadata, Path] = {}
    for metadata, path in figures.items():
        if all(getattr(metadata, key) == value for key, value in fixed_filters.items()):
            filtered[metadata] = path

    if not filtered:
        raise FileNotFoundError(
            f"No figures matched kind={kind!r} and filters={fixed_filters!r} in {figures_dir}"
        )

    row_values = _sorted_unique({getattr(metadata, row_key) for metadata in filtered})
    col_values = _sorted_unique({getattr(metadata, col_key) for metadata in filtered})

    fig, axes = plt.subplots(
        len(row_values),
        len(col_values),
        figsize=(3.2 * len(col_values) + 0.5, 2.8 * len(row_values)),
        squeeze=False,
    )

    for row_index, row_value in enumerate(row_values):
        for col_index, col_value in enumerate(col_values):
            axis = axes[row_index][col_index]
            match = None
            for metadata, path in filtered.items():
                if getattr(metadata, row_key) == row_value and getattr(metadata, col_key) == col_value:
                    match = path
                    break

            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_frame_on(True)
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)

            if match is None:
                axis.set_facecolor("white")
                axis.text(0.5, 0.5, "missing", ha="center", va="center")
                continue

            image = _pdf_to_image_array(match, dpi=dpi)
            axis.imshow(image)
            if row_index == 0:
                axis.set_title(f"{col_value}", fontsize=12)

        axes[row_index][0].set_ylabel(f"{row_value}", fontsize=12, rotation=0, labelpad=30, va="center")

    fig.suptitle(
        f"{kind.replace('_', ' ').title()} matrix",
        fontsize=16,
        y=0.995,
    )
    fig.text(
        0.5,
        0.91,
        col_key,
        ha="center",
        va="top",
        fontsize=14,
    )
    fig.text(
        0.03,
        0.5,
        row_key,
        rotation=90,
        ha="left",
        va="center",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.05, 0, 1, 0.94))

    if output_path is None:
        if output_dir is None:
            output_path = figures_dir / f"{kind}_{row_key}_by_{col_key}_matrix.png"
        else:
            output_path = output_dir / f"{kind}_{row_key}_by_{col_key}_matrix.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a matrix of saved experiment figures from the figures folder."
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path(__file__).parent / "figures",
        help="Directory that contains the saved experiment figures.",
    )
    parser.add_argument(
        "--kind",
        default="training_similarity",
        choices=("training_similarity", "training_loss"),
        help="Figure prefix to assemble.",
    )
    parser.add_argument(
        "--row-key",
        default="n_features",
        choices=("n_samples", "n_features", "n_classes", "hidden_dim", "epochs", "cluster_std"),
        help="Metadata field to use for matrix rows.",
    )
    parser.add_argument(
        "--col-key",
        default="cluster_std",
        choices=("n_samples", "n_features", "n_classes", "hidden_dim", "epochs", "cluster_std"),
        help="Metadata field to use for matrix columns.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Optional filter for n_samples.",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=2,
        help="Optional filter for n_classes.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=10,
        help="Optional filter for hidden_dim.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Optional filter for epochs.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output path for the assembled matrix image.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Rasterization DPI used when converting PDFs into images.",
    )
    args = parser.parse_args()

    fixed_filters = {
        "n_samples": args.n_samples,
        "n_classes": args.n_classes,
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
    }
    fixed_filters.pop(args.row_key, None)
    fixed_filters.pop(args.col_key, None)

    output_path = build_custom_matrix(
        args.figures_dir,
        kind=args.kind,
        row_key=args.row_key,
        col_key=args.col_key,
        fixed_filters=fixed_filters,
        output_path=args.output_path,
        dpi=args.dpi,
        output_dir=Path(__file__).parent / "assembled_matrices",
    )
    print(output_path)


if __name__ == "__main__":
    main()