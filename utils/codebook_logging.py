import csv
import datetime
import json
import os


METRIC_COLUMNS = (
    "one_to_one_accuracy",
    "macro_atom_purity",
    "legacy_codebook_accuracy",
    "codebook_purity",
    "active_codes",
    "usage_perplexity",
    "dominant_label_coverage",
    "dominant_label_code_count_min",
    "dominant_label_code_count_max",
    "dominant_label_code_count_cv",
    "dominant_label_code_counts",
    "alias_within_content_rms",
    "alias_between_content_nn_median",
    "alias_within_between_ratio",
    "alias_nearest_same_distance_median",
    "alias_nearest_other_distance_median",
    "alias_nearest_same_closer_fraction",
)

HISTORY_COLUMNS = (
    "timestamp",
    "partition",
    "epoch",
    "global_step",
    "sample_count",
    "fragment_count",
) + METRIC_COLUMNS


class CodebookMetricLogger:
    def __init__(self, log_dir):
        self.history_path = os.path.join(log_dir, "codebook_epoch_history.csv")
        self.figure_path = os.path.join(log_dir, "codebook_metrics.png")

    def log_epoch(
        self,
        partition,
        epoch,
        global_step,
        metrics,
        sample_count,
        fragment_count,
    ):
        row = {column: "" for column in HISTORY_COLUMNS}
        row.update(
            {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "partition": partition,
                "epoch": epoch,
                "global_step": global_step,
                "sample_count": sample_count,
                "fragment_count": fragment_count,
            }
        )
        for name in METRIC_COLUMNS:
            if name in metrics:
                row[name] = (
                    json.dumps(metrics[name], separators=(",", ":"))
                    if isinstance(metrics[name], (list, tuple))
                    else metrics[name]
                )

        write_header = not os.path.exists(self.history_path) or os.path.getsize(
            self.history_path
        ) == 0
        with open(self.history_path, "a", newline="") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=HISTORY_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def plot(self):
        if not os.path.exists(self.history_path):
            return

        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        rows = []
        with open(self.history_path, newline="") as history_file:
            for row in csv.DictReader(history_file):
                if row["partition"] == "val":
                    rows.append(row)
        if not rows:
            return

        epochs = [int(row["epoch"]) for row in rows]
        fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
        for name, label in (
            ("one_to_one_accuracy", "Hungarian one-to-one"),
            ("macro_atom_purity", "macro atom purity"),
            ("codebook_purity", "usage-weighted purity"),
        ):
            values = []
            for row in rows:
                value = row.get(name, "")
                if name == "macro_atom_purity" and value == "":
                    value = row.get("legacy_codebook_accuracy", "")
                values.append(float(value))
            axes[0].plot(
                epochs,
                values,
                label=label,
                linewidth=1.4,
            )
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("score")
        axes[0].grid(alpha=0.25)
        axes[0].legend()

        for name, label in (
            ("active_codes", "active codes"),
            ("usage_perplexity", "usage perplexity"),
            ("dominant_label_coverage", "dominant-label coverage"),
        ):
            axes[1].plot(
                epochs,
                [float(row[name]) for row in rows],
                label=label,
                linewidth=1.4,
            )
        axes[1].set_ylabel("count / effective count")
        axes[1].grid(alpha=0.25)
        axes[1].legend()

        for name, label in (
            ("alias_within_between_ratio", "alias within/between ratio"),
            (
                "alias_nearest_same_closer_fraction",
                "nearest same closer fraction",
            ),
        ):
            axes[2].plot(
                epochs,
                [float(row.get(name) or "nan") for row in rows],
                label=label,
                linewidth=1.4,
            )
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("alias geometry")
        axes[2].grid(alpha=0.25)
        axes[2].legend()

        fig.suptitle("Validation codebook health")
        fig.tight_layout()
        temporary_path = self.figure_path + ".tmp.png"
        fig.savefig(temporary_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        os.replace(temporary_path, self.figure_path)
