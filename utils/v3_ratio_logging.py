import csv
import datetime
import os

from model.v3_loss import V3_RATIO_KEYS


HISTORY_COLUMNS = (
    "timestamp",
    "partition",
    "epoch",
    "global_step",
) + V3_RATIO_KEYS


class V3RatioLogger:
    """Persist epoch-level V3 ratios and render their training trajectories."""

    def __init__(self, log_dir, relativity):
        self.history_path = os.path.join(log_dir, "v3_ratio_epoch_history.csv")
        self.figure_path = os.path.join(log_dir, "v3_ratios.png")
        self.relativity = float(relativity)

    def log_epoch(self, partition, epoch, global_step, ratios):
        row = {column: "" for column in HISTORY_COLUMNS}
        row.update(
            {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "partition": partition,
                "epoch": epoch,
                "global_step": global_step,
            }
        )
        for name in V3_RATIO_KEYS:
            if name in ratios:
                row[name] = float(ratios[name])

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

        series = {}
        with open(self.history_path, newline="") as history_file:
            for row in csv.DictReader(history_file):
                partition = row["partition"]
                epoch = int(row["epoch"])
                for name in V3_RATIO_KEYS:
                    value = row.get(name, "")
                    if value != "":
                        series.setdefault(name, {}).setdefault(partition, []).append(
                            (epoch, float(value))
                        )
        if not series:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        labels = {
            "content_fragment_to_sample_ratio": "content fragment / sample",
            "style_sample_to_fragment_ratio": "style sample / fragment",
            "style_to_content_sample_ratio": "style / content sample",
            "content_to_style_fragment_ratio": "content / style fragment",
        }
        for axis, name in zip(axes.flat, V3_RATIO_KEYS):
            values_for_scale = []
            for partition, color in (("train", "tab:blue"), ("val", "tab:orange")):
                values = series.get(name, {}).get(partition, [])
                if values:
                    values_for_scale.extend(value for _, value in values)
                    axis.plot(
                        [epoch for epoch, _ in values],
                        [value for _, value in values],
                        label=partition,
                        color=color,
                        linewidth=1.2,
                    )
            axis.axhline(
                self.relativity,
                color="tab:red",
                linestyle="--",
                linewidth=1,
                label=f"relativity={self.relativity:g}",
            )
            positive_values = [value for value in values_for_scale if value > 0]
            if positive_values and max(positive_values) / min(positive_values) > 100:
                axis.set_yscale("log")
            axis.set_title(labels[name])
            axis.set_xlabel("epoch")
            axis.set_ylabel("ratio")
            axis.grid(alpha=0.25)
            axis.legend()

        fig.suptitle("V3 raw variance ratios")
        fig.tight_layout()
        temporary_path = self.figure_path + ".tmp.png"
        fig.savefig(temporary_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        os.replace(temporary_path, self.figure_path)
