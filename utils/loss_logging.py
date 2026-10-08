import csv
import datetime
import os


LOSS_COLUMNS = (
    "total_loss",
    "recon_loss",
    "commit_loss",
    "v3_loss",
    "content_loss",
    "style_loss",
    "sample_loss",
    "fragment_loss",
    "cross_batch_loss",
    "cross_frag_loss",
)

HISTORY_COLUMNS = (
    "timestamp",
    "partition",
    "scope",
    "epoch",
    "global_step",
    "batch_step",
    "lr",
) + LOSS_COLUMNS


class LossLogger:
    """Persist loss components and periodically render epoch-level curves."""

    def __init__(self, log_dir, extra_columns=()):
        self.history_path = os.path.join(log_dir, "loss_history.csv")
        self.epoch_history_path = os.path.join(log_dir, "loss_epoch_history.csv")
        self.figure_path = os.path.join(log_dir, "loss_curves.png")
        self.loss_columns = tuple(dict.fromkeys((*LOSS_COLUMNS, *extra_columns)))
        self.history_columns = tuple(dict.fromkeys((*HISTORY_COLUMNS, *extra_columns)))
        self._csv_columns = {}

    @staticmethod
    def mean(loss_sums, count):
        if count <= 0:
            return {}
        return {name: value / count for name, value in loss_sums.items()}

    @staticmethod
    def format_losses(losses):
        ordered_names = [name for name in LOSS_COLUMNS if name in losses]
        ordered_names.extend(sorted(set(losses) - set(ordered_names)))
        return ", ".join(f"{name}={losses[name]:.6g}" for name in ordered_names)

    def log_step(
        self,
        partition,
        epoch,
        global_step,
        batch_step,
        losses,
        lr,
    ):
        self._append(
            self.history_path,
            partition,
            "step",
            epoch,
            global_step,
            batch_step,
            losses,
            lr,
        )

    def log_epoch(self, partition, epoch, global_step, losses, lr):
        record_args = (
            partition,
            "epoch",
            epoch,
            global_step,
            "",
            losses,
            lr,
        )
        self._append(self.history_path, *record_args)
        self._append(self.epoch_history_path, *record_args)

    def plot(self):
        if not os.path.exists(self.epoch_history_path):
            return

        # Import lazily so training can still start and produce CSV logs in a
        # minimal/headless environment where plotting is unavailable.
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        series = {}
        with open(self.epoch_history_path, newline="") as history_file:
            for row in csv.DictReader(history_file):
                partition = row["partition"]
                epoch = int(row["epoch"])
                for loss_name in self.loss_columns:
                    value = row.get(loss_name, "")
                    if value == "":
                        continue
                    series.setdefault(loss_name, {}).setdefault(partition, []).append(
                        (epoch, float(value))
                    )

        loss_names = [name for name in self.loss_columns if name in series]
        if not loss_names:
            return

        n_cols = 2
        n_rows = (len(loss_names) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(12, max(3.2 * n_rows, 4)),
            squeeze=False,
        )
        for axis, loss_name in zip(axes.flat, loss_names):
            plotted_values = []
            for partition, color in (("train", "tab:blue"), ("val", "tab:orange")):
                values = series[loss_name].get(partition, [])
                if values:
                    plotted_values.extend(item[1] for item in values)
                    axis.plot(
                        [item[0] for item in values],
                        [item[1] for item in values],
                        label=partition,
                        color=color,
                        linewidth=1.2,
                    )
            positive_values = [value for value in plotted_values if value > 0]
            if (
                positive_values
                and max(positive_values) / min(positive_values) > 100
            ):
                axis.set_yscale(
                    "symlog",
                    linthresh=max(min(positive_values), 1e-8),
                )
            axis.set_title(loss_name)
            axis.set_xlabel("epoch")
            axis.set_ylabel("loss" if loss_name.endswith("loss") else "value")
            axis.grid(alpha=0.25)
            axis.legend()

        for axis in axes.flat[len(loss_names) :]:
            axis.set_visible(False)

        fig.suptitle("Training and validation losses")
        fig.tight_layout()
        temporary_path = self.figure_path + ".tmp.png"
        fig.savefig(temporary_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        os.replace(temporary_path, self.figure_path)

    def _append(
        self,
        path,
        partition,
        scope,
        epoch,
        global_step,
        batch_step,
        losses,
        lr,
    ):
        row = {column: "" for column in self.history_columns}
        row.update(
            {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "partition": partition,
                "scope": scope,
                "epoch": epoch,
                "global_step": global_step,
                "batch_step": batch_step,
                "lr": lr,
            }
        )
        for name, value in losses.items():
            if name in row:
                row[name] = float(value)

        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        columns = self._csv_columns.get(path, self.history_columns)
        if not write_header and path not in self._csv_columns:
            with open(path, newline="") as incoming:
                old_columns = next(csv.reader(incoming))
            columns = tuple(dict.fromkeys((*old_columns, *self.history_columns)))
            if list(columns) != old_columns:
                # Resume histories may predate optional diagnostics. Expand the
                # header atomically; retain every existing column/value and put
                # blanks (not fabricated zeros) in historical rank columns.
                temporary_path = path + ".tmp"
                with open(path, newline="") as incoming, open(temporary_path, "w", newline="") as outgoing:
                    writer = csv.DictWriter(outgoing, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(csv.DictReader(incoming))
                os.replace(temporary_path, path)
        self._csv_columns[path] = columns
        with open(path, "a", newline="") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
