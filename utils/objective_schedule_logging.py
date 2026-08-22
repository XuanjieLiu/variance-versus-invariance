import csv
import datetime
import os


HISTORY_COLUMNS = (
    "timestamp",
    "epoch",
    "global_step",
    "relativity",
    "recon_loss_weight",
    "commit_loss_weight",
)


class ObjectiveScheduleLogger:
    def __init__(self, log_dir):
        self.history_path = os.path.join(
            log_dir, "objective_schedule_epoch_history.csv"
        )
        self.figure_path = os.path.join(log_dir, "objective_schedules.png")

    def log_epoch(self, epoch, global_step, values):
        row = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "epoch": int(epoch),
            "global_step": int(global_step),
        }
        row.update({name: float(values[name]) for name in HISTORY_COLUMNS[3:]})
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

        with open(self.history_path, newline="") as history_file:
            rows = list(csv.DictReader(history_file))
        if not rows:
            return

        epochs = [int(row["epoch"]) for row in rows]
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for axis, name, label in (
            (axes[0], "relativity", "relativity"),
            (axes[1], "recon_loss_weight", "reconstruction weight"),
            (axes[2], "commit_loss_weight", "commitment weight"),
        ):
            axis.plot(
                epochs,
                [float(row[name]) for row in rows],
                linewidth=1.5,
            )
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
        axes[-1].set_xlabel("absolute epoch")
        fig.suptitle("Objective schedules")
        fig.tight_layout()
        temporary_path = self.figure_path + ".tmp.png"
        fig.savefig(temporary_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        os.replace(temporary_path, self.figure_path)
