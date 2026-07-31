"""Create total-loss curves for runs produced before component logging existed."""

import argparse
import glob
import os
import re

from utils.loss_logging import LossLogger


TRAIN_PATTERN = re.compile(
    r"TRAIN - Epoch \[(?P<epoch>\d+)/\d+\], "
    r"Step \[(?P<step>\d+)/\d+\], Loss: (?P<loss>[0-9.eE+-]+)"
)
VAL_PATTERN = re.compile(
    r"VALIDATION - Epoch \[(?P<epoch>\d+)/\d+\], "
    r"Loss: (?P<loss>[0-9.eE+-]+)"
)


def parse_legacy_log(log_path):
    train = {}
    val = {}
    with open(log_path) as log_file:
        for line in log_file:
            match = TRAIN_PATTERN.search(line)
            if match:
                epoch = int(match.group("epoch"))
                step = int(match.group("step"))
                loss = float(match.group("loss"))
                state = train.setdefault(
                    epoch, {"last_step": -1, "loss_sum": 0.0, "count": 0}
                )
                # A repeated/lower step means this epoch was restarted from a
                # checkpoint. Keep the latest attempt rather than mixing attempts.
                if step <= state["last_step"]:
                    state.update(last_step=-1, loss_sum=0.0, count=0)
                state["last_step"] = step
                state["loss_sum"] += loss
                state["count"] += 1
                continue

            match = VAL_PATTERN.search(line)
            if match:
                val[int(match.group("epoch"))] = float(match.group("loss"))

    train_means = {
        epoch: state["loss_sum"] / state["count"]
        for epoch, state in train.items()
        if state["count"]
    }
    return train_means, val


def backfill_run(run_dir):
    logger = LossLogger(run_dir)
    if os.path.exists(logger.epoch_history_path):
        return False

    train, val = parse_legacy_log(os.path.join(run_dir, "log.txt"))
    for epoch, loss in sorted(train.items()):
        logger.log_epoch("train", epoch, "", {"total_loss": loss}, "")
    for epoch, loss in sorted(val.items()):
        logger.log_epoch("val", epoch, "", {"total_loss": loss}, "")
    logger.plot()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logs_dir",
        nargs="?",
        default="./logs",
        help="Directory containing one subdirectory per run.",
    )
    args = parser.parse_args()

    run_dirs = sorted(
        os.path.dirname(path)
        for path in glob.glob(os.path.join(args.logs_dir, "*", "log.txt"))
    )
    for run_dir in run_dirs:
        status = "created" if backfill_run(run_dir) else "skipped (already exists)"
        print(f"{run_dir}: {status}")


if __name__ == "__main__":
    main()
