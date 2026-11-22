"""Read-only, full-source-validation acceptance of retained RD-V3 states."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from scripts.accept_insnotes_mel import fingerprint
from trainer import Trainer
from utils.codebook_metrics import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Run on a SLURM GPU, never on the login node")
    torch.set_num_threads(4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for run in args.runs:
        run = run.resolve()
        best = json.loads((run / "best_validation_loss.json").read_text())
        checkpoints = [("best", run / best["checkpoint"]),
                       ("final", next(run.glob("cp_current_epoch*.pt")))]
        with (run / "loss_epoch_history.csv").open() as handle:
            logged = {int(r["epoch"]): r for r in csv.DictReader(handle) if r["partition"] == "val"}
        for role, path in checkpoints:
            epoch = int(path.stem.split("epoch")[-1])
            config = yaml.safe_load((run / "config.yaml").read_text())
            config.update(log_dir=str(output), name=run.name + "__" + role,
                          load_checkpoint=str(path), save_best_validation_loss=False,
                          snapshot_every_n_epochs=0, snapshot_epochs=[], wandb=False)
            config["disentanglement_probes"]["extra_grid_epochs"] = [epoch]
            trainer = Trainer(config)
            trainer.prepare_data()
            trainer.build_model()
            trainer.model.eval().requires_grad_(False)
            before = fingerprint(trainer.model)
            teacher_before = fingerprint(trainer.loss.teacher)
            result = trainer._collect_validation(epoch=epoch, global_step=(epoch + 1) * 2500)
            assert fingerprint(trainer.model) == before
            assert fingerprint(trainer.loss.teacher) == teacher_before
            assert all(p.grad is None for p in trainer.model.parameters())
            for key, value in result["losses"].items():
                if key in logged[epoch] and logged[epoch][key]:
                    assert abs(float(logged[epoch][key]) - value) < 1e-5, (role, epoch, key)
            metrics = result["codebook_metrics"]
            result.update(run=str(run), role=role, checkpoint=str(path), epoch=epoch,
                          checkpoint_sha256=file_sha256(path), unchanged_state=True,
                          split="full source validation; test and OOD not used",
                          basic_source_gate=bool(metrics["one_to_one_accuracy"] >= .99
                              and metrics["codebook_purity"] >= .99 and metrics["active_codes"] == 16
                              and metrics["usage_perplexity"] >= 15
                              and result["losses"]["recon_loss"] <= 1.25 * .04327775950431824))
            Trainer._write_json_atomic(str(Path(trainer.log_dir) / "audit.json"), result)
            results.append(result)
            Trainer._write_json_atomic(str(output / "summary.json"), results)
            print("RD_ACCEPTANCE", run.name, role, epoch, metrics, flush=True)
            del trainer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
