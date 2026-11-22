"""SLURM-only one-full-epoch resume check; temporary checkpoints are removed."""

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from trainer import Trainer
from utils.codebook_metrics import file_sha256


def assert_same(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            assert_same(first, second)
    else:
        assert a == b, (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Run on a SLURM GPU node")
    os.chdir(ROOT)
    torch.set_num_threads(4)
    reports = []
    with tempfile.TemporaryDirectory(prefix="insnotes-resume-") as temp:
        for index, path in enumerate(args.config):
            config = yaml.safe_load(path.read_text())
            source = Path(config["load_checkpoint"])
            parent = yaml.safe_load((source.parent / "config.yaml").read_text())
            expected = dict(parent, name=config["name"], load_checkpoint=str(source),
                            epochs=8000, inherit_resume_history=True)
            expected["load_checkpoint"] = config["load_checkpoint"]
            assert config == expected, "Continuation must only change budget and resume metadata"
            before_sha = file_sha256(source)
            state = torch.load(source, map_location="cpu", weights_only=False)
            assert state["epoch"] == 1999
            config.update(log_dir=temp, name=f"arm{index}", epochs=1)
            trainer = Trainer(config)
            trainer.prepare_data()
            trainer.build_model()
            assert trainer.start_epoch == 2000 and len(trainer.train_loader) == 30
            assert_same(state["model"], trainer.model.state_dict())
            assert_same(state["optimizer"], trainer.optimizer.state_dict())
            assert_same(state["scheduler"], trainer.scheduler.state_dict())
            assert_same(state["scaler"], trainer.scaler.state_dict())
            starting_lr = trainer.optimizer.param_groups[0]["lr"]
            run = Path(trainer.log_dir)
            for name in ("best_validation_loss.json", "best_macro_atom_purity.json"):
                old = json.loads((source.parent / name).read_text())
                new = json.loads((run / name).read_text())
                assert old["epoch"] == new["epoch"]
                assert file_sha256(source.parent / old["checkpoint"]) == file_sha256(run / new["checkpoint"])
            assert (source.parent / "audio_diagnostic_history.csv").read_bytes() == (
                run / "audio_diagnostic_history.csv").read_bytes()
            old_best = state["best_validation_loss"]
            del state
            trainer.train()
            grad_norms = {}
            for name in ("encoder", "decoder"):
                gradients = [p.grad.detach().double().square().sum()
                             for p in getattr(trainer.model, name).parameters() if p.grad is not None]
                grad_norms[name] = float(torch.stack(gradients).sum().sqrt())
                assert np.isfinite(grad_norms[name]) and grad_norms[name] > 0
            with (run / "loss_epoch_history.csv").open() as handle:
                new_rows = [r for r in csv.DictReader(handle) if int(r["epoch"]) == 2000]
            assert {r["partition"] for r in new_rows} == {"train", "val"}
            for row in new_rows:
                assert int(row["global_step"]) == 60030
                for key in ("total_loss", "recon_loss", "commit_loss", "v3_loss"):
                    assert np.isfinite(float(row[key]))
            val = float(next(r for r in new_rows if r["partition"] == "val")["total_loss"])
            best = json.loads((run / "best_validation_loss.json").read_text())
            assert best["validation_total_loss"] == min(old_best, val)
            current = torch.load(run / "cp_current_epoch2000.pt", map_location="cpu", weights_only=False)
            assert current["epoch"] == 2000 and current["scheduler"]["last_epoch"] == 2001
            assert len(list(run.glob("cp_*.pt"))) == 3
            with (run / "audio_diagnostic_history.csv").open() as handle:
                audio_rows = list(csv.DictReader(handle))
            assert int(audio_rows[0]["epoch"]) == 0 and int(audio_rows[-1]["epoch"]) == 2000
            for tag in ("epoch02000", "best_validation"):
                assert len(list((run / "audio_examples" / tag).glob("*.wav"))) == 9
            audio = json.loads((run / "audio_examples/best_validation/manifest.json").read_text())
            assert audio["epoch"] == best["epoch"]
            assert file_sha256(source) == before_sha
            reports.append(dict(config=str(path), status="passed", source_sha256=before_sha,
                next_epoch=2000, final_epoch=9999, total_steps=300000, smoke_updates=30,
                starting_lr=starting_lr, encoder_decoder_grad_norms=grad_norms,
                validation_loss=val, retained_best_epoch=best["epoch"],
                model_optimizer_scheduler_scaler_restored_exactly=True,
                inherited_audio_history=True, parent_checkpoint_unchanged=True))
            print("RESUME_SMOKE_ARM_PASSED", reports[-1], flush=True)
            del trainer, current
            gc.collect()
            torch.cuda.empty_cache()
    output = ROOT / "logs" / f"insnotes_resume_preflight_{os.environ['SLURM_JOB_ID']}.json"
    output.write_text(json.dumps(dict(job=os.environ["SLURM_JOB_ID"], node=socket.gethostname(),
        temporary_artifacts_deleted=True, arms=reports), indent=2) + "\n")
    print("RESUME_SMOKE_DONE", output, flush=True)


if __name__ == "__main__":
    main()
