"""SLURM-only RD-V3 teacher audit and two-arm train/eval/resume preflight."""

import argparse
import copy
import csv
import json
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from model.distillation_loss import DISTILLATION_COLUMNS, build_distillation_loss
from scripts.accept_insnotes_mel import fingerprint
from trainer import Trainer


def config_for(lr):
    return yaml.safe_load((ROOT / f"configs/hex/rd/cfg_rd_v3_hex_rank2_lr{lr}_seed0.yaml").read_text())


def teacher_reference(directory):
    options = config_for("1e4")["representation_distillation"]
    config = yaml.safe_load(Path(options["teacher_config"]).read_text())
    config.update(log_dir=str(directory), name="teacher_reference", load_checkpoint=options["teacher_checkpoint"],
                  save_best_validation_loss=False, snapshot_every_n_epochs=0, wandb=False)
    config["disentanglement_probes"]["extra_grid_epochs"].append(59)
    trainer = Trainer(config)
    trainer.prepare_data()
    trainer.build_model()
    trainer.model.eval().requires_grad_(False)
    before = fingerprint(trainer.model)
    reference = trainer._collect_validation(epoch=59, global_step=150000)
    assert fingerprint(trainer.model) == before
    assert all(p.grad is None for p in trainer.model.parameters())
    m = reference["codebook_metrics"]
    assert m["one_to_one_accuracy"] >= .99 and m["codebook_purity"] >= .99
    assert m["active_codes"] == 16 and m["usage_perplexity"] >= 15
    reference.update(checkpoint=options["teacher_checkpoint"], sha256=options["teacher_sha256"],
                     checkpoint_epoch=59, unchanged_state=True, split="full source validation",
                     test_or_ood_accessed=False)
    Trainer._write_json_atomic(str(directory / "teacher_reference.json"), reference)
    print("TEACHER_REFERENCE", json.dumps(m), flush=True)
    del trainer
    torch.cuda.empty_cache()
    return reference


def smoke_pair():
    results, initial_hashes = [], []
    with tempfile.TemporaryDirectory(prefix="smoke-rd-v3-", dir=ROOT / "logs") as temp:
        for lr in ("1e4", "3e4"):
            config = config_for(lr)
            config.update(log_dir=temp, name="lr" + lr, epochs=1, num_workers=0,
                          log_every_n_steps=5, save_best_macro_atom_purity=False,
                          macro_best_health_gate={})
            config["representation_distillation"]["calibration_pages"] = 64
            for key in ("bn_diagnostic", "disentanglement_probes", "training_usage_monitor",
                        "optimization_diagnostics", "per_style_codebook_monitor"):
                config[key] = {"enabled": False}
            trainer = Trainer(copy.deepcopy(config))
            trainer.prepare_data()
            # Same ten full source batches in both arms, all test/OOD data sealed.
            ids = torch.randperm(len(trainer.train_loader.dataset),
                                 generator=torch.Generator().manual_seed(71))[:320].tolist()
            trainer.train_loader = DataLoader(Subset(trainer.train_loader.dataset, ids),
                batch_size=32, shuffle=False, generator=torch.Generator().manual_seed(72))
            trainer.val_loader = DataLoader(Subset(trainer.val_loader.dataset, list(range(64))),
                batch_size=32, shuffle=False, generator=torch.Generator().manual_seed(73))
            trainer.build_model()
            run = Path(trainer.log_dir)
            initial_hashes.append(json.loads((run / "initial_model_state.json").read_text())["sha256"])
            teacher_before = fingerprint(trainer.loss.teacher)
            trainer.train()
            encoder_grad = sum(float(p.grad.square().sum()) for p in trainer.model.encoder.parameters()
                               if p.grad is not None) ** .5
            decoder_grad = sum(float(p.grad.square().sum()) for p in trainer.model.decoder.parameters()
                               if p.grad is not None) ** .5
            assert encoder_grad > 0 and decoder_grad > 0
            assert all(p.grad is None for p in trainer.loss.teacher.parameters())
            assert fingerprint(trainer.loss.teacher) == teacher_before
            assert set(map(id, trainer.model.parameters())).isdisjoint(
                set(map(id, trainer.loss.teacher.parameters())))
            for file in ("loss_curves.png", "codebook_metrics.png", "v3_ratios.png",
                         "distillation_protocol.json", "cp_current_epoch0.pt"):
                assert (run / file).is_file(), file
            with (run / "loss_epoch_history.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                for key in DISTILLATION_COLUMNS:
                    assert row[key] != "" and torch.isfinite(torch.tensor(float(row[key])))
                total = sum(float(row[k]) for k in ("weighted_recon_commit_loss",
                    "weighted_distill_content_loss", "weighted_distill_style_loss"))
                assert abs(total - float(row["total_loss"])) < 1e-4
            checkpoint = run / "cp_current_epoch0.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert "representation_distillation" in state
            assert not any("teacher" in name for name in state["model"])
            x = next(iter(trainer.val_loader))[0].cuda()
            trainer.model.eval()
            with torch.no_grad():
                y, c, q, _, commit, s = trainer.model(x, freeze_codebook=True)
                old = trainer.loss.compute_loss(y, c, q, commit, s, x)
            reloaded_loss = build_distillation_loss(config, "cuda",
                saved_metadata=state["representation_distillation"])
            with torch.no_grad():
                new = reloaded_loss.compute_loss(y, c, q, commit, s, x)
            for key in old: torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
            for key, bad in (("content_weight", 2), ("teacher_sha256", "bad")):
                changed = copy.deepcopy(config)
                changed["representation_distillation"][key] = bad
                try:
                    build_distillation_loss(changed, "cuda", saved_metadata=state["representation_distillation"])
                except ValueError:
                    pass
                else:
                    raise AssertionError("Changed checkpoint protocol accepted")
            del reloaded_loss, state, old, new, y, c, q, commit, s
            validation_before = {k: float(v) for k, v in trainer._collect_validation()["losses"].items()}
            teacher_metadata = copy.deepcopy(trainer.loss.distillation_metadata)
            train_loader, val_loader = trainer.train_loader, trainer.val_loader
            del trainer
            torch.cuda.empty_cache()
            config.update(name="resume_lr" + lr, load_checkpoint=str(checkpoint),
                          inherit_resume_history=True)
            resumed = Trainer(config)
            resumed.prepare_data()
            resumed.train_loader, resumed.val_loader = train_loader, val_loader
            resumed.build_model()
            assert resumed.start_epoch == 1
            assert resumed.loss.distillation_metadata == teacher_metadata
            validation_after = resumed._collect_validation()["losses"]
            for key, value in validation_before.items():
                assert abs(validation_after[key] - value) < 1e-6, key
            resumed.train()
            assert (Path(resumed.log_dir) / "cp_current_epoch1.pt").is_file()
            results.append({"lr": lr, "initial_updates": 10, "resumed_updates": 10,
                            "encoder_grad_norm": encoder_grad, "decoder_grad_norm": decoder_grad,
                            "teacher_unchanged": True, "resume_verified": True,
                            "last_validation": validation_before,
                            "calibration": teacher_metadata["calibration"]})
            del resumed
            torch.cuda.empty_cache()
        assert initial_hashes[0] == initial_hashes[1]
        for handler in list(logging.getLogger().handlers):
            if isinstance(handler, logging.FileHandler) and str(handler.baseFilename).startswith(temp + "/"):
                handler.close()
                logging.getLogger().removeHandler(handler)
    assert not Path(temp).exists()
    return {"arms": results, "matched_initial_state_sha256": initial_hashes[0], "smoke_deleted": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("SLURM GPU allocation required; never run on login")
    os.chdir(ROOT)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    reference = teacher_reference(args.output)
    result = smoke_pair()
    result.update(teacher_metrics=reference["codebook_metrics"], job=os.environ["SLURM_JOB_ID"],
                  host=socket.gethostname(), all_checks_passed=True)
    Trainer._write_json_atomic(str(args.output / "preflight.json"), result)
    print("RD_V3_PREFLIGHT_PASSED", args.output, flush=True)


if __name__ == "__main__":
    main()
