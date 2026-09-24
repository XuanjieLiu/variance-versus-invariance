"""GPU-only checks: real CTRL/W100 resume states and the S3 scratch config."""
import csv
import itertools
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from trainer import Trainer


class TwoBatchLoader:
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return itertools.islice(iter(self.loader), 2)


def main():
    assert socket.gethostname().startswith("ws-") and torch.cuda.is_available()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cases = (
        ("ctrl", "cfg_vvi_rq2_c128_k26_control_seed0_resume1.yaml", 38),
        ("w100", "cfg_vvi_rq2_c128_k26_meanwarm100_seed0_resume1.yaml", 41),
        ("c512", "cfg_vvi_rq2_c512_k26_control_seed0.yaml", 0),
    )
    for name, filename, first_epoch in cases:
        config = yaml.safe_load((root / "configs/uppercase/rq2" / filename).read_text())
        config.update(name=f"smoke-s3-resume-{name}", debug=True, debug_portion=1,
                      snapshot_every_n_epochs=1)
        trainer = Trainer(config)
        trainer.prepare_data()
        trainer.train_loader = TwoBatchLoader(trainer.train_loader)
        trainer.build_model()
        assert trainer.start_epoch == first_epoch
        directory = Path(trainer.log_dir)
        if first_epoch:
            source = torch.load(config["load_checkpoint"], map_location="cpu", weights_only=False)
            assert trainer.scheduler.state_dict() == source["scheduler"]
            assert trainer.scaler.state_dict() == source["scaler"]
            assert abs(trainer.optimizer.param_groups[0]["lr"] - source["optimizer"]["param_groups"][0]["lr"]) < 1e-12
            for key, value in source["model"].items():
                torch.testing.assert_close(trainer.model.state_dict()[key].cpu(), value)
            if name == "w100":
                metadata = json.loads((directory / "best_macro_atom_purity.json").read_text())
                assert metadata["epoch"] == 39
                assert trainer.model.active_decoder_style_mode == "page_mean"
            del source
        trainer.train()
        assert (directory / f"cp_current_epoch{first_epoch}.pt").exists()
        assert (directory / f"cp_snapshot_epoch{first_epoch}.pt").exists()
        for name_csv in ("loss_epoch_history.csv", "codebook_epoch_history.csv", "disentanglement_probe_history.csv"):
            with (directory / name_csv).open() as handle:
                rows = list(csv.DictReader(handle))
            assert int(rows[-1]["epoch"]) == first_epoch
            assert int(rows[-1]["global_step"]) == first_epoch * 650 + 2
        for png in ("loss_curves.png", "codebook_metrics.png", "v3_ratios.png", "disentanglement_probes.png"):
            assert (directory / png).exists(), png
        print("SMOKE_RESUME_OK", name, "first_epoch", first_epoch, flush=True)
        del trainer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
