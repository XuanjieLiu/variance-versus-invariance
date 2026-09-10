"""GPU-only integration smoke for the three style-decoder regimes.

Artifacts are disposable smoke directories, never formal ledger entries.
"""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader, Subset
import yaml
from model.factory import get_model
from trainer import Trainer


def main():
    assert socket.gethostname().startswith("ws-"), socket.gethostname()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    summaries = []
    for name in ("control", "pagemean", "meanwarm100"):
        path = root / "configs/uppercase/rq2" / f"cfg_vvi_rq2_c128_k26_{name}_seed0.yaml"
        config = yaml.safe_load(path.read_text())
        config.update(name=f"smoke-style-bypass-{name}", debug=True, debug_portion=1,
                      macro_best_health_gate={}, snapshot_every_n_epochs=1)
        config["disentanglement_probes"]["reconstruction_every_n_epochs"] = 1
        trainer = Trainer(config)
        trainer.prepare_data()
        trainer.train_loader = DataLoader(Subset(trainer.train_loader.dataset, list(range(64))),
                                         batch_size=32, shuffle=True, num_workers=4)
        trainer.build_model()
        checksum = hashlib.sha256()
        for key, value in trainer.model.state_dict().items():
            if key != "_decoder_epoch":
                checksum.update(key.encode())
                checksum.update(value.detach().cpu().numpy().tobytes())
        if name == "meanwarm100":
            trainer.start_epoch = 99
            trainer.config["epochs"] = 2
        trainer.train()
        directory = Path(trainer.log_dir)
        for artifact in ("loss_curves.png", "codebook_metrics.png", "v3_ratios.png",
                         "disentanglement_probes.png", "disentanglement_probe_history.csv",
                         "current_checkpoint.json", "best_macro_atom_purity.json"):
            assert (directory / artifact).exists(), artifact
        with (directory / "disentanglement_probe_history.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        if name == "meanwarm100":
            assert [row["decoder_style_mode"] for row in rows] == ["page_mean", "fragment"]
        for row in rows:
            assert int(row["fit_pages"]) == int(row["score_pages"]) == 256
            epoch = int(row["epoch"])
            saved = torch.load(directory / f"cp_snapshot_epoch{epoch}.pt", map_location="cpu", weights_only=False)
            model = get_model(config["dataloader"], config["model_config"])
            model.load_state_dict(saved["model"], strict=True)
            assert model.active_decoder_style_mode == row["decoder_style_mode"]
            assert all(key in saved for key in ("optimizer", "scheduler", "scaler"))
            metadata_path = next((directory / "reconstruction_diagnostics").glob(f"reconstruction_epoch{epoch:03d}*.json"))
            metadata = json.loads(metadata_path.read_text())
            assert len(metadata["cells"]) == 26 * 8
            assert metadata["mapping_fragments"] == 2600 * 26
            assert metadata["checkpoint"] == f"cp_snapshot_epoch{epoch}.pt"
        assert len(list(directory.glob("cp*.pt"))) <= len(rows) + 2
        summaries.append({"mode": name, "initial_model_sha256": checksum.hexdigest(),
                          "epochs": [int(row["epoch"]) for row in rows],
                          "probe_rows": rows, "path": str(directory)})
        del trainer, saved, model
        torch.cuda.empty_cache()
    assert len({row["initial_model_sha256"] for row in summaries}) == 1
    print("SMOKE_STYLE_BYPASS_OK", json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
