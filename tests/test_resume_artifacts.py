import csv
import json
from pathlib import Path
import tempfile
import unittest

from utils.resume_artifacts import HISTORIES, copy_completed_history, inherit_resume_artifacts


class ResumeArtifactTest(unittest.TestCase):
    def test_inherits_all_registered_diagnostics_without_future_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "source", Path(directory) / "dest"
            source.mkdir(); dest.mkdir()
            checkpoint = source / "cp_current_epoch199.pt"
            checkpoint.write_bytes(b"current")
            for name in HISTORIES:
                (source / name).write_text("epoch,global_step,value\n199,130000,1\n200,130650,2\n")
            inherit_resume_artifacts(checkpoint, dest, dict(epoch=199))
            lineage = json.loads((dest / "resume_lineage.json").read_text())
            self.assertEqual(set(lineage["inherited_history_rows"]), set(HISTORIES))
            for name in HISTORIES:
                self.assertEqual(lineage["inherited_history_rows"][name], 1)
                with (dest / name).open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual([row["epoch"] for row in rows], ["199"])

    def test_copies_only_completed_epochs_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, dest = root / "source.csv", root / "dest.csv"
            source.write_text("epoch,scope,value\n36,epoch,1\n37,epoch,2\n38,step,3\n")
            self.assertEqual(copy_completed_history(source, dest, 37), 2)
            with dest.open() as handle:
                self.assertEqual([int(row["epoch"]) for row in csv.DictReader(handle)], [36, 37])
            with self.assertRaises(FileExistsError):
                copy_completed_history(source, dest, 37)

    def test_inherits_actual_best_file_not_current_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "source", Path(directory) / "dest"
            source.mkdir(); dest.mkdir()
            checkpoint = source / "cp_current_epoch40.pt"
            checkpoint.write_bytes(b"current epoch40")
            (source / "cp_best_macro_atom_purity_epoch39.pt").write_bytes(b"best epoch39")
            (source / "loss_epoch_history.csv").write_text("epoch,value\n39,1\n40,2\n41,3\n")
            state = dict(epoch=40, best_macro_epoch=39, best_macro_atom_purity=.8,
                         best_macro_val_loss=.15, best_macro_stage="training")
            best = inherit_resume_artifacts(checkpoint, dest, state)
            self.assertEqual(Path(best).read_bytes(), b"best epoch39")
            self.assertEqual(checkpoint.read_bytes(), b"current epoch40")
            lineage = json.loads((dest / "resume_lineage.json").read_text())
            self.assertEqual(lineage["next_epoch"], 41)
            self.assertEqual(lineage["inherited_history_rows"]["loss_epoch_history.csv"], 2)
            self.assertEqual(json.loads((dest / "best_macro_atom_purity.json").read_text())["epoch"], 39)

    def test_rejects_missing_best_and_probe_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory) / "source", Path(directory) / "dest"
            source.mkdir(); dest.mkdir()
            checkpoint = source / "cp_current_epoch40.pt"
            checkpoint.write_bytes(b"current")
            state = dict(epoch=40, best_macro_epoch=39, best_macro_atom_purity=.8)
            with self.assertRaises(FileNotFoundError):
                inherit_resume_artifacts(checkpoint, dest, state)
            protocol = dict(fit_pages=["a"], score_pages=["b"], grid_pages=["c"], protocol={})
            (source / "disentanglement_probe_protocol.json").write_text(json.dumps(protocol))
            protocol["fit_pages"] = ["other"]
            (dest / "disentanglement_probe_protocol.json").write_text(json.dumps(protocol))
            with self.assertRaises(ValueError):
                inherit_resume_artifacts(checkpoint, dest, dict(epoch=40))


if __name__ == "__main__":
    unittest.main()
