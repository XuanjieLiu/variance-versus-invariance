import json
import tempfile
import unittest
from pathlib import Path

from utils.evaluation_paths import resolve_evaluation_paths


class EvaluationPathTest(unittest.TestCase):
    def _make_run(self, root, name="run-a"):
        run_dir = Path(root) / "logs" / name
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text("active_checkpoint: null\n")
        current = run_dir / "cp_current_epoch9.pt"
        best = run_dir / "cp_best_macro_atom_purity_epoch7.pt"
        current.touch()
        best.touch()
        (run_dir / "current_checkpoint.json").write_text(
            json.dumps({"checkpoint": current.name, "epoch": 9})
        )
        (run_dir / "best_macro_atom_purity.json").write_text(
            json.dumps({"checkpoint": best.name, "epoch": 7})
        )
        return run_dir, current, best

    def test_run_name_defaults_to_current_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir, current, _ = self._make_run(temporary_dir)

            resolved = resolve_evaluation_paths(
                run=run_dir.name,
                logs_dir=Path(temporary_dir) / "logs",
            )

            self.assertEqual(resolved["run_dir"], str(run_dir.resolve()))
            self.assertEqual(resolved["active_checkpoint"], str(current.resolve()))
            self.assertEqual(
                resolved["config"], str((run_dir / "config.yaml").resolve())
            )

    def test_best_alias_uses_macro_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir, _, best = self._make_run(temporary_dir)

            resolved = resolve_evaluation_paths(
                run=run_dir,
                active_checkpoint="best",
            )

            self.assertEqual(resolved["active_checkpoint"], str(best.resolve()))

    def test_checkpoint_path_infers_run_and_config(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir, current, _ = self._make_run(temporary_dir)

            resolved = resolve_evaluation_paths(active_checkpoint=current)

            self.assertEqual(resolved["run_dir"], str(run_dir.resolve()))
            self.assertEqual(
                resolved["config"], str((run_dir / "config.yaml").resolve())
            )

    def test_explicit_config_has_precedence(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir, current, _ = self._make_run(temporary_dir)
            explicit_config = Path(temporary_dir) / "override.yaml"
            explicit_config.write_text("active_checkpoint: null\n")

            resolved = resolve_evaluation_paths(
                run=run_dir,
                active_checkpoint=current.name,
                config=explicit_config,
            )

            self.assertEqual(resolved["config"], str(explicit_config.resolve()))

    def test_rejects_checkpoint_outside_requested_run(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir, _, _ = self._make_run(temporary_dir)
            other_dir, other_checkpoint, _ = self._make_run(temporary_dir, "run-b")
            self.assertNotEqual(run_dir, other_dir)

            with self.assertRaisesRegex(ValueError, "outside the requested run"):
                resolve_evaluation_paths(
                    run=run_dir,
                    active_checkpoint=other_checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
