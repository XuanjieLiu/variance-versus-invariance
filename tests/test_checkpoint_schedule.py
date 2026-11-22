import unittest

from utils.checkpoint_schedule import snapshot_due, validate_snapshot_schedule


class SnapshotScheduleTests(unittest.TestCase):
    def test_default_and_periodic_unchanged(self):
        self.assertFalse(any(snapshot_due({}, e) for e in range(200)))
        self.assertEqual([e for e in range(50) if snapshot_due(
            {"snapshot_every_n_epochs": 25}, e)], [24, 49])

    def test_explicit_and_union(self):
        config = {"snapshot_epochs": [0, 4, 19, 49]}
        validate_snapshot_schedule(config)
        self.assertEqual([e for e in range(50) if snapshot_due(config, e)], [0, 4, 19, 49])
        config["snapshot_every_n_epochs"] = 25
        self.assertEqual([e for e in range(50) if snapshot_due(config, e)], [0, 4, 19, 24, 49])

    def test_reject_invalid(self):
        for value in ([True], [-1], [0, 0], [1.5], "0", None):
            with self.assertRaises(ValueError):
                validate_snapshot_schedule({"snapshot_epochs": value})
        for value in (-1, True, 2.5):
            with self.assertRaises(ValueError):
                validate_snapshot_schedule({"snapshot_every_n_epochs": value})
