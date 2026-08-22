import unittest

from utils.objective_schedule import apply_loss_schedules, piecewise_linear_value


class ObjectiveScheduleTest(unittest.TestCase):
    def test_piecewise_linear_interpolation_and_clamped_boundaries(self):
        knots = [
            {"epoch": 60, "value": 15},
            {"epoch": 100, "value": 25},
            {"epoch": 140, "value": 25},
            {"epoch": 170, "value": 15},
        ]
        self.assertEqual(piecewise_linear_value(knots, 0), 15)
        self.assertEqual(piecewise_linear_value(knots, 60), 15)
        self.assertEqual(piecewise_linear_value(knots, 80), 20)
        self.assertEqual(piecewise_linear_value(knots, 100), 25)
        self.assertEqual(piecewise_linear_value(knots, 155), 20)
        self.assertEqual(piecewise_linear_value(knots, 999), 15)

    def test_resume_uses_absolute_epoch(self):
        loss_config = {
            "relativity": 15,
            "weights": {"recon_loss": 1, "commit_loss": 0.1},
        }
        schedules = {
            "relativity": [[0, 15], [100, 25], [200, 15]],
            "weights": {
                "recon_loss": [[0, 1], [200, 0.5]],
                "commit_loss": [[0, 0.1], [200, 0.2]],
            },
        }

        values = apply_loss_schedules(loss_config, schedules, epoch=150)

        self.assertEqual(values["relativity"], 20)
        self.assertAlmostEqual(values["recon_loss_weight"], 0.625)
        self.assertAlmostEqual(values["commit_loss_weight"], 0.175)
        self.assertEqual(loss_config["relativity"], 20)

    def test_invalid_duplicate_knots_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            piecewise_linear_value([[1, 2], [1, 3]], 1)


if __name__ == "__main__":
    unittest.main()
