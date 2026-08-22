import tempfile
import unittest
from pathlib import Path

from dataset.uppercase_letters.generate_ood_styles import (
    OOD_STYLE_NAMES,
    SEEN_STYLE_NAMES,
    build_page_specs,
)


class UppercaseOODGeneratorTest(unittest.TestCase):
    def test_palette_and_split_specs_are_disjoint_and_balanced(self):
        self.assertTrue(set(SEEN_STYLE_NAMES).isdisjoint(OOD_STYLE_NAMES))
        with tempfile.TemporaryDirectory() as temp_dir:
            specs = build_page_specs(
                Path(temp_dir),
                {"adapt": 2, "val": 1, "test": 3},
            )

        self.assertEqual(len(specs), len(OOD_STYLE_NAMES) * 6)
        paths = [spec.output_path for spec in specs]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(0 <= spec.image_seed < 2**32 for spec in specs))
        for split, expected in {"adapt": 2, "val": 1, "test": 3}.items():
            for style in OOD_STYLE_NAMES:
                count = sum(
                    1 for spec in specs if spec.split == split and spec.style == style
                )
                self.assertEqual(count, expected)


if __name__ == "__main__":
    unittest.main()
