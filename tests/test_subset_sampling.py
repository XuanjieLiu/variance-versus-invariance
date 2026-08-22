import unittest

from utils.subset_sampling import select_subset_indices


class FakeLettersDataset:
    def __init__(self):
        self.s_list = [
            "black",
            "blue",
            "green",
            "red",
            "teal",
            "purple",
            "orange",
            "brown",
        ]
        self.png_paths = [
            f"/unused/ABCDEFGHIJKLMNOPQRSTUVWXYZ_{style}_{index}_{style}.png"
            for style in self.s_list
            for index in range(100)
        ]

    def __len__(self):
        return len(self.png_paths)


class SubsetSamplingTest(unittest.TestCase):
    def test_style_stratified_512_is_balanced_and_reproducible(self):
        dataset = FakeLettersDataset()

        first, first_counts = select_subset_indices(
            dataset, 512, seed=0, strategy="style_stratified"
        )
        second, second_counts = select_subset_indices(
            dataset, 512, seed=0, strategy="style_stratified"
        )

        self.assertEqual(first, second)
        self.assertEqual(first_counts, second_counts)
        self.assertEqual(len(first), 512)
        self.assertEqual(set(first_counts.values()), {64})

    def test_different_seed_changes_selection(self):
        dataset = FakeLettersDataset()

        first, _ = select_subset_indices(
            dataset, 512, seed=0, strategy="style_stratified"
        )
        second, _ = select_subset_indices(
            dataset, 512, seed=1, strategy="style_stratified"
        )

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
