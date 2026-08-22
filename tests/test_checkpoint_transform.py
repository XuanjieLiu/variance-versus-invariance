import unittest

import torch

from utils.checkpoint_transform import (
    expand_ema_codebook_state,
    pca_project_and_expand_ema_codebook_state,
)


class EmaCodebookExpansionTest(unittest.TestCase):
    def _state(self):
        embed = torch.tensor(
            [[[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]]], dtype=torch.float32
        )
        cluster = torch.tensor([[2.0, 4.0, 8.0]])
        return {
            "vq._codebook.embed": embed,
            "vq._codebook.cluster_size": cluster,
            "vq._codebook.embed_avg": embed * cluster.unsqueeze(-1),
            "encoder.weight": torch.tensor([7.0]),
        }

    def test_expansion_preserves_pair_centers_and_ema_mass(self):
        source = self._state()

        expanded, metadata = expand_ema_codebook_state(
            source, 3, 6, jitter_fraction=0.01, random_seed=0
        )

        embed = expanded["vq._codebook.embed"][0].reshape(3, 2, 2)
        cluster = expanded["vq._codebook.cluster_size"][0].reshape(3, 2)
        torch.testing.assert_close(embed.mean(dim=1), source["vq._codebook.embed"][0])
        torch.testing.assert_close(
            cluster.sum(dim=1), source["vq._codebook.cluster_size"][0]
        )
        torch.testing.assert_close(
            expanded["vq._codebook.embed_avg"],
            expanded["vq._codebook.embed"]
            * expanded["vq._codebook.cluster_size"].unsqueeze(-1),
        )
        self.assertAlmostEqual(metadata["ema_mass_before"], metadata["ema_mass_after"])
        self.assertIs(expanded["encoder.weight"], source["encoder.weight"])

    def test_jitter_is_reproducible_and_seeded(self):
        first, _ = expand_ema_codebook_state(self._state(), 3, 6, 0.01, 2)
        second, _ = expand_ema_codebook_state(self._state(), 3, 6, 0.01, 2)
        third, _ = expand_ema_codebook_state(self._state(), 3, 6, 0.01, 3)
        torch.testing.assert_close(
            first["vq._codebook.embed"], second["vq._codebook.embed"]
        )
        self.assertFalse(
            torch.equal(
                first["vq._codebook.embed"], third["vq._codebook.embed"]
            )
        )

    def test_rejects_non_double_expansion(self):
        with self.assertRaisesRegex(ValueError, "2x"):
            expand_ema_codebook_state(self._state(), 3, 7)

    def test_pca_projection_expansion_sets_affine_projection_and_preserves_mass(self):
        source = self._state()
        transformed, metadata = pca_project_and_expand_ema_codebook_state(
            source,
            source_atoms=3,
            target_atoms=6,
            target_dim=2,
            jitter_fraction=0.01,
            random_seed=4,
        )

        projected = transformed["vq._codebook.embed"][0].reshape(3, 2, 2)
        pair_centers = projected.mean(dim=1)
        weight_in = transformed["vq.project_in.weight"].double()
        bias_in = transformed["vq.project_in.bias"].double()
        source_centers = source["vq._codebook.embed"][0].double()
        expected = source_centers @ weight_in.t() + bias_in
        torch.testing.assert_close(pair_centers.double(), expected)

        weight_out = transformed["vq.project_out.weight"].double()
        bias_out = transformed["vq.project_out.bias"].double()
        reconstructed = expected @ weight_out.t() + bias_out
        torch.testing.assert_close(reconstructed, source_centers, atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(metadata["pca_explained_variance_ratio"], 1.0)
        self.assertAlmostEqual(metadata["ema_mass_before"], metadata["ema_mass_after"])

    def test_pca_projection_is_deterministic_including_component_signs(self):
        first, _ = pca_project_and_expand_ema_codebook_state(
            self._state(), 3, 6, 2, random_seed=9
        )
        second, _ = pca_project_and_expand_ema_codebook_state(
            self._state(), 3, 6, 2, random_seed=9
        )
        for key in (
            "vq._codebook.embed",
            "vq.project_in.weight",
            "vq.project_in.bias",
            "vq.project_out.weight",
            "vq.project_out.bias",
        ):
            torch.testing.assert_close(first[key], second[key])


if __name__ == "__main__":
    unittest.main()
