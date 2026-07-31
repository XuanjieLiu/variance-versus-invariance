import unittest

import torch.nn as nn

from model.autoencoder import CSAE


class DummyEncoder(nn.Module):
    def __init__(self, n_channels, width, height, d_emb_c, d_emb_s):
        super().__init__()


class DummyDecoder(nn.Module):
    def __init__(self, n_channels, width, height, d_emb_c, d_emb_s):
        super().__init__()


class ProjectedCodebookTest(unittest.TestCase):
    def test_codebook_dimension_can_differ_from_content_dimension(self):
        config = {
            "n_channels": 1,
            "n_feature": 1,
            "fragment_len": 1,
            "d_emb_c": 512,
            "d_emb_s": 512,
            "n_atoms": 26,
            "vq_codebook_dim": 4,
            "threshold_ema_dead_code": 16,
            "vq_ema_decay": 0.98,
        }

        model = CSAE(config, DummyEncoder, DummyDecoder)

        self.assertEqual(tuple(model.vq.codebook.shape), (26, 4))
        self.assertEqual(model.vq.project_in.in_features, 512)
        self.assertEqual(model.vq.project_in.out_features, 4)
        self.assertEqual(model.vq.project_out.in_features, 4)
        self.assertEqual(model.vq.project_out.out_features, 512)


if __name__ == "__main__":
    unittest.main()
