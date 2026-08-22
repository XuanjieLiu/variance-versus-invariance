import unittest

import torch
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

    def test_native_vq_interface_preserves_shapes_values_and_encoder_gradient(self):
        config = {
            "n_channels": 1,
            "n_feature": 1,
            "fragment_len": 1,
            "d_emb_c": 512,
            "d_emb_s": 512,
            "n_atoms": 26,
            "vq_codebook_dim": 4,
            "threshold_ema_dead_code": 0,
            "vq_ema_decay": 0.98,
        }
        model = CSAE(config, DummyEncoder, DummyDecoder)
        model.eval()
        content = torch.randn(2, 3, 512, requires_grad=True)

        high, native, indices, _ = model.quantize_with_native(
            content, freeze_codebook=True
        )
        hard_native = model.get_native_vq_codes(indices)

        self.assertEqual(tuple(high.shape), (2, 3, 512))
        self.assertEqual(tuple(native.shape), (2, 3, 4))
        self.assertEqual(tuple(indices.shape), (2, 3))
        torch.testing.assert_close(native.detach(), hard_native.detach())
        torch.testing.assert_close(
            high.detach(), model.lift_vq_codes(hard_native).detach()
        )
        native.square().mean().backward()
        self.assertIsNotNone(content.grad)
        self.assertGreater(float(content.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
