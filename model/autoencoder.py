import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


class CSAE(nn.Module):
    def __init__(self, config, Encoder, Decoder):
        """
        CSAE means Content-Style AutoEncoder
        """
        super().__init__()
        n_channels = config["n_channels"]
        W = config["n_feature"]
        H = config["fragment_len"]
        d_emb_c = config["d_emb_c"]
        d_emb_s = config["d_emb_s"]
        self.encoder = Encoder(n_channels, W, H, d_emb_c, d_emb_s)
        self.vq = VectorQuantize(
            dim=d_emb_c,
            codebook_size=config["n_atoms"],
            codebook_dim=config.get("vq_codebook_dim"),
            commitment_weight=1,
            decay=config["vq_ema_decay"] if "vq_ema_decay" in config else 0.98,
            kmeans_init=True,
            ema_update=True,
            rotation_trick=False,
            threshold_ema_dead_code=config["threshold_ema_dead_code"]
            if "threshold_ema_dead_code" in config
            else 0,
        )
        self.decoder = Decoder(n_channels, W, H, d_emb_c, d_emb_s)

    def encode(self, x):
        emb_c, emb_s = self.encoder(x)

        return emb_c, emb_s

    def quantize(self, x, freeze_codebook=False):
        quantized, indices, commit_loss = self.vq(x, freeze_codebook=freeze_codebook)

        return quantized, indices, commit_loss

    def decode(self, emb_c, emb_s):
        output = self.decoder(emb_c, emb_s)

        return output

    def forward(self, x, freeze_codebook=False):
        emb_c, emb_s = self.encoder(x)
        emb_c_vq, vq_indices, commit_loss = self.quantize(
            emb_c, freeze_codebook=freeze_codebook
        )
        output = self.decoder(emb_c_vq, emb_s)

        return output, emb_c, emb_c_vq, vq_indices, commit_loss, emb_s

    def get_model_size(self):
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params

        message = (
            f"Encoder params: {encoder_params}\n"
            + f"Decoder params: {decoder_params}\n"
            + f"Total params: {total_params}"
        )

        return message
