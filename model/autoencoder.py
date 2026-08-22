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

    @property
    def vq_code_dim(self):
        return int(self.vq.codebook.shape[-1])

    def project_content_to_vq(self, emb_c):
        """Project encoder content into the native VQ/codebook space."""
        return self.vq.project_in(emb_c)

    def lift_vq_codes(self, vq_codes):
        """Lift native VQ codes into the decoder's content space."""
        return self.vq.project_out(vq_codes)

    def get_native_vq_codes(self, indices):
        """Return native (pre-project-out) codebook atoms for code indices."""
        return self.vq.get_codes_from_indices(indices)

    def native_vq_ste_from_indices(self, emb_c, indices):
        """Build a native-space STE value while preserving encoder gradients."""
        projected = self.project_content_to_vq(emb_c)
        hard_codes = self.get_native_vq_codes(indices).to(projected)
        return projected + (hard_codes - projected).detach()

    def quantize_with_native(self, x, freeze_codebook=False, ema_update_weight=None):
        """Quantize once and expose both decoder-space and native VQ outputs."""
        if ema_update_weight is None:
            quantized, indices, commit_loss = self.vq(
                x, freeze_codebook=freeze_codebook
            )
        else:
            quantized, indices, commit_loss = self.vq(
                x,
                freeze_codebook=freeze_codebook,
                ema_update_weight=ema_update_weight,
            )
        native_quantized = self.native_vq_ste_from_indices(x, indices)
        return quantized, native_quantized, indices, commit_loss

    def quantize_native_prediction(
        self, native_prediction, freeze_codebook=False, ema_update_weight=None
    ):
        """Quantize an arithmetic prediction expressed directly in VQ space."""
        decoder_space_prediction = self.lift_vq_codes(native_prediction)
        _, native_quantized, indices, commit_loss = self.quantize_with_native(
            decoder_space_prediction,
            freeze_codebook=freeze_codebook,
            ema_update_weight=ema_update_weight,
        )
        return native_quantized, indices, commit_loss

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
