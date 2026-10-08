import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize
from model.normalization import configure_decoder_normalization, configure_backbone_normalization


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
        # Build the complete original architecture before replacing norms: keep
        # all shared Conv/Linear/VQ initialization and the RNG stream identical.
        self.encoder_normalization = config.get('encoder_normalization', 'batch')
        self.encoder_normalization_layers = configure_backbone_normalization(
            self.encoder, self.encoder_normalization, scope='encoder')
        self.decoder_normalization = config.get('decoder_normalization', 'batch')
        self.decoder_normalization_layers = configure_decoder_normalization(
            self.decoder, self.decoder_normalization, config.get('decoder_groupnorm_groups', 8))
        self.decoder_style_mode = config.get("decoder_style_mode", "fragment")
        if self.decoder_style_mode not in ("fragment", "page_mean", "page_mean_warmup"):
            raise ValueError(f"Invalid decoder_style_mode: {self.decoder_style_mode}")
        self.decoder_style_warmup_epochs = int(config.get("decoder_style_warmup_epochs", 100))
        if self.decoder_style_warmup_epochs < 0:
            raise ValueError("decoder_style_warmup_epochs must be nonnegative")
        # Opt-in buffer: old config/checkpoint state dicts remain unchanged.
        # Saving the epoch in model state makes standalone evaluation reproduce
        # the checkpoint's decoder regime, including checkpoints during warmup.
        if "decoder_style_mode" in config:
            self.register_buffer("_decoder_epoch", torch.tensor(0, dtype=torch.long))

    def set_decoder_epoch(self, epoch):
        if hasattr(self, "_decoder_epoch"):
            self._decoder_epoch.fill_(int(epoch))

    @property
    def active_decoder_style_mode(self):
        if self.decoder_style_mode == "page_mean_warmup":
            if int(self._decoder_epoch.item()) < self.decoder_style_warmup_epochs:
                return "page_mean"
            return "fragment"
        return self.decoder_style_mode

    def decoder_style_input(self, emb_s):
        if self.active_decoder_style_mode == "page_mean":
            if emb_s.ndim != 3:
                raise ValueError("page_mean requires [pages, fragments, style_dim]")
            return emb_s.mean(dim=1, keepdim=True).expand_as(emb_s)
        return emb_s

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
        """Expose the actual pre-project-out STE tensor from ONE VQ forward.

        Looking up indices after forward can read EMA-updated/replaced atoms,
        not the atoms just used by the decoder. Capture the native tensor at
        project_out instead; no second assignment or EMA update.
        The temporary hook is removed even when the quantizer raises.
        """
        captured = []

        def capture_native(module, inputs):
            captured.append(inputs[0])

        handle = self.vq.project_out.register_forward_pre_hook(capture_native)
        kwargs = {} if ema_update_weight is None else {"ema_update_weight": ema_update_weight}
        try:
            quantized, indices, commit_loss = self.vq(
                x, freeze_codebook=freeze_codebook, **kwargs
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one native-VQ project_out input")
        native_quantized = captured[0]
        # Preserve this interface's differentiability for eval-mode encoders
        # (e.g. frozen BN during downstream adaptation). The VQ library itself
        # only attaches STE in train mode; the captured hard value is still the
        # right one, unlike an EMA-updated index lookup.
        if torch.is_grad_enabled() and not native_quantized.requires_grad:
            projected = self.project_content_to_vq(x)
            native_quantized = projected + (native_quantized - projected).detach()
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
        output = self.decoder(emb_c, self.decoder_style_input(emb_s))

        return output

    def forward(self, x, freeze_codebook=False, return_native=False):
        emb_c, emb_s = self.encoder(x)
        if return_native:
            emb_c_vq, native_vq, vq_indices, commit_loss = self.quantize_with_native(
                emb_c, freeze_codebook=freeze_codebook
            )
        else:
            emb_c_vq, vq_indices, commit_loss = self.quantize(
                emb_c, freeze_codebook=freeze_codebook
            )
        output = self.decode(emb_c_vq, emb_s)

        result = (output, emb_c, emb_c_vq, vq_indices, commit_loss, emb_s)
        return (*result, native_vq) if return_native else result

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
