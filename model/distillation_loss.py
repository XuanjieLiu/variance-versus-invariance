"""RD-V3: frozen source representations supervise an independently trained V3.

The shared trainer still owns data, optimization, validation and checkpoints.
Only the objective changes: V3 structure statistics remain detached diagnostics.
No labels, teacher decoder outputs, or teacher weights initialize the student.
"""

import copy
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml

from model.factory import get_model
from model.v3_loss import V3Loss
from utils.codebook_metrics import file_sha256


DISTILLATION_COLUMNS = (
    "distill_content_mse", "distill_style_mse",
    "distill_content_nmse", "distill_style_nmse",
    "weighted_distill_content_loss", "weighted_distill_style_loss",
    "weighted_recon_commit_loss", "distill_vq_vector_mse",
    "distill_teacher_cell_agreement",
)


def validate_distillation_config(config):
    options = config.get("representation_distillation", {})
    if not isinstance(options, dict):
        raise ValueError("representation_distillation must be a mapping")
    allowed = {"enabled", "teacher_config", "teacher_checkpoint", "teacher_sha256",
               "content_weight", "style_weight", "calibration_pages",
               "calibration_seed", "variance_floor"}
    if set(options) - allowed:
        raise ValueError(f"Unknown representation_distillation options: {set(options) - allowed}")
    if not isinstance(options.get("enabled", False), bool):
        raise ValueError("representation_distillation.enabled must be boolean")
    if not options.get("enabled", False):
        return False
    for key in ("teacher_config", "teacher_checkpoint", "teacher_sha256"):
        if not options.get(key):
            raise ValueError(f"RD-V3 requires {key}")
    for key in ("content_weight", "style_weight"):
        value = float(options.get(key, 1.0))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    floor = float(options.get("variance_floor", 1e-8))
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("variance_floor must be positive and finite")
    pages = options.get("calibration_pages", 2048)
    if not isinstance(pages, int) or isinstance(pages, bool) or pages < 2:
        raise ValueError("calibration_pages must be an integer >= 2")
    if config.get("method") != "V3" or config.get("loss_schedules"):
        raise ValueError("RD-V3 requires method V3, with no objective schedules")
    loss = config["loss_config"]
    if loss.get("rank_regularization", {}).get("enabled", False):
        raise ValueError("RD-V3 does not optimize the teacher's rank regularizer")
    if "supersample_content" in loss or "widen_style" in loss:
        raise ValueError("RD-V3 uses ordinary, diagnostic-only V3 statistics")
    for key, weight in loss["weights"].items():
        if not math.isfinite(float(weight)) or weight < 0:
            raise ValueError(f"Invalid weight for {key}")
        if key not in ("recon_loss", "commit_loss") and weight != 0:
            raise ValueError(f"RD-V3 must not optimize {key}")
    if loss["weights"].get("recon_loss", 0) <= 0:
        raise ValueError("RD-V3 must train its own decoder via reconstruction")
    if config.get("checkpoint_transform"):
        raise ValueError("RD-V3 does not transform or inherit the teacher weights")
    return True


@torch.no_grad()
def calibrate_teacher(teacher, dataset, batch_size, options, device):
    """Fixed train-only scalar variances; do not consume the training RNG stream."""
    count = min(int(options.get("calibration_pages", 2048)), len(dataset))
    if count < 2:
        raise ValueError("RD-V3 calibration requires at least two source train pages")
    generator = torch.Generator().manual_seed(int(options.get("calibration_seed", 20261120)))
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=0, generator=generator)
    moments = [None, None]
    teacher.eval()
    for x, *_ in loader:
        for branch, value in enumerate(teacher.encode(x.to(device))):
            value = value.detach().reshape(-1, value.shape[-1]).double()
            n, mean = len(value), value.mean(0)
            m2 = (value - mean).square().sum(0)
            previous = moments[branch]
            if previous is not None:
                n0, mean0, m20 = previous
                delta = mean - mean0
                m2 = m20 + m2 + delta.square() * n0 * n / (n0 + n)
                mean = mean0 + delta * n / (n0 + n)
                n += n0
            moments[branch] = (n, mean, m2)
    floor = float(options.get("variance_floor", 1e-8))
    result = {"split": "train", "pages": count,
              "page_indices_sha256": hashlib.sha256(str(indices).encode()).hexdigest()}
    for name, (n, mean, m2) in zip(("content", "style"), moments):
        variance = float((m2 / n).mean())
        if not math.isfinite(variance):
            raise ValueError("Nonfinite teacher calibration variance")
        result[name + "_variance"] = max(variance, floor)
        result[name + "_variance_before_floor"] = variance
        result[name + "_mean"] = mean.cpu().tolist()
        result[name + "_fragments"] = n
    return result


class RepresentationDistillationLoss(V3Loss):
    """Keep V3 diagnostic keys, but backprop only recon/commit and distillation."""

    def __init__(self, loss_config, model_config, teacher, metadata):
        super().__init__(loss_config, model_config=model_config)
        self.teacher = teacher.eval().requires_grad_(False)
        self.distillation_metadata = copy.deepcopy(metadata)
        options = metadata["options"]
        self.content_weight = float(options.get("content_weight", 1.0))
        self.style_weight = float(options.get("style_weight", 1.0))
        self.content_variance = float(metadata["calibration"]["content_variance"])
        self.style_variance = float(metadata["calibration"]["style_variance"])
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.content_variance, self.style_variance)):
            raise ValueError("Invalid saved teacher calibration")

    def compute_loss(self, x_hat, z_c, z_c_vq, commit_loss, z_s, x, *, native_vq=None):
        self.teacher.eval()
        with torch.no_grad():
            # Detached monitoring preserves the original V3 ratios/hinges without
            # accidentally retaining their gradients when their weights are zero.
            losses = super().compute_loss(x_hat.detach(), z_c.detach(), z_c_vq.detach(),
                                           commit_loss.detach(), z_s.detach(), x.detach())
            target_c, target_s = self.teacher.encode(x)
            target_vq, target_ids, _ = self.teacher.quantize(target_c, freeze_codebook=True)
            _, student_in_teacher_ids, _ = self.teacher.quantize(z_c.detach(), freeze_codebook=True)
            cell_agreement = (student_in_teacher_ids == target_ids).float().mean()
        if z_c.shape != target_c.shape or z_s.shape != target_s.shape:
            raise ValueError("Teacher/student embedding shapes do not match")
        content_mse = F.mse_loss(z_c, target_c)
        style_mse = F.mse_loss(z_s, target_s)
        content_nmse = content_mse / self.content_variance
        style_nmse = style_mse / self.style_variance
        recon = F.mse_loss(x_hat, x)
        commit = commit_loss.mean()
        base = self.config["weights"].get("recon_loss", 0) * recon
        base = base + self.config["weights"].get("commit_loss", 0) * commit
        weighted_c = self.content_weight * content_nmse
        weighted_s = self.style_weight * style_nmse
        losses.update({
            "recon_loss": recon, "commit_loss": commit,
            "total_loss": base + weighted_c + weighted_s,
            "distill_content_mse": content_mse, "distill_style_mse": style_mse,
            "distill_content_nmse": content_nmse, "distill_style_nmse": style_nmse,
            "weighted_distill_content_loss": weighted_c,
            "weighted_distill_style_loss": weighted_s,
            "weighted_recon_commit_loss": base,
            "distill_vq_vector_mse": F.mse_loss(z_c_vq.detach(), target_vq),
            "distill_teacher_cell_agreement": cell_agreement,
        })
        if not all(bool(torch.isfinite(v).all()) for v in losses.values()):
            raise ValueError("Nonfinite RD-V3 objective or diagnostics")
        return losses


def build_distillation_loss(config, device, *, train_dataset=None, saved_metadata=None):
    """Create a verified frozen teacher; checkpoint metadata makes eval/resume exact."""
    if not validate_distillation_config(config):
        raise ValueError("Distillation is not enabled")
    options = copy.deepcopy(config["representation_distillation"])
    teacher_path = Path(options["teacher_checkpoint"]).resolve()
    sha = file_sha256(teacher_path)
    if sha != options["teacher_sha256"]:
        raise ValueError("Teacher checkpoint checksum mismatch")
    teacher_config = yaml.safe_load(Path(options["teacher_config"]).read_text())
    if (teacher_config["dataloader"] != config["dataloader"]
            or teacher_config["model_config"] != config["model_config"]):
        raise ValueError("RD-V3 requires the identical teacher/student architecture")
    if (Path(teacher_config["data_dir"]).resolve() != Path(config["data_dir"]).resolve()
            or teacher_config.get("dataset_manifest_sha256") != config.get("dataset_manifest_sha256")):
        raise ValueError("RD-V3 must use the teacher's source data, not OOD data")
    signature = {"options": options, "teacher_sha256": sha,
                 "teacher_config_sha256": file_sha256(options["teacher_config"]),
                 "dataset_manifest_sha256": config.get("dataset_manifest_sha256"),
                 "student_model_config": config["model_config"],
                 "loss_config": config["loss_config"], "version": 1}
    if saved_metadata is not None:
        for key, value in signature.items():
            if saved_metadata.get(key) != value:
                raise ValueError(f"RD-V3 checkpoint protocol mismatch: {key}")
    elif train_dataset is None:
        raise ValueError("RD-V3 eval/resume requires saved calibration; do not recalibrate on val/test")
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        teacher = get_model(config["dataloader"], teacher_config["model_config"]).to(device)
        state = torch.load(teacher_path, map_location="cpu", weights_only=False)
        teacher.load_state_dict(state["model"], strict=True)
        teacher.eval().requires_grad_(False)
        epoch = int(state["epoch"])
        teacher.set_decoder_epoch(epoch)
        del state
        calibration = (copy.deepcopy(saved_metadata["calibration"]) if saved_metadata is not None
                       else calibrate_teacher(teacher, train_dataset, config["batch_size"], options, device))
    metadata = {**signature, "calibration": calibration, "teacher_epoch": epoch,
                "teacher_checkpoint": str(teacher_path),
                "student_initialization": "independent random encoder/decoder; own data-initialized EMA VQ",
                "targets": "per-fragment raw pre-VQ content and raw style; teacher eval mode"}
    return RepresentationDistillationLoss(config["loss_config"], config["model_config"], teacher, metadata)
