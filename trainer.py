"""
Pure engineering code for training.
The only algorithmic part is the optimizer.
"""

import os
import sys
import datetime
import json
import hashlib
import shutil
import logging
import yaml
from importlib import import_module
from glob import glob

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler

from utils.training_utils import *
from utils.eval_utils import *
from utils.codebook_logging import CodebookMetricLogger
from utils.codebook_metrics import (
    batch_confusion_counts,
    compute_alias_geometry_metrics,
    compute_assignment_metrics,
    file_sha256,
)
from utils.checkpoint_transform import (
    expand_ema_codebook_state,
    pca_project_and_expand_ema_codebook_state,
)
from utils.loss_logging import LossLogger
from utils.resume_artifacts import inherit_resume_artifacts
from utils.disentanglement_monitor import DisentanglementMonitor
from utils.objective_schedule import apply_loss_schedules
from utils.objective_schedule_logging import ObjectiveScheduleLogger
from utils.v3_ratio_logging import V3RatioLogger
from utils.training_diagnostics import TrainingUsageMonitor, BatchNormDiagnostic
from utils.normfree_diagnostics import (
    OptimizationScaleMonitor, PerStyleCodebookMonitor, non_normalization_parameters,
)
from model.factory import get_model
from model.v3_loss import V3_RATIO_KEYS


class Trainer:
    def __init__(self, config):
        # basic configs
        self.config = config
        if self.config["debug"]:
            self.portion = self.config.get("debug_portion", 0.01)
            self.config["epochs"] = 1
            self.config["log_every_n_steps"] = 1
            self.config["val_every_n_epochs"] = 1
            self.config["save_every_n_epochs"] = 1
            self.config["save_top_k"] = 1
        else:
            self.portion = 1

        if self.config["random_seed"] is not None:
            setup_seed(self.config["random_seed"])

        # device and dtype
        if self.config["device"] == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if self.config.get("precision", "float32") == "float32":
            self.dtype = torch.float32
        elif self.config.get("precision") == "bfloat16":
            self.dtype = torch.bfloat16
        elif self.config.get("precision") == "float16":
            self.dtype = torch.float16

        # log dir
        if "name" not in config:
            config["name"] = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.name = config["name"]
        self.log_dir = os.path.join(config["log_dir"], self.name)
        os.makedirs(self.log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d_%H:%M:%S",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(self.log_dir, "log.txt")),
            ],
        )

        # backup the config file
        with open(os.path.join(self.log_dir, "config.yaml"), "w") as f:
            yaml.dump(config, f)

        # in case you want to use wandb
        if not config["debug"] and config["wandb"]:
            import wandb

            self.wandb = wandb

            if "project" in config:
                wandb.init(project=config["project"], name=self.name)
            else:
                wandb.init(project="V3", name=self.name)
            wandb.config.update(config)

        # performance history: {epoch: val_loss}
        self.performance_history = {}
        self.loss_logger = LossLogger(self.log_dir)
        self.codebook_logger = CodebookMetricLogger(self.log_dir)
        self.v3_ratio_logger = V3RatioLogger(
            self.log_dir, config["loss_config"]["relativity"]
        )
        self.objective_schedule_logger = (
            ObjectiveScheduleLogger(self.log_dir)
            if config.get("loss_schedules")
            else None
        )
        self.best_macro_atom_purity = None
        self.best_macro_val_loss = None
        self.best_macro_epoch = None
        self.best_macro_stage = None
        self.best_macro_checkpoint_path = None
        self.best_validation_loss = None
        self.best_validation_epoch = None
        self.seed_resumed_macro_checkpoint = False
        self.checkpoint_transform_metadata = None
        self.training_usage_monitor = None
        usage_config = config.get("training_usage_monitor", {})
        if usage_config.get("enabled", False):
            self.training_usage_monitor = TrainingUsageMonitor(
                self.log_dir, config["model_config"]["n_atoms"],
                usage_config.get("every_n_steps", 100))

    def prepare_data(self):
        """
        Load the data.
        """
        config = self.config
        dataloader_module = import_module("dataloader." + config["dataloader"])

        self.S_LIST = dataloader_module.S_LIST
        self.C_LIST = dataloader_module.C_LIST

        self.data_dir = config["data_dir"]
        if config.get("dataset_manifest_sha256"):
            manifest_path = os.path.abspath(os.path.join(self.data_dir, "manifest.json"))
            actual_hash = file_sha256(manifest_path)
            if actual_hash != config["dataset_manifest_sha256"]:
                raise ValueError(f"Dataset manifest checksum mismatch: {manifest_path}")
            with open(manifest_path) as handle:
                manifest = json.load(handle)
            self._write_json_atomic(os.path.join(self.log_dir, "dataset_provenance.json"),
                {"manifest": manifest_path, "manifest_sha256": actual_hash,
                 "dataset": manifest.get("dataset"), "generation_seed": manifest.get("generation_seed"),
                 "split_seed": manifest.get("split_seed"), "generator_sha256": manifest.get("generator_sha256"),
                 "page_count": manifest.get("page_count")})
            logging.info("Dataset manifest SHA256: %s", actual_hash)
        self.train_loader = dataloader_module.get_dataloader(
            os.path.join(self.data_dir, "train"),
            portion=self.portion,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            n_fragments=config["model_config"]["n_fragments"],
            fragment_len=config["model_config"]["fragment_len"],
            shuffle=True,
        )
        logging.info("Train dataloader ready.")
        self.val_loader = dataloader_module.get_dataloader(
            os.path.join(self.data_dir, "val"),
            portion=self.portion,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            n_fragments=config["model_config"]["n_fragments"],
            fragment_len=config["model_config"]["fragment_len"],
            shuffle=False,
        )
        logging.info("Validation dataloader ready.")
        self.bn_diagnostic = None
        if config.get("bn_diagnostic", {}).get("enabled", False):
            self.bn_diagnostic = BatchNormDiagnostic(
                self.log_dir, self.val_loader.dataset, config["bn_diagnostic"])
        self.disentanglement_monitor = None
        if config.get("disentanglement_probes", {}).get("enabled", False):
            if not config["loss_config"].get("monitor_raw_mpd", False):
                raise ValueError("disentanglement_probes requires monitor_raw_mpd")
            self.disentanglement_monitor = DisentanglementMonitor(
                self.log_dir, self.val_loader.dataset,
                config["disentanglement_probes"], config["model_config"]["n_atoms"], self.C_LIST,
            )
        self.per_style_codebook_monitor = None
        if config.get('per_style_codebook_monitor', {}).get('enabled', False):
            self.per_style_codebook_monitor = PerStyleCodebookMonitor(
                self.log_dir, config['model_config']['n_atoms'], self.C_LIST, self.S_LIST,
                getattr(dataloader_module, 'CONTENT_GROUPS', None))

    def build_model(self):
        """
        Set up model & optimizer & loss functions.
        Load previous model if specified.
        """
        config = self.config
        method_specs = self.config["method"].split("_")
        self.method_specs = method_specs

        model_config = self.config["model_config"]
        optimizer_config = self.config["optimizer_config"]
        loss_config = self.config["loss_config"]

        if "V3" in method_specs:
            self.model = get_model(config["dataloader"], model_config).to(self.device)
            from model.v3_loss import V3Loss as Loss
        if config.get("freeze_vq_projection", False):
            for module_name in ("project_in", "project_out"):
                module = getattr(self.model.vq, module_name)
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            logging.info("VQ project_in/project_out parameters are frozen.")
        logging.info("Model set up.")
        logging.info(self.model.get_model_size())
        if 'decoder_normalization' in model_config or 'encoder_normalization' in model_config:
            normalization = {
                'encoder': self.model.encoder_normalization,
                'encoder_converted_layers': self.model.encoder_normalization_layers,
                'decoder': self.model.decoder_normalization,
                'converted_layers': self.model.decoder_normalization_layers,
                'encoder_bn_layers': sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                                         for m in self.model.encoder.modules()),
                'decoder_bn_layers': sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                                         for m in self.model.decoder.modules()),
                'decoder_groupnorm_layers': sum(isinstance(m, torch.nn.GroupNorm)
                                                for m in self.model.decoder.modules()),
                'scope': 'explicit backbone choices before checkpoint load; defaults preserve historical BN',
            }
            self._write_json_atomic(os.path.join(self.log_dir, 'normalization_config.json'), normalization)
            logging.info('Normalization configuration: %s', normalization)
        if config.get("record_initial_model_hash", False) and not config.get("load_checkpoint"):
            digest = hashlib.sha256()
            for name, tensor in sorted(self.model.state_dict().items()):
                digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
                digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
            parameter_digest = hashlib.sha256()
            for name, tensor in sorted(self.model.named_parameters()):
                parameter_digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
                parameter_digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
            non_norm_digest = hashlib.sha256()
            for name, tensor in sorted(non_normalization_parameters(self.model)):
                non_norm_digest.update(f"{name}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
                non_norm_digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
            self._write_json_atomic(os.path.join(self.log_dir, "initial_model_state.json"),
                {"sha256": digest.hexdigest(), "seed": config["random_seed"],
                 "scope": "all parameters and buffers before any forward",
                 "parameters_sha256": parameter_digest.hexdigest(),
                 "non_normalization_parameters_sha256": non_norm_digest.hexdigest(),
                 "non_normalization_scope": "shared Conv/Linear/VQ parameters, excludes normalization affine parameters",
                 "parameter_scope": "all named parameters; excludes BN running buffers"})
            logging.info("Initial model state SHA256: %s", digest.hexdigest())

        # precision
        self.scaler = GradScaler()

        # optimizer
        if optimizer_config["optimizer"] in ("Adam", "AdamW"):
            optimizer_class = optim.Adam if optimizer_config["optimizer"] == "Adam" else optim.AdamW
            self.optimizer = optimizer_class(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                weight_decay=optimizer_config["weight_decay"],
                eps=optimizer_config.get("eps", 1e-8),
            )
        elif optimizer_config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                momentum=optimizer_config["momentum"],
                weight_decay=optimizer_config["weight_decay"],
            )
        logging.info(f"Optimizer {optimizer_config['optimizer']} set up.")

        # loss function
        self.loss = Loss(loss_config)

        # load previous model
        self.start_epoch = 0
        self.reset_training_state = False
        checkpoint_state = None
        if "load_checkpoint" in self.config and self.config["load_checkpoint"]:
            cp_path = self.config["load_checkpoint"]
            if os.path.exists(cp_path):
                checkpoint_state = torch.load(
                    cp_path, map_location=self.device, weights_only=False
                )
                completed_epoch = int(checkpoint_state["epoch"])
                self.start_epoch = completed_epoch + 1
                transform_config = self.config.get("checkpoint_transform")
                if transform_config:
                    transform_type = transform_config.get("type")
                    if transform_type == "expand_ema_codebook":
                        transformed_model, transform_metadata = (
                            expand_ema_codebook_state(
                                checkpoint_state["model"],
                                transform_config["source_atoms"],
                                transform_config["target_atoms"],
                                transform_config.get("jitter_fraction", 0.01),
                                transform_config.get("random_seed", 0),
                            )
                        )
                    elif transform_type == "pca_project_and_expand_ema_codebook":
                        transformed_model, transform_metadata = (
                            pca_project_and_expand_ema_codebook_state(
                                checkpoint_state["model"],
                                transform_config["source_atoms"],
                                transform_config["target_atoms"],
                                transform_config["target_dim"],
                                transform_config.get("jitter_fraction", 0.01),
                                transform_config.get("random_seed", 0),
                            )
                        )
                    else:
                        raise ValueError(
                            f"Unsupported checkpoint transform: {transform_type}"
                        )
                    checkpoint_state["model"] = transformed_model
                    self.reset_training_state = bool(
                        transform_config.get("reset_training_state", False)
                    )
                    if self.reset_training_state:
                        self.start_epoch = 0
                    self.checkpoint_transform_metadata = {
                        **transform_metadata,
                        "source_checkpoint": os.path.abspath(cp_path),
                        "source_checkpoint_sha256": file_sha256(cp_path),
                        "source_epoch": completed_epoch,
                        "reset_training_state": self.reset_training_state,
                    }
                self.model.load_state_dict(checkpoint_state["model"])
                if not self.reset_training_state:
                    self.optimizer.load_state_dict(checkpoint_state["optimizer"])
                if not self.reset_training_state and "scaler" in checkpoint_state:
                    self.scaler.load_state_dict(checkpoint_state["scaler"])
                if transform_config:
                    self.best_macro_atom_purity = None
                    self.best_macro_val_loss = None
                    self.best_macro_epoch = None
                    self.best_macro_stage = None
                    logging.info(
                        "Reset macro-best state after checkpoint transform."
                    )
                else:
                    if config.get("save_best_validation_loss", False):
                        self._restore_best_validation_state(checkpoint_state, cp_path)
                    self.best_macro_atom_purity = checkpoint_state.get(
                        "best_macro_atom_purity"
                    )
                    self.best_macro_val_loss = checkpoint_state.get(
                        "best_macro_val_loss"
                    )
                    self.best_macro_epoch = checkpoint_state.get("best_macro_epoch")
                    self.best_macro_stage = checkpoint_state.get("best_macro_stage")
                if (
                    self.best_macro_atom_purity is None
                    and not transform_config
                    and self.config.get("resume_macro_atom_purity") is not None
                ):
                    self.best_macro_atom_purity = float(
                        self.config["resume_macro_atom_purity"]
                    )
                    self.best_macro_val_loss = float(
                        self.config["resume_macro_val_loss"]
                    )
                    self.best_macro_epoch = completed_epoch
                    self.seed_resumed_macro_checkpoint = True
                logging.info(
                    f"Checkpoint loaded from {cp_path}; completed epoch "
                    f"{completed_epoch}, resuming at epoch {self.start_epoch}."
                )
                if self.best_macro_atom_purity is not None:
                    logging.info(
                        "Restored best macro atom purity %.6g from epoch %s.",
                        self.best_macro_atom_purity,
                        self.best_macro_epoch,
                    )
            else:
                raise FileNotFoundError(f"Requested resume checkpoint is missing: {cp_path}")

        # scheduler.
        if optimizer_config["scheduler"] == "cosine_annealing":
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda epoch: cosine_annealing_with_warmup(
                    epoch,
                    optimizer_config["lr_anneal_epochs"],
                    optimizer_config["lr_anneal_min_factor"],
                    optimizer_config["warmup_epochs"],
                    optimizer_config["warmup_factor"],
                ),
                last_epoch=self.start_epoch - 1,  # important for resuming training
            )
        elif optimizer_config["scheduler"] == "exponential_decay":
            self.scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda epoch: exponential_decay_with_warmup(
                    epoch,
                    optimizer_config["lr_decay_factor"],
                    optimizer_config["lr_decay_epochs"],
                    optimizer_config["lr_decay_min_factor"],
                    optimizer_config["warmup_epochs"],
                    optimizer_config["warmup_factor"],
                ),
                last_epoch=self.start_epoch - 1,  # important for resuming training
            )
        if (
            checkpoint_state is not None
            and not self.reset_training_state
            and "scheduler" in checkpoint_state
        ):
            self.scheduler.load_state_dict(checkpoint_state["scheduler"])
            logging.info("Scheduler state restored from checkpoint.")
        elif checkpoint_state is not None and not self.reset_training_state:
            logging.info(
                "Legacy checkpoint has no scheduler state; reconstructed it from epoch."
            )
        logging.info(f"Scheduler {optimizer_config['scheduler']} set up.")
        if config.get("inherit_resume_history", False):
            if checkpoint_state is None or self.reset_training_state or config.get("checkpoint_transform"):
                raise ValueError("inherit_resume_history requires an untransformed resume checkpoint")
            self.best_macro_checkpoint_path = inherit_resume_artifacts(
                config["load_checkpoint"], self.log_dir, checkpoint_state
            )
            logging.info("Inherited completed histories and retained macro-best into new run.")
        if self.seed_resumed_macro_checkpoint:
            self._persist_best_macro_checkpoint(self.best_macro_epoch)
            logging.info(
                "Seeded macro-best tracking from legacy resume checkpoint "
                "at epoch %s (macro=%.6g, val_loss=%.6g).",
                self.best_macro_epoch,
                self.best_macro_atom_purity,
                self.best_macro_val_loss,
            )
        self.optimization_scale_monitor = None
        if config.get('optimization_diagnostics', {}).get('enabled', False):
            self.optimization_scale_monitor = OptimizationScaleMonitor(
                self.log_dir, self.model, config['optimization_diagnostics'].get('every_n_steps', 100))
        if self.checkpoint_transform_metadata is not None:
            self._write_json_atomic(
                os.path.join(self.log_dir, "initialization.json"),
                self.checkpoint_transform_metadata,
            )
            logging.info(
                "Checkpoint transform metadata written to %s.",
                os.path.join(self.log_dir, "initialization.json"),
            )

    def train(self):
        """
        It turns out having a Lightning-like style. But I hope I can make myself clear and aware of what it is doing.
        """
        config = self.config
        n_epochs = config["epochs"]
        if n_epochs <= 0:
            raise ValueError("epochs must be a positive incremental epoch count.")
        end_epoch = self.start_epoch + n_epochs - 1
        global_step = self.start_epoch * len(self.train_loader)
        logging.info(
            "Training epoch range: %s-%s (%s incremental epochs).",
            self.start_epoch,
            end_epoch,
            n_epochs,
        )
        if config.get("validate_before_training", False):
            apply_loss_schedules(
                self.loss.config,
                config.get("loss_schedules"),
                self.start_epoch,
            )
            initialization = self._collect_validation()
            initialization_summary = {
                "evaluated_before_epoch": int(self.start_epoch),
                "source_completed_epoch": int(
                    (self.checkpoint_transform_metadata or {}).get(
                        "source_epoch", self.start_epoch - 1
                    )
                ),
                "sample_count": initialization["sample_count"],
                "fragment_count": initialization["fragment_count"],
                "losses": initialization["losses"],
                "v3_ratios": initialization["v3_ratios"],
                "codebook_metrics": initialization["codebook_metrics"],
                "objective": apply_loss_schedules(
                    self.loss.config,
                    config.get("loss_schedules"),
                    self.start_epoch,
                ),
            }
            gate = config.get("initialization_health_gate")
            gate_failures = []
            if gate:
                gate_failures.extend(
                    self._gate_failures(
                        initialization["codebook_metrics"], gate.get("metrics", {})
                    )
                )
                gate_failures.extend(
                    self._gate_failures(
                        initialization["losses"], gate.get("losses", {}), "losses."
                    )
                )
            initialization_summary["health_gate"] = {
                "passed": not gate_failures,
                "failures": gate_failures,
                "config": gate,
            }
            self._write_json_atomic(
                os.path.join(self.log_dir, "initialization_metrics.json"),
                initialization_summary,
            )
            logging.info(
                "INITIALIZATION VALIDATION - before epoch %s | total_loss=%.6g, "
                "macro_atom_purity=%.6g, active_codes=%s, perplexity=%.6g, coverage=%s",
                self.start_epoch,
                initialization["losses"]["total_loss"],
                initialization["codebook_metrics"]["macro_atom_purity"],
                initialization["codebook_metrics"]["active_codes"],
                initialization["codebook_metrics"]["usage_perplexity"],
                initialization["codebook_metrics"]["dominant_label_coverage"],
            )
            if gate_failures:
                raise RuntimeError(
                    "Initialization health gate failed: " + "; ".join(gate_failures)
                )
            if config.get("save_best_macro_atom_purity", False):
                self._save_best_macro_checkpoint(
                    -1,
                    initialization["losses"]["total_loss"],
                    initialization["codebook_metrics"]["macro_atom_purity"],
                    metrics=initialization["codebook_metrics"],
                    stage="initialization",
                )
        for epoch in range(self.start_epoch, end_epoch + 1):
            self.model.set_decoder_epoch(epoch)
            logging.info("DECODER - Epoch [%s/%s] | style_mode=%s", epoch, end_epoch,
                         self.model.active_decoder_style_mode)
            objective_values = apply_loss_schedules(
                self.loss.config,
                config.get("loss_schedules"),
                epoch,
            )
            if self.objective_schedule_logger is not None:
                self.objective_schedule_logger.log_epoch(
                    epoch, global_step, objective_values
                )
                logging.info(
                    "OBJECTIVE - Epoch [%s/%s] | relativity=%.6g, "
                    "recon_weight=%.6g, commit_weight=%.6g",
                    epoch,
                    end_epoch,
                    objective_values["relativity"],
                    objective_values["recon_loss_weight"],
                    objective_values["commit_loss_weight"],
                )
                if config["wandb"]:
                    self._write_summary(
                        global_step, epoch, objective_values, "objective"
                    )
            # training loop
            self.model.train()
            usage_monitor = self.training_usage_monitor
            scale_monitor = getattr(self, 'optimization_scale_monitor', None)
            if scale_monitor is not None:
                scale_monitor.begin(epoch)
            if usage_monitor is not None:
                usage_monitor.begin(epoch, self.device)
            running_losses_train = {}
            running_count_train = 0
            epoch_losses_train = {}
            epoch_count_train = 0
            epoch_v3_ratios_train = {}
            for i, (batch_data, c_labels, s_labels) in enumerate(self.train_loader):
                # Move data to device
                batch_data = batch_data.to(device=self.device)
                if scale_monitor is not None:
                    scale_monitor.before_forward(global_step + 1, i == len(self.train_loader) - 1)

                with torch.autocast(self.device.type, dtype=self.dtype):
                    # forward
                    (
                        outputs,
                        emb_c,
                        emb_c_vq,
                        vq_indices,
                        vq_commit_loss,
                        emb_s,
                        *rest,
                    ) = self.model(batch_data)

                    # loss
                    loss_outputs = self.loss.compute_loss(
                        outputs,
                        emb_c,
                        emb_c_vq,
                        vq_commit_loss,
                        emb_s,
                        batch_data,
                    )
                    v3_ratios = {
                        name: loss_outputs.pop(name) for name in V3_RATIO_KEYS
                    }
                    losses = loss_outputs
                # backward
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(losses["total_loss"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                global_step += 1
                if scale_monitor is not None:
                    scale_row = scale_monitor.collect(global_step, emb_c, emb_c_vq, emb_s)
                    if scale_row is not None:
                        logging.info('OPTIMIZATION SCALES - %s', scale_row)
                        if config['wandb']:
                            self._write_summary(global_step, epoch, scale_row, 'train_scales')
                if usage_monitor is not None:
                    usage_monitor.collect(vq_indices, global_step)
                # accumulate running loss
                for k, v in losses.items():
                    if k not in running_losses_train:
                        running_losses_train[k] = 0
                    if k not in epoch_losses_train:
                        epoch_losses_train[k] = 0
                    loss_value = v.item()
                    running_losses_train[k] += loss_value
                    epoch_losses_train[k] += loss_value
                running_count_train += 1
                epoch_count_train += 1
                for name, value in v3_ratios.items():
                    epoch_v3_ratios_train[name] = (
                        epoch_v3_ratios_train.get(name, 0.0) + value.item()
                    )
                # write to log
                is_log_step = (i + 1) % config["log_every_n_steps"] == 0
                is_last_step = i == len(self.train_loader) - 1
                if is_log_step or is_last_step:
                    mean_losses_train = LossLogger.mean(
                        running_losses_train, running_count_train
                    )
                    logging.info(
                        f"TRAIN - Epoch [{epoch}/{end_epoch}], Step [{i}/{len(self.train_loader)}], "
                        f"Loss: {mean_losses_train['total_loss']:.4f} | "
                        f"{LossLogger.format_losses(mean_losses_train)}"
                    )
                    self.loss_logger.log_step(
                        "train",
                        epoch,
                        global_step,
                        i,
                        mean_losses_train,
                        self.optimizer.param_groups[0]["lr"],
                    )
                    # write summary for this log cycle
                    if config["wandb"]:
                        self._write_summary(
                            global_step, epoch, mean_losses_train, "train"
                        )
                    running_losses_train = {}
                    running_count_train = 0

            mean_epoch_losses_train = LossLogger.mean(
                epoch_losses_train, epoch_count_train
            )
            if usage_monitor is not None:
                usage_row = usage_monitor.finish(global_step)
                logging.info("ONLINE TRAINING USAGE - %s", usage_row)
                if config["wandb"]:
                    self._write_summary(global_step, epoch, usage_row, "train_online_usage")
            if scale_monitor is not None:
                scale_monitor.finish(global_step)
            self.loss_logger.log_epoch(
                "train",
                epoch,
                global_step,
                mean_epoch_losses_train,
                self.optimizer.param_groups[0]["lr"],
            )
            mean_epoch_v3_ratios_train = LossLogger.mean(
                epoch_v3_ratios_train, epoch_count_train
            )
            self.v3_ratio_logger.log_epoch(
                "train",
                epoch,
                global_step,
                mean_epoch_v3_ratios_train,
            )
            if config["wandb"]:
                self._write_summary(
                    global_step,
                    epoch,
                    mean_epoch_v3_ratios_train,
                    "train_v3_ratio",
                )

            # Validation reuses the full forward pass to collect codebook metrics.
            running_losses_val = None
            codebook_metrics_val = None
            if epoch % config["val_every_n_epochs"] == 0:
                validation = self._collect_validation(epoch, global_step)
                running_losses_val = validation["losses"]
                validation_v3_ratios = validation["v3_ratios"]
                codebook_metrics_val = validation["codebook_metrics"]
                validation_sample_count = validation["sample_count"]
                validation_fragment_count = validation["fragment_count"]
                logging.info(
                    f"VALIDATION - Epoch [{epoch}/{end_epoch}], "
                    f"Loss: {running_losses_val['total_loss']:.4f} | "
                    f"{LossLogger.format_losses(running_losses_val)}"
                )
                self.loss_logger.log_epoch(
                    "val",
                    epoch,
                    global_step,
                    running_losses_val,
                    self.optimizer.param_groups[0]["lr"],
                )
                self.v3_ratio_logger.log_epoch(
                    "val",
                    epoch,
                    global_step,
                    validation_v3_ratios,
                )
                logging.info(
                    "VALIDATION V3 RATIOS - Epoch [%s/%s] | %s",
                    epoch,
                    end_epoch,
                    ", ".join(
                        f"{name}={validation_v3_ratios[name]:.6g}"
                        for name in V3_RATIO_KEYS
                    ),
                )
                self.codebook_logger.log_epoch(
                    "val",
                    epoch,
                    global_step,
                    codebook_metrics_val,
                    validation_sample_count,
                    validation_fragment_count,
                )
                logging.info(
                    "VALIDATION CODEBOOK - Epoch [%s/%s] | "
                    "one_to_one_accuracy=%.6g, legacy_codebook_accuracy=%.6g, "
                    "macro_atom_purity=%.6g, codebook_purity=%.6g, active_codes=%s, "
                    "usage_perplexity=%.6g, dominant_label_coverage=%s, "
                    "dominant_codes[min,max,cv]=[%s,%s,%.6g]",
                    epoch,
                    end_epoch,
                    codebook_metrics_val["one_to_one_accuracy"],
                    codebook_metrics_val["legacy_codebook_accuracy"],
                    codebook_metrics_val["macro_atom_purity"],
                    codebook_metrics_val["codebook_purity"],
                    codebook_metrics_val["active_codes"],
                    codebook_metrics_val["usage_perplexity"],
                    codebook_metrics_val["dominant_label_coverage"],
                    codebook_metrics_val["dominant_label_code_count_min"],
                    codebook_metrics_val["dominant_label_code_count_max"],
                    codebook_metrics_val["dominant_label_code_count_cv"],
                )

                # write summary for this validation cycle
                if config["wandb"]:
                    self._write_summary(
                        global_step,
                        epoch,
                        running_losses_val,
                        "val",
                    )
                    self._write_summary(
                        global_step,
                        epoch,
                        codebook_metrics_val,
                        "val_codebook",
                    )
                    self._write_summary(
                        global_step,
                        epoch,
                        validation_v3_ratios,
                        "val_v3_ratio",
                    )

            plot_every_n_epochs = config.get(
                "plot_every_n_epochs", config["val_every_n_epochs"]
            )
            if epoch % plot_every_n_epochs == 0:
                try:
                    self.loss_logger.plot()
                except Exception:
                    logging.exception("Failed to update loss curves.")
                try:
                    self.codebook_logger.plot()
                except Exception:
                    logging.exception("Failed to update codebook metric curves.")
                try:
                    self.v3_ratio_logger.plot()
                except Exception:
                    logging.exception("Failed to update V3 ratio curves.")
                if self.objective_schedule_logger is not None:
                    try:
                        self.objective_schedule_logger.plot()
                    except Exception:
                        logging.exception("Failed to update objective schedule curves.")

            # Advance before checkpointing so a resumed checkpoint contains the
            # learning rate and scheduler state for the next epoch.
            self.scheduler.step()

            best_val_changed = False
            if codebook_metrics_val is not None and config.get("save_best_validation_loss", False):
                best_val_changed = self._consider_best_validation_loss(epoch, running_losses_val["total_loss"])

            if (
                codebook_metrics_val is not None
                and config.get("save_best_macro_atom_purity", False)
            ):
                self._save_best_macro_checkpoint(
                    epoch,
                    running_losses_val["total_loss"],
                    codebook_metrics_val["macro_atom_purity"],
                    metrics=codebook_metrics_val,
                )

            if best_val_changed:
                self._persist_best_validation_checkpoint(epoch)

            # Keep exactly one current checkpoint; macro-best is maintained
            # independently above.
            if epoch % config["save_every_n_epochs"] == 0:
                if running_losses_val is None:
                    logging.warning(
                        "Skipping checkpoint at epoch %s because validation did not run.",
                        epoch,
                    )
                else:
                    checkpoint_policy = config.get(
                        "checkpoint_policy", "current_and_best_macro"
                    )
                    if checkpoint_policy != "current_and_best_macro":
                        raise ValueError(
                            f"Unsupported checkpoint_policy: {checkpoint_policy}"
                        )
                    self._save_current_checkpoint(
                        epoch,
                        running_losses_val["total_loss"],
                        codebook_metrics_val["macro_atom_purity"],
                    )

            snapshot_every = int(config.get("snapshot_every_n_epochs", 0))
            if snapshot_every > 0 and (epoch + 1) % snapshot_every == 0:
                self._save_periodic_checkpoint(epoch)

    def _save_periodic_checkpoint(self, epoch):
        snapshot_path = os.path.join(self.log_dir, f"cp_snapshot_epoch{epoch}.pt")
        torch.save(self._checkpoint_state(epoch), snapshot_path + ".tmp")
        os.replace(snapshot_path + ".tmp", snapshot_path)
        logging.info("Permanent periodic checkpoint saved at %s", snapshot_path)

    def _collect_validation(self, epoch=None, global_step=0):
        config = self.config
        running_losses = {}
        running_v3_ratios = {}
        confusion_counts = torch.zeros(
            (config["model_config"]["n_atoms"], len(self.C_LIST)),
            dtype=torch.int64,
            device=self.device,
        )
        sample_count = 0
        monitor = getattr(self, "disentanglement_monitor", None) if epoch is not None else None
        bn_diagnostic = getattr(self, "bn_diagnostic", None) if epoch is not None else None
        style_monitor = getattr(self, 'per_style_codebook_monitor', None) if epoch is not None else None
        if style_monitor is not None:
            style_monitor.begin(epoch, global_step, self.device)
        if bn_diagnostic is not None:
            bn_diagnostic.begin(epoch, global_step)
        if monitor is not None:
            monitor.begin(epoch, global_step, self.model.active_decoder_style_mode)
        self.model.eval()
        with torch.inference_mode():
            for batch_data, c_labels, s_labels in self.val_loader:
                batch_data = batch_data.to(device=self.device, non_blocking=True)
                if bn_diagnostic is not None:
                    bn_diagnostic.collect(batch_data, c_labels, s_labels)
                sample_count += int(batch_data.shape[0])
                (
                    outputs,
                    emb_c,
                    emb_c_vq,
                    vq_indices,
                    vq_commit_loss,
                    emb_s,
                    *rest,
                ) = self.model(batch_data, freeze_codebook=True)
                if style_monitor is not None:
                    style_monitor.collect(vq_indices, c_labels, s_labels)
                loss_outputs = self.loss.compute_loss(
                    outputs,
                    emb_c,
                    emb_c_vq,
                    vq_commit_loss,
                    emb_s,
                    batch_data,
                )
                if monitor is not None:
                    monitor.collect(batch_data, outputs, emb_c, emb_s, vq_indices,
                                    c_labels, s_labels, self.loss.last_statistics)
                ratios = {name: loss_outputs.pop(name) for name in V3_RATIO_KEYS}
                for name, value in loss_outputs.items():
                    running_losses[name] = running_losses.get(name, 0.0) + value.item()
                for name, value in ratios.items():
                    running_v3_ratios[name] = (
                        running_v3_ratios.get(name, 0.0) + value.item()
                    )
                confusion_counts += batch_confusion_counts(
                    vq_indices,
                    c_labels,
                    config["model_config"]["n_atoms"],
                    len(self.C_LIST),
                )

        batch_count = len(self.val_loader)
        losses = {name: value / batch_count for name, value in running_losses.items()}
        ratios = {
            name: value / batch_count for name, value in running_v3_ratios.items()
        }
        confusion_counts_cpu = confusion_counts.cpu().numpy()
        if style_monitor is not None:
            for row in style_monitor.finish(confusion_counts_cpu):
                logging.info('VALIDATION PER-STYLE CODEBOOK - %s', row)
                if config['wandb']:
                    self._write_summary(global_step, epoch, row, f"val_style_{row['style']}")
            for row in style_monitor.group_rows:
                logging.info('VALIDATION CONTENT GROUP - %s', row)
                if config['wandb']:
                    self._write_summary(global_step, epoch, row, f"val_content_{row['group']}")
        if bn_diagnostic is not None:
            for row in bn_diagnostic.finish(self.model, confusion_counts_cpu):
                logging.info("VALIDATION BN DIAGNOSTIC - %s", row)
                if config["wandb"]:
                    self._write_summary(global_step, epoch, row, f"val_bn_{row['mode']}")
        if monitor is not None:
            probe_metrics = monitor.finish(confusion_counts_cpu, self.device)
            logging.info("VALIDATION PROBES - Epoch %s | %s", epoch, probe_metrics)
            if config["wandb"]:
                self._write_summary(global_step, epoch, probe_metrics, "val_probe")
        metrics = {
            **compute_assignment_metrics(confusion_counts_cpu),
            **compute_alias_geometry_metrics(
                confusion_counts_cpu, self.model.vq.codebook.detach().cpu()
            ),
        }
        return {
            "losses": losses,
            "v3_ratios": ratios,
            "codebook_metrics": metrics,
            "sample_count": sample_count,
            "fragment_count": int(confusion_counts.sum().item()),
        }

    def _write_summary(self, i_step, i_epoch, losses, partition="train", fig=None):
        if self.config["debug"]:
            return
        log_dict = {"epoch": i_epoch}
        for k, v in losses.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                log_dict[f"{partition}/{k}"] = v
        if partition == "val":
            log_dict["lr"] = self.optimizer.param_groups[0]["lr"]
        self.wandb.log(log_dict, step=i_step)
        if fig is not None:
            if isinstance(fig, list):
                for i, f in enumerate(fig):
                    if f is not None:
                        self.wandb.log(
                            {f"{partition}/fig_{i}": self.wandb.Image(f)}, step=i_step
                        )
            elif isinstance(fig, np.ndarray):
                self.wandb.log({f"{partition}/fig": self.wandb.Image(fig)}, step=i_step)

    def _checkpoint_state(self, epoch):
        return {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_macro_atom_purity": self.best_macro_atom_purity,
            "best_macro_val_loss": self.best_macro_val_loss,
            "best_macro_epoch": self.best_macro_epoch,
            "best_macro_stage": getattr(self, "best_macro_stage", None),
            "best_validation_loss": getattr(self, "best_validation_loss", None),
            "best_validation_epoch": getattr(self, "best_validation_epoch", None),
        }

    def _consider_best_validation_loss(self, epoch, val_loss):
        value = float(val_loss)
        previous = getattr(self, "best_validation_loss", None)
        if not np.isfinite(value) or (previous is not None and value >= previous):
            return False
        self.best_validation_loss, self.best_validation_epoch = value, int(epoch)
        return True

    def _best_validation_metadata(self):
        return {"epoch": self.best_validation_epoch, "validation_total_loss": self.best_validation_loss,
                "checkpoint": f"cp_best_validation_loss_epoch{self.best_validation_epoch}.pt"}

    def _persist_best_validation_checkpoint(self, epoch):
        metadata = self._best_validation_metadata()
        path = os.path.join(self.log_dir, metadata["checkpoint"])
        torch.save(self._checkpoint_state(epoch), path + ".tmp")
        os.replace(path + ".tmp", path)
        self._write_json_atomic(os.path.join(self.log_dir, "best_validation_loss.json"), metadata)
        for old in glob(os.path.join(self.log_dir, "cp_best_validation_loss_epoch*.pt")):
            if os.path.abspath(old) != os.path.abspath(path):
                os.remove(old)
        logging.info("Best validation-loss checkpoint saved at %s (loss=%.6g)", path, self.best_validation_loss)

    def _restore_best_validation_state(self, state, source_checkpoint):
        self.best_validation_loss = state.get("best_validation_loss")
        self.best_validation_epoch = state.get("best_validation_epoch")
        if self.best_validation_loss is None:
            self.best_validation_epoch = None
            return
        if not np.isfinite(self.best_validation_loss) or self.best_validation_epoch is None:
            raise ValueError("Invalid best-validation state in resume checkpoint")
        metadata = self._best_validation_metadata()
        source = os.path.join(os.path.dirname(os.path.abspath(source_checkpoint)), metadata["checkpoint"])
        target = os.path.abspath(os.path.join(self.log_dir, metadata["checkpoint"]))
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Cannot restore retained best-validation weights: {source}")
        if source != target:
            if os.path.exists(target) and file_sha256(target) != file_sha256(source):
                raise FileExistsError(f"Refusing to replace different retained best-validation weights: {target}")
            shutil.copy2(source, target)
        self._write_json_atomic(os.path.join(self.log_dir, "best_validation_loss.json"), metadata)

    @staticmethod
    def _write_json_atomic(path, payload):
        temporary_path = path + ".tmp"
        with open(temporary_path, "w") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        os.replace(temporary_path, path)

    @staticmethod
    def _is_better_macro_checkpoint(
        macro_atom_purity,
        val_loss,
        best_macro_atom_purity,
        best_macro_val_loss,
    ):
        if best_macro_atom_purity is None:
            return True
        tolerance = 1e-12
        if macro_atom_purity > best_macro_atom_purity + tolerance:
            return True
        return (
            abs(macro_atom_purity - best_macro_atom_purity) <= tolerance
            and (best_macro_val_loss is None or val_loss < best_macro_val_loss)
        )

    @staticmethod
    def _gate_failures(values, gate, prefix=""):
        failures = []
        for name, threshold in gate.get("min", {}).items():
            value = values.get(name)
            if value is None or not np.isfinite(float(value)) or float(value) < float(threshold):
                failures.append(f"{prefix}{name}={value} < {threshold}")
        for name, threshold in gate.get("max", {}).items():
            value = values.get(name)
            if value is None or not np.isfinite(float(value)) or float(value) > float(threshold):
                failures.append(f"{prefix}{name}={value} > {threshold}")
        return failures

    def _save_best_macro_checkpoint(
        self, epoch, val_loss, macro_atom_purity, metrics=None, stage="training"
    ):
        macro_atom_purity = float(macro_atom_purity)
        val_loss = float(val_loss)
        health_gate = getattr(self, "config", {}).get("macro_best_health_gate")
        if health_gate and metrics is not None:
            failures = self._gate_failures(metrics, health_gate)
            if failures:
                logging.info(
                    "Macro-best candidate at epoch %s is ineligible: %s",
                    epoch,
                    "; ".join(failures),
                )
                return
        if not self._is_better_macro_checkpoint(
            macro_atom_purity,
            val_loss,
            self.best_macro_atom_purity,
            self.best_macro_val_loss,
        ):
            return

        self.best_macro_atom_purity = macro_atom_purity
        self.best_macro_val_loss = val_loss
        self.best_macro_epoch = int(epoch)
        self.best_macro_stage = stage

        self._persist_best_macro_checkpoint(epoch)

    def _persist_best_macro_checkpoint(self, epoch):

        epoch_label = "init" if int(epoch) < 0 else str(epoch)
        save_name = f"cp_best_macro_atom_purity_epoch{epoch_label}.pt"
        save_path = os.path.join(self.log_dir, save_name)
        temporary_path = save_path + ".tmp"
        torch.save(self._checkpoint_state(epoch), temporary_path)
        os.replace(temporary_path, save_path)
        self.best_macro_checkpoint_path = save_path

        metadata = {
            "epoch": int(epoch),
            "macro_atom_purity": self.best_macro_atom_purity,
            "validation_total_loss": self.best_macro_val_loss,
            "checkpoint": save_name,
            "stage": self.best_macro_stage,
        }
        metadata_path = os.path.join(self.log_dir, "best_macro_atom_purity.json")
        metadata_temporary_path = metadata_path + ".tmp"
        with open(metadata_temporary_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
        os.replace(metadata_temporary_path, metadata_path)

        for old_path in glob(
            os.path.join(self.log_dir, "cp_best_macro_atom_purity_epoch*.pt")
        ):
            if os.path.abspath(old_path) != os.path.abspath(save_path):
                os.remove(old_path)
        logging.info(
            "Best macro atom purity checkpoint saved at %s "
            "(macro=%.6g, val_loss=%.6g).",
            save_path,
            self.best_macro_atom_purity,
            self.best_macro_val_loss,
        )

    def _save_current_checkpoint(self, epoch, val_loss, macro_atom_purity):
        save_name = f"cp_current_epoch{epoch}.pt"
        save_path = os.path.join(self.log_dir, save_name)
        temporary_path = save_path + ".tmp"
        torch.save(self._checkpoint_state(epoch), temporary_path)
        os.replace(temporary_path, save_path)

        metadata = {
            "epoch": int(epoch),
            "macro_atom_purity": float(macro_atom_purity),
            "validation_total_loss": float(val_loss),
            "checkpoint": save_name,
        }
        metadata_path = os.path.join(self.log_dir, "current_checkpoint.json")
        metadata_temporary_path = metadata_path + ".tmp"
        with open(metadata_temporary_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
        os.replace(metadata_temporary_path, metadata_path)

        for old_path in glob(os.path.join(self.log_dir, "cp_current_epoch*.pt")):
            if os.path.abspath(old_path) != os.path.abspath(save_path):
                os.remove(old_path)
        logging.info("Current checkpoint saved at %s", save_path)
