"""
Pure engineering code for training.
The only algorithmic part is the optimizer.
"""

import os
import sys
import datetime
import logging
import yaml
from importlib import import_module

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler

from utils.training_utils import *
from utils.eval_utils import *
from utils.loss_logging import LossLogger
from model.factory import get_model


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

    def prepare_data(self):
        """
        Load the data.
        """
        config = self.config
        dataloader_module = import_module("dataloader." + config["dataloader"])

        self.S_LIST = dataloader_module.S_LIST
        self.C_LIST = dataloader_module.C_LIST

        self.data_dir = config["data_dir"]
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
        logging.info("Model set up.")
        logging.info(self.model.get_model_size())

        # precision
        self.scaler = GradScaler()

        # optimizer
        if optimizer_config["optimizer"] == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
                weight_decay=optimizer_config["weight_decay"],
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
        if "load_checkpoint" in self.config and self.config["load_checkpoint"]:
            cp_path = self.config["load_checkpoint"]
            if os.path.exists(cp_path):
                save_info = torch.load(cp_path)
                self.start_epoch = save_info["epoch"]
                self.model.load_state_dict(save_info["model"])
                self.optimizer.load_state_dict(save_info["optimizer"])
                logging.info(
                    f"Checkpoint loaded from {cp_path} at epoch {self.start_epoch}."
                )
            else:
                logging.info(
                    f"No checkpoint found at {cp_path}. Will start from scratch."
                )

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
        logging.info(f"Scheduler {optimizer_config['scheduler']} set up.")

    def train(self):
        """
        It turns out having a Lightning-like style. But I hope I can make myself clear and aware of what it is doing.
        """
        config = self.config
        n_epochs = config["epochs"]
        global_step = self.start_epoch * len(self.train_loader)
        for epoch in range(self.start_epoch, self.start_epoch + n_epochs):
            # training loop
            self.model.train()
            running_losses_train = {}
            running_count_train = 0
            epoch_losses_train = {}
            epoch_count_train = 0
            for i, (batch_data, c_labels, s_labels) in enumerate(self.train_loader):
                # Move data to device
                batch_data = batch_data.to(device=self.device)

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
                    losses = self.loss.compute_loss(
                        outputs,
                        emb_c,
                        emb_c_vq,
                        vq_commit_loss,
                        emb_s,
                        batch_data,
                    )
                # backward
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(losses["total_loss"]).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                global_step += 1
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
                # write to log
                is_log_step = (i + 1) % config["log_every_n_steps"] == 0
                is_last_step = i == len(self.train_loader) - 1
                if is_log_step or is_last_step:
                    mean_losses_train = LossLogger.mean(
                        running_losses_train, running_count_train
                    )
                    logging.info(
                        f"TRAIN - Epoch [{epoch}/{n_epochs}], Step [{i}/{len(self.train_loader)}], "
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
            self.loss_logger.log_epoch(
                "train",
                epoch,
                global_step,
                mean_epoch_losses_train,
                self.optimizer.param_groups[0]["lr"],
            )

            # validation loop
            running_losses_val = None
            with torch.no_grad():
                if epoch % config["val_every_n_epochs"] == 0:
                    self.model.eval()
                    running_losses_val = {}
                    sample_vq_indices = []
                    for i, (batch_data, c_labels, s_labels) in enumerate(
                        self.val_loader
                    ):
                        # Move data to device
                        batch_data = batch_data.to(device=self.device)
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
                        losses = self.loss.compute_loss(
                            outputs,
                            emb_c,
                            emb_c_vq,
                            vq_commit_loss,
                            emb_s,
                            batch_data,
                        )
                        # accumulate running loss
                        for k, v in losses.items():
                            if k not in running_losses_val:
                                running_losses_val[k] = 0
                            running_losses_val[k] += v.item()
                        # check vq indices
                        vq_indices = vq_indices.detach().cpu().numpy()
                        sample_vq_indices.append(vq_indices)

                    # write to log
                    for k, v in running_losses_val.items():
                        running_losses_val[k] /= len(self.val_loader)
                    logging.info(
                        f"VALIDATION - Epoch [{epoch}/{n_epochs}], "
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

                    # write summary for this validation cycle
                    if config["wandb"]:
                        self._write_summary(
                            global_step,
                            epoch,
                            running_losses_val,
                            "val",
                        )

            plot_every_n_epochs = config.get(
                "plot_every_n_epochs", config["val_every_n_epochs"]
            )
            if epoch % plot_every_n_epochs == 0:
                try:
                    self.loss_logger.plot()
                except Exception:
                    logging.exception("Failed to update loss curves.")

            # save checkpoint
            if epoch % config["save_every_n_epochs"] == 0:
                if running_losses_val is None:
                    logging.warning(
                        "Skipping checkpoint at epoch %s because validation did not run.",
                        epoch,
                    )
                else:
                    self._save_checkpoint(epoch, running_losses_val["total_loss"])

            # scheduler step
            self.scheduler.step()

    def _write_summary(self, i_step, i_epoch, losses, partition="train", fig=None):
        if self.config["debug"]:
            return
        log_dict = {"epoch": i_epoch}
        for k, v in losses.items():
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

    def _save_checkpoint(self, epoch, val_loss):
        if self.config["save_top_k"] is not None:
            # keep the best checkpoints
            self.performance_history = self.performance_history or {}
            if len(self.performance_history) > self.config["save_top_k"]:
                # remove the worst checkpoint
                worst_epoch = max(
                    self.performance_history, key=self.performance_history.get
                )
                worst_path = os.path.join(self.log_dir, f"cp_epoch{worst_epoch}.pt")
                os.remove(worst_path)
                del self.performance_history[worst_epoch]

            # save the current checkpoint
            save_info = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_name = f"cp_epoch{epoch}.pt"
            save_path = os.path.join(self.log_dir, save_name)
            torch.save(save_info, save_path)
            self.performance_history[epoch] = val_loss
            logging.info(f"Checkpoint saved at {save_path}")
        else:  # directly save
            save_info = {
                "epoch": epoch,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_name = f"cp_epoch{epoch}.pt"
            save_path = os.path.join(self.log_dir, save_name)
            torch.save(save_info, save_path)
            logging.info(f"Checkpoint saved at {save_path}")
