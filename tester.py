import os
from importlib import import_module
from copy import deepcopy
from glob import glob

import numpy as np
import torch
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from utils.eval_utils import *
from model.factory import get_model


plt.rc("font", family="Times New Roman")


class Tester:
    def __init__(self, config):
        """
        config: a dict of config read from a yaml file
        """
        super(Tester, self).__init__()
        self.config = config
        if self.config["debug"]:
            self.portion = 0.1
        else:
            self.portion = 1

        self.output_dir = os.path.dirname(config["active_checkpoint"])

        # device
        self.device = torch.device("cpu")
        if self.config["device"] == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.config["device"] == "cuda":
            print("CUDA is not available; falling back to CPU.")

    def prepare_data(self):
        config = self.config
        dataloader_module = import_module("dataloader." + config["dataloader"])

        self.C_LIST = dataloader_module.C_LIST
        self.S_LIST = dataloader_module.S_LIST

        if os.path.exists(os.path.join(config["data_dir"], "test")):
            self.data_dir = os.path.join(config["data_dir"], "test")
        elif os.path.exists(os.path.join(config["data_dir"], "test.hdf5")):
            self.data_dir = os.path.join(config["data_dir"], "test.hdf5")
        elif os.path.exists(os.path.join(config["data_dir"], "val")):
            self.data_dir = os.path.join(config["data_dir"], "val")
        elif os.path.exists(os.path.join(config["data_dir"], "val.hdf5")):
            self.data_dir = os.path.join(config["data_dir"], "val.hdf5")

        self.test_loader = dataloader_module.get_dataloader(
            self.data_dir,
            portion=self.portion,
            batch_size=config["batch_size"],
            n_fragments=config["model_config"]["n_fragments"],
            fragment_len=config["model_config"]["fragment_len"],
            shuffle=False,
        )
        self._apply_test_subset()

        # for totally continuous styles
        if len(self.S_LIST) == 0:
            self.CONTINUOUS_STYLE = True
        else:
            self.CONTINUOUS_STYLE = False

    def _apply_test_subset(self):
        subset_size = self.config.get("test_subset_size")
        if subset_size is None or subset_size == "None":
            return

        subset_size = int(subset_size)
        if subset_size <= 0:
            raise ValueError("test_subset_size must be a positive integer.")

        dataset = self.test_loader.dataset
        dataset_size = len(dataset)
        actual_size = min(subset_size, dataset_size)

        seed = self.config.get("test_subset_seed", self.config.get("random_seed"))
        if seed is None or seed == "None":
            seed = 0

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        indices = torch.randperm(dataset_size, generator=generator)[:actual_size]
        indices = sorted(indices.tolist())

        loader_kwargs = {
            "batch_size": self.test_loader.batch_size or self.config["batch_size"],
            "shuffle": False,
            "num_workers": self.test_loader.num_workers,
            "collate_fn": self.test_loader.collate_fn,
            "pin_memory": self.test_loader.pin_memory,
            "drop_last": False,
        }
        if self.test_loader.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.test_loader.prefetch_factor
            loader_kwargs["persistent_workers"] = self.test_loader.persistent_workers

        self.test_loader = DataLoader(Subset(dataset, indices), **loader_kwargs)
        print(
            f"Evaluating on {actual_size}/{dataset_size} test samples "
            f"(test_subset_seed={seed})."
        )

    def build_model(self):
        config = self.config
        method_specs = self.config["method"].split("_")
        self.method_specs = method_specs

        model_config = self.config["model_config"]
        loss_config = self.config["loss_config"]

        if "V3" in method_specs:
            self.model = get_model(config["dataloader"], model_config).to(self.device)
            from model.v3_loss import V3Loss as Loss

        cp_state_dict = torch.load(
            config["active_checkpoint"], map_location=self.device
        )["model"]

        self.model.load_state_dict(cp_state_dict, strict=False)
        self.model.eval()

        self.loss = Loss(loss_config)

    def test(
        self,
        pr_metrics=False,
        vis_tsne=False,
        confusion_mtx=False,
        zero_shot_ood=False,
        few_shot_ood=False,
    ):
        self.codebook = self.model.vq.codebook.detach().cpu().numpy()
        self.ground_truth = []  # (input, content_idx, style_idx)
        self.results = []  # (recon, emb_c, emb_c_vq, emb_s)
        self.sample_vq_indices = []  # vq_indices
        self.confusion_counts = np.zeros(
            (self.config["model_config"]["n_atoms"], len(self.C_LIST))
        )
        keep_full_outputs = pr_metrics or vis_tsne

        n_rounds = (
            5 if self.config["model_config"]["n_fragments"] < len(self.C_LIST) else 1
        )  # to make sure more things are covered, as dataloaders might not use all fragments
        for round in range(n_rounds):
            for i, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                batch_data, content_idx, style_idx = batch
                batch_data = batch_data.to(self.device)

                with torch.no_grad():
                    recon, emb_c, emb_c_vq, vq_indices, vq_commit_loss, emb_s, *rest = (
                        self.model(batch_data)
                    )

                # detach everything
                content_idx_np = content_idx.detach().cpu().numpy()
                vq_indices = vq_indices.detach().cpu().numpy()
                for atom_idx, content_label in zip(
                    vq_indices.reshape(-1), content_idx_np.reshape(-1)
                ):
                    self.confusion_counts[int(atom_idx), int(content_label)] += 1

                if not keep_full_outputs:
                    continue

                batch_data = batch_data.detach().cpu().numpy()
                if not self.CONTINUOUS_STYLE:
                    style_idx = style_idx.detach().cpu().numpy()
                else:  # continuous-style dataloaders don't give a full batch of style_idx, so we need to generate it
                    style_idx = list(style_idx)
                    style_idx = [
                        [x for fi in range(self.config["model_config"]["n_fragments"])]
                        for x in style_idx
                    ]
                recon = recon.detach().cpu().numpy()
                emb_c = emb_c.detach().cpu().numpy()
                emb_c_vq = emb_c_vq.detach().cpu().numpy()
                emb_s = emb_s.detach().cpu().numpy()
                for j in range(emb_c.shape[0]):  # for every sample in the batch
                    for k in range(emb_c.shape[1]):  # for every fragment in the sample
                        self.ground_truth.append(
                            (
                                batch_data[j, k],
                                content_idx[j][k],
                                style_idx[j][k],
                            )
                        )
                        self.results.append(
                            (
                                recon[j, k],
                                emb_c[j, k],
                                emb_c_vq[j, k],
                                emb_s[j, k],
                            )
                        )
                    self.sample_vq_indices.append(vq_indices[j])

        if pr_metrics:
            print(
                f"Computing PR metrics for {len(self.results)} fragments. "
                "This is O(N^2) and can be very slow on the full test set."
            )
            self.compute_retrieval_metrics()
        if vis_tsne:
            print("Computing t-SNE visualization.")
            self.vis_tsne(self.output_dir + "/vis")
        if confusion_mtx:
            print("Writing confusion matrix.")
            self.confusion_mtx(self.output_dir + "/vis")
        if zero_shot_ood:
            self.zero_shot_ood(self.output_dir + "/ood")
        if few_shot_ood:
            self.few_shot_ood(self.output_dir + "/ood")

    def vis_tsne(self, output_dir):
        """
        This method takes all outputs from the test dataset, which means TSNE will be slow and crowded.
        It's recommended to run this one alone with a small portion.
        """
        os.makedirs(output_dir, exist_ok=True)

        all_content_idx = [x[1] for x in self.ground_truth]
        all_content_idx = np.array(all_content_idx)
        all_style_idx = [x[2] for x in self.ground_truth]
        all_style_idx = np.array(all_style_idx)

        all_emb_c = [x[1] for x in self.results]
        all_emb_c = np.array(all_emb_c)
        all_emb_c_vq = [x[2] for x in self.results]
        all_emb_c_vq = np.array(all_emb_c_vq)
        all_emb_s = [x[3] for x in self.results]
        all_emb_s = np.array(all_emb_s)

        tsne_c = TSNE(
            n_components=3, max_iter=1000
        )  # make use of the minor intra-class difference to plot TSNE
        tsne_s = TSNE(n_components=3, max_iter=1000)
        all_emb_c_and_codebook = np.concatenate((all_emb_c, self.codebook), axis=0)
        emb_c_and_codebook_tsne = tsne_c.fit_transform(all_emb_c_and_codebook)
        emb_s_tsne = tsne_s.fit_transform(all_emb_s)
        emb_c_tsne = emb_c_and_codebook_tsne[: all_emb_c.shape[0]]

        # plot z_content tsne using content labels
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = mscatter_3d(
            emb_c_tsne[:, 0],
            emb_c_tsne[:, 1],
            emb_c_tsne[:, 2],
            ax=ax,
            c=all_content_idx,
            m="o",
            s=50,
            cmap="tab20",
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.savefig(
            os.path.join(output_dir, "tsne_c_label_c.svg"),
            dpi=200,
            bbox_inches="tight",
        )

        # plot z_content tsne using style labels
        if not self.CONTINUOUS_STYLE:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")
            scatter = mscatter_3d(
                emb_c_tsne[:, 0],
                emb_c_tsne[:, 1],
                emb_c_tsne[:, 2],
                ax=ax,
                c=all_style_idx,
                m="o",
                s=50,
                cmap="tab20",
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            plt.savefig(
                os.path.join(output_dir, "tsne_c_label_s.svg"),
                dpi=200,
                bbox_inches="tight",
            )

        # plot z_style tsne using content labels
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        scatter = mscatter_3d(
            emb_s_tsne[:, 0],
            emb_s_tsne[:, 1],
            emb_s_tsne[:, 2],
            ax=ax,
            c=all_content_idx,
            m="o",
            s=50,
            cmap="tab20",
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        plt.savefig(
            os.path.join(output_dir, "tsne_s_label_c.svg"),
            dpi=200,
            bbox_inches="tight",
        )

        # plot z_style tsne using style labels
        if not self.CONTINUOUS_STYLE:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="3d")
            scatter = mscatter_3d(
                emb_s_tsne[:, 0],
                emb_s_tsne[:, 1],
                emb_s_tsne[:, 2],
                ax=ax,
                c=all_style_idx,
                m="o",
                s=50,
                cmap="tab20",
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            plt.savefig(
                os.path.join(output_dir, "tsne_s_label_s.svg"),
                dpi=200,
                bbox_inches="tight",
            )

    def confusion_mtx(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # plot the confusion matrix of the codebook
        confusion_matrix = self._normalized_codebook_confusion_matrix()
        # permute the rows to look like an eye
        codebook_permutation = Tester._get_confusion_matrix_permutation(
            confusion_matrix
        )

        confusion_matrix = confusion_matrix[codebook_permutation]
        plt.figure(figsize=(6, 6))
        sns.heatmap(confusion_matrix, cmap="Purples", vmin=0, vmax=1, cbar=False)
        plt.gca().set_aspect(1)
        plt.xticks(
            list(range(0, len(self.C_LIST), 3)),
            [str(x) for x in list(range(0, len(self.C_LIST), 3))],
        )

        plt.xlabel("Content Index", fontsize=20)
        plt.ylabel("Codebook Index", fontsize=20)
        plt.savefig(
            os.path.join(output_dir, "codebook_confusion_matrix.svg"),
            dpi=200,
            bbox_inches="tight",
        )

        print("Codebook Accuracy:", confusion_mtx_acc(confusion_matrix))

    def _normalized_codebook_confusion_matrix(self):
        return self.confusion_counts / (
            self.confusion_counts.sum(axis=1, keepdims=True) + 1e-7
        )

    @staticmethod
    def _get_confusion_matrix_permutation(confusion_matrix):
        """
        confusion_matrix: (n_atoms, n_classes)
        """
        assignments = np.argmax(confusion_matrix, axis=1)
        perm = np.argsort(assignments)

        return perm

    def compute_retrieval_metrics(self):
        k_list = [
            1,
            2,
            5,
            10,
            20,
            50,
            75,
            100,
            200,
            300,
            400,
            500,
            750,
            1000,
            1500,
            2000,
            2500,
            3000,
        ]

        all_content_idx = [x[1] for x in self.ground_truth]
        all_content_idx = np.array(all_content_idx)
        all_style_idx = [x[2] for x in self.ground_truth]
        all_style_idx = np.array(all_style_idx)

        all_emb_c = [x[1] for x in self.results]
        all_emb_c = np.array(all_emb_c)
        all_emb_s = [x[3] for x in self.results]
        all_emb_s = np.array(all_emb_s)

        # compute the precision & recall at k metrics
        c_precisions, c_recalls, c_f1s = precision_recall_at_k(
            all_emb_c, all_content_idx, k_list
        )
        s_precisions, s_recalls, s_f1s = precision_recall_at_k(
            all_emb_s, all_style_idx, k_list
        )
        # check using the other label
        c_precisions_using_s, c_recalls_using_s, c_f1s_using_s = precision_recall_at_k(
            all_emb_s, all_content_idx, k_list
        )
        s_precisions_using_c, s_recalls_using_c, s_f1s_using_c = precision_recall_at_k(
            all_emb_c, all_style_idx, k_list
        )

        c_auc = area_under_prcurve(c_recalls, c_precisions)
        s_auc = area_under_prcurve(s_recalls, s_precisions)
        c_auc_using_s = area_under_prcurve(c_recalls_using_s, c_precisions_using_s)
        s_auc_using_c = area_under_prcurve(s_recalls_using_c, s_precisions_using_c)

        # Check themselves
        print("Content F1 Range: ", min(c_f1s), max(c_f1s))
        print("Style F1 Range: ", min(s_f1s), max(s_f1s))
        print("Content AUC:", c_auc)
        print("Style AUC:", s_auc)
        # Check using the other label
        print(
            "Content F1 (Using Style Emb) Range: ",
            min(c_f1s_using_s),
            max(c_f1s_using_s),
        )
        print(
            "Style F1 (Using Content Emb) Range: ",
            min(s_f1s_using_c),
            max(s_f1s_using_c),
        )
        print("Content AUC (Using Style Emb):", c_auc_using_s)
        print("Style AUC (Using Content Emb):", s_auc_using_c)

    def zero_shot_ood(self, output_dir):
        config = self.config

        ood_dataloader_module = import_module(
            "dataloader." + config["dataloader"] + "_ood"
        )

        data_dir = config["ood_data_dir"] + "/test"
        print("OOD Data Directory:", data_dir)

        # first compute the confusion matrix using original test data
        os.makedirs(output_dir, exist_ok=True)
        confusion_matrix = self._normalized_codebook_confusion_matrix()
        # permute the rows to look like an eye
        codebook_permutation = Tester._get_confusion_matrix_permutation(
            confusion_matrix
        )
        print(codebook_permutation)

        # then compute the ood classification accuracy
        test_loader = ood_dataloader_module.get_dataloader(  # batch inference
            data_dir,
            batch_size=config["batch_size"],
            n_fragments=config["model_config"]["n_fragments"],
            fragment_len=config["model_config"]["fragment_len"],
            shuffle=False,
        )

        ground_truth = []
        results = []
        sample_vq_indices = []

        for i, batch in enumerate(test_loader):
            batch_data, content_idx, style_idx = batch
            batch_data = batch_data.to(self.device)

            with torch.no_grad():
                recon, emb_c, emb_c_vq, vq_indices, vq_commit_loss, emb_s, *rest = (
                    self.model(batch_data)
                )

            # detach everything
            batch_data = batch_data.detach().cpu().numpy()
            content_idx = content_idx.detach().cpu().numpy()
            style_idx = style_idx.detach().cpu().numpy()
            recon = recon.detach().cpu().numpy()
            emb_c = emb_c.detach().cpu().numpy()
            emb_c_vq = emb_c_vq.detach().cpu().numpy()
            vq_indices = vq_indices.detach().cpu().numpy()
            emb_s = emb_s.detach().cpu().numpy()
            for j in range(emb_c.shape[0]):
                for k in range(emb_c.shape[1]):
                    ground_truth.append(
                        (
                            batch_data[j, k],
                            content_idx[j, k],
                            style_idx[j, k],
                        )
                    )
                    results.append(
                        (
                            recon[j, k],
                            emb_c[j, k],
                            emb_c_vq[j, k],
                            emb_s[j, k],
                        )
                    )
                sample_vq_indices.append(vq_indices[j])

        sample_vq_indices = np.array(sample_vq_indices).flatten()
        all_content_idx = [x[1] for x in ground_truth]
        all_content_idx = np.array(all_content_idx)

        accuracy = 0
        for i in range(len(sample_vq_indices)):
            if all_content_idx[i] == np.argwhere(
                codebook_permutation == sample_vq_indices[i]
            ):
                accuracy += 1
        accuracy /= len(sample_vq_indices)

        print("OOD Classification Accuracy:", accuracy)

    def few_shot_ood(self, output_dir, lr=1e-4, n_epochs=1):
        """
        On top of zero-shot OOD, this method trains the model on the OOD data with few shots.
        The optimizer is AdamW.
        """
        config = self.config

        ood_dataloader_module = import_module(
            "dataloader." + config["dataloader"] + "_ood"
        )

        data_dir = config["ood_data_dir"]
        print("OOD Data Directory:", data_dir)

        # first compute the confusion matrix using the original test data
        os.makedirs(output_dir, exist_ok=True)
        confusion_matrix = self._normalized_codebook_confusion_matrix()
        # permute the rows to look like an eye
        codebook_permutation = Tester._get_confusion_matrix_permutation(
            confusion_matrix
        )

        # then compute the ood classification accuracy
        train_dirs = glob(data_dir + "/*shot")
        for train_dir in train_dirs:
            n_shots = int(
                train_dir.split("/")[-1].split("_")[0]
            )  # this is how it's named

            train_loader = ood_dataloader_module.get_dataloader(
                train_dir,
                batch_size=config["batch_size"],
                n_fragments=config["model_config"]["n_fragments"],
                fragment_len=config["model_config"]["fragment_len"],
                shuffle=True,
            )
            test_loader = ood_dataloader_module.get_dataloader(
                data_dir + "/test",
                batch_size=config["batch_size"],
                n_fragments=config["model_config"]["n_fragments"],
                fragment_len=config["model_config"]["fragment_len"],
                shuffle=False,
            )

            model_adapted = deepcopy(self.model).to(self.device)
            model_adapted.train()
            optimizer = torch.optim.AdamW(model_adapted.parameters(), lr=lr)
            for i in tqdm(range(n_epochs)):
                for j, batch in enumerate(train_loader):
                    batch_data, content_idx, style_idx = batch
                    batch_data = batch_data.to(self.device)
                    recon, emb_c, emb_c_vq, vq_indices, vq_commit_loss, emb_s, *rest = (
                        model_adapted(batch_data, freeze_codebook=True)
                    )
                    losses = self.loss.compute_loss(
                        recon,
                        emb_c,
                        emb_c_vq,
                        vq_commit_loss,
                        emb_s,
                        batch_data,
                    )
                    optimizer.zero_grad()
                    losses["total_loss"].backward()
                    optimizer.step()

            model_adapted.eval()

            ground_truth = []
            results = []
            sample_vq_indices = []

            for i, batch in enumerate(test_loader):
                batch_data, content_idx, style_idx = batch
                batch_data = batch_data.to(self.device)

                with torch.no_grad():
                    recon, emb_c, emb_c_vq, vq_indices, vq_commit_loss, emb_s, *rest = (
                        model_adapted(batch_data)
                    )

                # detach everything
                batch_data = batch_data.detach().cpu().numpy()
                content_idx = content_idx.detach().cpu().numpy()
                style_idx = style_idx.detach().cpu().numpy()
                recon = recon.detach().cpu().numpy()
                emb_c = emb_c.detach().cpu().numpy()
                emb_c_vq = emb_c_vq.detach().cpu().numpy()
                vq_indices = vq_indices.detach().cpu().numpy()
                emb_s = emb_s.detach().cpu().numpy()
                for j in range(emb_c.shape[0]):
                    for k in range(emb_c.shape[1]):
                        ground_truth.append(
                            (
                                batch_data[j, k],
                                content_idx[j, k],
                                style_idx[j, k],
                            )
                        )
                        results.append(
                            (
                                recon[j, k],
                                emb_c[j, k],
                                emb_c_vq[j, k],
                                emb_s[j, k],
                            )
                        )
                    sample_vq_indices.append(vq_indices[j])

            sample_vq_indices = np.array(sample_vq_indices).flatten()
            all_content_idx = [x[1] for x in ground_truth]
            all_content_idx = np.array(all_content_idx)

            accuracy = 0
            for i in range(len(sample_vq_indices)):
                if all_content_idx[i] == np.argwhere(
                    codebook_permutation == sample_vq_indices[i]
                ):
                    accuracy += 1
            accuracy /= len(sample_vq_indices)

            print(f"OOD Classification Accuracy @ {n_shots} Shots:", accuracy)
