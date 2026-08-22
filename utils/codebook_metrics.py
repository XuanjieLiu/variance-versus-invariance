import hashlib

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def batch_confusion_counts(indices, labels, n_atoms, n_labels):
    """Count atom/label assignments without a Python loop."""
    indices = indices.detach().long().reshape(-1)
    labels = labels.detach().to(device=indices.device, dtype=torch.long).reshape(-1)
    if indices.numel() != labels.numel():
        raise ValueError(
            f"indices and labels differ in size: {indices.numel()} != {labels.numel()}"
        )

    valid = (
        (indices >= 0)
        & (indices < int(n_atoms))
        & (labels >= 0)
        & (labels < int(n_labels))
    )
    flat = indices[valid] * int(n_labels) + labels[valid]
    return torch.bincount(
        flat, minlength=int(n_atoms) * int(n_labels)
    ).reshape(int(n_atoms), int(n_labels))


def normalized_confusion_matrix(confusion_counts):
    counts = np.asarray(confusion_counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"Expected a 2D confusion matrix, got {counts.shape}")
    return counts / (counts.sum(axis=1, keepdims=True) + 1e-7)


def compute_assignment_metrics(confusion_counts):
    """Compute per-code purity, usage health, and one-to-one alignment."""
    counts = np.asarray(confusion_counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"Expected a 2D confusion matrix, got {counts.shape}")

    usage = counts.sum(axis=1)
    total = float(usage.sum())
    if total <= 0:
        raise ValueError("No code assignments were collected.")

    active_mask = usage > 0
    usage_prob = usage[active_mask] / total
    normalized = normalized_confusion_matrix(counts)
    normalized_total = float(normalized.sum())
    macro_atom_purity = (
        float(normalized.max(axis=1).sum() / normalized_total)
        if normalized_total > 0
        else 0.0
    )
    row_ind, col_ind = linear_sum_assignment(-counts)
    dominant_labels = np.argmax(counts[active_mask], axis=1)
    dominant_label_code_counts = np.bincount(
        dominant_labels, minlength=counts.shape[1]
    )
    dominant_count_mean = float(dominant_label_code_counts.mean())
    dominant_count_cv = (
        float(dominant_label_code_counts.std() / dominant_count_mean)
        if dominant_count_mean > 0
        else 0.0
    )

    return {
        "codebook_size": int(counts.shape[0]),
        "content_label_count": int(counts.shape[1]),
        "active_codes": int(active_mask.sum()),
        "usage_perplexity": float(
            np.exp(-np.sum(usage_prob * np.log(usage_prob)))
        ),
        "macro_atom_purity": macro_atom_purity,
        # Compatibility alias for old logs and downstream analysis.
        "legacy_codebook_accuracy": macro_atom_purity,
        "codebook_purity": float(counts.max(axis=1).sum() / total),
        "one_to_one_accuracy": float(counts[row_ind, col_ind].sum() / total),
        "dominant_label_coverage": int(np.count_nonzero(dominant_label_code_counts)),
        "dominant_label_code_counts": dominant_label_code_counts.astype(int).tolist(),
        "dominant_label_code_count_min": int(dominant_label_code_counts.min()),
        "dominant_label_code_count_max": int(dominant_label_code_counts.max()),
        "dominant_label_code_count_cv": dominant_count_cv,
    }


def compute_alias_geometry_metrics(confusion_counts, codebook):
    """Measure whether redundant pure codes form compact content clusters.

    Labels are used only as an evaluation probe: every active code is assigned
    its dominant content label from ``confusion_counts``.  Distances are
    measured in the native codebook space, before any decoder projection.
    """
    counts = np.asarray(confusion_counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"Expected a 2D confusion matrix, got {counts.shape}")
    centers = torch.as_tensor(codebook, dtype=torch.float64)
    if centers.ndim == 3 and centers.shape[0] == 1:
        centers = centers.squeeze(0)
    if centers.ndim != 2 or centers.shape[0] != counts.shape[0]:
        raise ValueError(
            "Codebook and confusion row counts differ: "
            f"{tuple(centers.shape)} vs {counts.shape}."
        )

    usage = torch.as_tensor(counts.sum(axis=1), dtype=torch.float64)
    active_mask = usage > 0
    active_centers = centers[active_mask]
    active_usage = usage[active_mask]
    active_counts = torch.as_tensor(counts[active_mask.numpy()])
    dominant = active_counts.argmax(dim=1)
    covered_labels = torch.unique(dominant, sorted=True)

    centroids = []
    within_weighted_sq = torch.zeros((), dtype=torch.float64)
    total_weight = torch.zeros((), dtype=torch.float64)
    for label in covered_labels:
        mask = dominant == label
        label_usage = active_usage[mask]
        label_centers = active_centers[mask]
        centroid = (label_centers * label_usage.unsqueeze(1)).sum(dim=0) / label_usage.sum()
        centroids.append(centroid)
        within_weighted_sq += (
            label_centers.sub(centroid).square().sum(dim=1) * label_usage
        ).sum()
        total_weight += label_usage.sum()

    within_rms = float(
        (within_weighted_sq / total_weight.clamp_min(1e-30)).sqrt().item()
    )
    if len(centroids) > 1:
        centroid_tensor = torch.stack(centroids)
        centroid_distances = torch.cdist(centroid_tensor, centroid_tensor)
        centroid_distances.fill_diagonal_(float("inf"))
        between_nn_median = float(
            centroid_distances.min(dim=1).values.median().item()
        )
    else:
        between_nn_median = float("nan")

    if active_centers.shape[0] > 1:
        pairwise = torch.cdist(active_centers, active_centers)
        pairwise.fill_diagonal_(float("inf"))
        same_mask = dominant[:, None] == dominant[None, :]
        same_mask.fill_diagonal_(False)
        other_mask = ~same_mask
        other_mask.fill_diagonal_(False)
        has_same = same_mask.any(dim=1)
        has_other = other_mask.any(dim=1)
        valid = has_same & has_other
        nearest_same = pairwise.masked_fill(~same_mask, float("inf")).min(dim=1).values
        nearest_other = pairwise.masked_fill(~other_mask, float("inf")).min(dim=1).values
        if bool(valid.any().item()):
            same_median = float(nearest_same[valid].median().item())
            other_median = float(nearest_other[valid].median().item())
            same_closer_fraction = float(
                (nearest_same[valid] < nearest_other[valid]).double().mean().item()
            )
        else:
            same_median = other_median = same_closer_fraction = float("nan")
    else:
        same_median = other_median = same_closer_fraction = float("nan")

    return {
        "alias_within_content_rms": within_rms,
        "alias_between_content_nn_median": between_nn_median,
        "alias_within_between_ratio": (
            within_rms / between_nn_median
            if np.isfinite(between_nn_median) and between_nn_median > 0
            else float("nan")
        ),
        "alias_nearest_same_distance_median": same_median,
        "alias_nearest_other_distance_median": other_median,
        "alias_nearest_same_closer_fraction": same_closer_fraction,
    }


def hungarian_row_permutation(confusion_counts):
    """Order matched rows by label, then append any unmatched rows."""
    counts = np.asarray(confusion_counts, dtype=np.float64)
    row_ind, col_ind = linear_sum_assignment(-counts)
    matched_rows = [row for _, row in sorted(zip(col_ind.tolist(), row_ind.tolist()))]
    matched = set(matched_rows)
    unmatched = [row for row in range(counts.shape[0]) if row not in matched]
    unmatched.sort(key=lambda row: int(np.argmax(counts[row])))
    return np.asarray(matched_rows + unmatched, dtype=np.int64)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
