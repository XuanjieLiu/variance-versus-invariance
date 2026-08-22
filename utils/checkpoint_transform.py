"""Explicit, auditable checkpoint transforms used by research runs."""

import torch


EMA_CODEBOOK_SUFFIXES = {
    "embed": "vq._codebook.embed",
    "embed_avg": "vq._codebook.embed_avg",
    "cluster_size": "vq._codebook.cluster_size",
}

VQ_PROJECTION_KEYS = {
    "project_in_weight": "vq.project_in.weight",
    "project_in_bias": "vq.project_in.bias",
    "project_out_weight": "vq.project_out.weight",
    "project_out_bias": "vq.project_out.bias",
}


def _unique_key_with_suffix(state_dict, suffix):
    matches = [key for key in state_dict if key.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one EMA codebook tensor ending in {suffix!r}, found {matches}."
        )
    return matches[0]


def expand_ema_codebook_state(
    state_dict,
    source_atoms,
    target_atoms,
    jitter_fraction=0.01,
    random_seed=0,
):
    """Split each EMA atom into an interleaved symmetric pair."""
    source_atoms = int(source_atoms)
    target_atoms = int(target_atoms)
    if target_atoms != 2 * source_atoms:
        raise ValueError(
            "EMA expansion currently supports exactly a 2x codebook expansion."
        )
    jitter_fraction = float(jitter_fraction)
    if jitter_fraction < 0:
        raise ValueError("jitter_fraction must be non-negative.")

    keys = {
        name: _unique_key_with_suffix(state_dict, suffix)
        for name, suffix in EMA_CODEBOOK_SUFFIXES.items()
    }
    embed = state_dict[keys["embed"]]
    embed_avg = state_dict[keys["embed_avg"]]
    cluster_size = state_dict[keys["cluster_size"]]
    if embed.ndim != 3 or embed.shape[0] != 1:
        raise ValueError(
            f"Only a single EMA Euclidean codebook is supported, got {embed.shape}."
        )
    if embed.shape[1] != source_atoms:
        raise ValueError(
            f"Expected {source_atoms} source atoms, got embed shape {embed.shape}."
        )
    if embed_avg.shape != embed.shape or tuple(cluster_size.shape) != (1, source_atoms):
        raise ValueError(
            "EMA embed, embed_avg, and cluster_size shapes are inconsistent: "
            f"{embed.shape}, {embed_avg.shape}, {cluster_size.shape}."
        )

    centers = embed.detach().to(device="cpu", dtype=torch.float64)[0]
    pairwise = torch.cdist(centers, centers)
    pairwise.fill_diagonal_(float("inf"))
    median_nearest_neighbor_distance = float(
        pairwise.min(dim=1).values.median().item()
    )
    jitter_norm = jitter_fraction * median_nearest_neighbor_distance

    generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
    directions = torch.randn(
        centers.shape, generator=generator, dtype=torch.float64
    )
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    delta = directions * jitter_norm
    expanded = torch.stack((centers + delta, centers - delta), dim=1).reshape(
        target_atoms, centers.shape[1]
    )

    source_cluster = cluster_size.detach().to(device="cpu", dtype=torch.float64)[0]
    expanded_cluster = (
        torch.stack((source_cluster / 2, source_cluster / 2), dim=1)
        .reshape(target_atoms)
    )
    expanded_avg = expanded * expanded_cluster.unsqueeze(1)

    transformed = dict(state_dict)
    transformed[keys["embed"]] = expanded.unsqueeze(0).to(
        device=embed.device, dtype=embed.dtype
    )
    transformed[keys["cluster_size"]] = expanded_cluster.unsqueeze(0).to(
        device=cluster_size.device, dtype=cluster_size.dtype
    )
    transformed[keys["embed_avg"]] = expanded_avg.unsqueeze(0).to(
        device=embed_avg.device, dtype=embed_avg.dtype
    )

    metadata = {
        "type": "expand_ema_codebook",
        "source_shape": list(embed.shape),
        "target_shape": list(transformed[keys["embed"]].shape),
        "source_atoms": source_atoms,
        "target_atoms": target_atoms,
        "copies_per_atom": 2,
        "jitter_fraction": jitter_fraction,
        "median_nearest_neighbor_distance": median_nearest_neighbor_distance,
        "jitter_norm": jitter_norm,
        "random_seed": int(random_seed),
        "ema_mass_before": float(cluster_size.sum().item()),
        "ema_mass_after": float(
            transformed[keys["cluster_size"]].sum().item()
        ),
        "state_keys": keys,
    }
    return transformed, metadata


def _canonicalize_component_signs(components):
    """Remove the arbitrary SVD sign so transforms are byte-for-byte repeatable."""
    components = components.clone()
    pivots = components.abs().argmax(dim=1)
    signs = components[
        torch.arange(components.shape[0], device=components.device), pivots
    ].sign()
    signs[signs == 0] = 1
    return components * signs.unsqueeze(1)


def pca_project_and_expand_ema_codebook_state(
    state_dict,
    source_atoms,
    target_atoms,
    target_dim,
    jitter_fraction=0.01,
    random_seed=0,
):
    """Project a full-dimensional EMA codebook with PCA and split every atom.

    The generated projection is an affine PCA encoder and its tied affine
    decoder.  It is suitable for loading a non-projected VectorQuantize
    checkpoint into a model configured with ``codebook_dim=target_dim``.
    """
    source_atoms = int(source_atoms)
    target_atoms = int(target_atoms)
    target_dim = int(target_dim)
    if target_atoms != 2 * source_atoms:
        raise ValueError(
            "PCA EMA expansion currently supports exactly a 2x codebook expansion."
        )
    if target_dim <= 0:
        raise ValueError("target_dim must be positive.")
    jitter_fraction = float(jitter_fraction)
    if jitter_fraction < 0:
        raise ValueError("jitter_fraction must be non-negative.")

    keys = {
        name: _unique_key_with_suffix(state_dict, suffix)
        for name, suffix in EMA_CODEBOOK_SUFFIXES.items()
    }
    embed = state_dict[keys["embed"]]
    embed_avg = state_dict[keys["embed_avg"]]
    cluster_size = state_dict[keys["cluster_size"]]
    if embed.ndim != 3 or embed.shape[0] != 1:
        raise ValueError(
            f"Only a single EMA Euclidean codebook is supported, got {embed.shape}."
        )
    if embed.shape[1] != source_atoms:
        raise ValueError(
            f"Expected {source_atoms} source atoms, got embed shape {embed.shape}."
        )
    if target_dim > min(source_atoms, embed.shape[2]):
        raise ValueError(
            f"target_dim={target_dim} exceeds PCA rank bound for {tuple(embed.shape)}."
        )
    if embed_avg.shape != embed.shape or tuple(cluster_size.shape) != (1, source_atoms):
        raise ValueError(
            "EMA embed, embed_avg, and cluster_size shapes are inconsistent: "
            f"{embed.shape}, {embed_avg.shape}, {cluster_size.shape}."
        )
    existing_projection_keys = [
        key
        for key in state_dict
        if key.startswith("vq.project_in.") or key.startswith("vq.project_out.")
    ]
    if existing_projection_keys:
        raise ValueError(
            "PCA warm projection requires a full-dimensional source without VQ "
            f"projection parameters, found {existing_projection_keys}."
        )

    centers = embed.detach().to(device="cpu", dtype=torch.float64)[0]
    mean = centers.mean(dim=0)
    centered = centers - mean
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    components = _canonicalize_component_signs(vh[:target_dim].contiguous())
    projected_centers = centered @ components.t()
    reconstructed_centers = projected_centers @ components + mean
    residual = reconstructed_centers - centers
    total_variance = singular_values.square().sum()
    retained_variance = singular_values[:target_dim].square().sum()
    explained_variance_ratio = float(
        (retained_variance / total_variance.clamp_min(1e-30)).item()
    )

    pairwise = torch.cdist(projected_centers, projected_centers)
    pairwise.fill_diagonal_(float("inf"))
    median_nearest_neighbor_distance = float(
        pairwise.min(dim=1).values.median().item()
    )
    jitter_norm = jitter_fraction * median_nearest_neighbor_distance
    generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
    directions = torch.randn(
        projected_centers.shape, generator=generator, dtype=torch.float64
    )
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    delta = directions * jitter_norm
    expanded = torch.stack(
        (projected_centers + delta, projected_centers - delta), dim=1
    ).reshape(target_atoms, target_dim)

    source_cluster = cluster_size.detach().to(device="cpu", dtype=torch.float64)[0]
    expanded_cluster = torch.stack(
        (source_cluster / 2, source_cluster / 2), dim=1
    ).reshape(target_atoms)
    expanded_avg = expanded * expanded_cluster.unsqueeze(1)

    transformed = dict(state_dict)
    transformed[keys["embed"]] = expanded.unsqueeze(0).to(
        device=embed.device, dtype=embed.dtype
    )
    transformed[keys["cluster_size"]] = expanded_cluster.unsqueeze(0).to(
        device=cluster_size.device, dtype=cluster_size.dtype
    )
    transformed[keys["embed_avg"]] = expanded_avg.unsqueeze(0).to(
        device=embed_avg.device, dtype=embed_avg.dtype
    )

    projection_dtype = embed.dtype
    projection_device = embed.device
    transformed[VQ_PROJECTION_KEYS["project_in_weight"]] = components.to(
        device=projection_device, dtype=projection_dtype
    )
    transformed[VQ_PROJECTION_KEYS["project_in_bias"]] = (
        -(components @ mean)
    ).to(device=projection_device, dtype=projection_dtype)
    transformed[VQ_PROJECTION_KEYS["project_out_weight"]] = components.t().to(
        device=projection_device, dtype=projection_dtype
    )
    transformed[VQ_PROJECTION_KEYS["project_out_bias"]] = mean.to(
        device=projection_device, dtype=projection_dtype
    )

    metadata = {
        "type": "pca_project_and_expand_ema_codebook",
        "source_shape": list(embed.shape),
        "target_shape": list(transformed[keys["embed"]].shape),
        "source_atoms": source_atoms,
        "target_atoms": target_atoms,
        "source_dim": int(embed.shape[2]),
        "target_dim": target_dim,
        "copies_per_atom": 2,
        "pca_fit": "unweighted_source_codebook",
        "pca_sign_canonicalized": True,
        "pca_explained_variance_ratio": explained_variance_ratio,
        "pca_reconstruction_rms": float(residual.square().mean().sqrt().item()),
        "pca_reconstruction_max_l2": float(residual.norm(dim=1).max().item()),
        "jitter_fraction": jitter_fraction,
        "median_nearest_neighbor_distance": median_nearest_neighbor_distance,
        "jitter_norm": jitter_norm,
        "random_seed": int(random_seed),
        "ema_mass_before": float(cluster_size.sum().item()),
        "ema_mass_after": float(
            transformed[keys["cluster_size"]].sum().item()
        ),
        "state_keys": {**keys, **VQ_PROJECTION_KEYS},
    }
    return transformed, metadata
