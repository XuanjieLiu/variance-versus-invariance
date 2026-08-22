import os

import torch
from torch.utils.data import DataLoader, Subset


SUBSET_STRATEGIES = ("random", "style_stratified")


def select_subset_indices(dataset, subset_size, seed=0, strategy="random"):
    dataset_size = len(dataset)
    count = min(int(subset_size), dataset_size)
    if count <= 0:
        raise ValueError("test_subset_size must be a positive integer.")
    if strategy not in SUBSET_STRATEGIES:
        raise ValueError(
            f"Unknown subset strategy {strategy!r}; choose from {SUBSET_STRATEGIES}."
        )

    generator = torch.Generator().manual_seed(int(seed))
    if strategy == "random":
        indices = torch.randperm(dataset_size, generator=generator)[:count].tolist()
        return sorted(indices), {}

    if not hasattr(dataset, "png_paths") or not hasattr(dataset, "s_list"):
        raise ValueError(
            "style_stratified sampling requires a dataset with png_paths and s_list."
        )

    groups = {style: [] for style in dataset.s_list}
    for index, path in enumerate(dataset.png_paths):
        style = os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1]
        if style in groups:
            groups[style].append((path, index))

    n_styles = len(dataset.s_list)
    base, remainder = divmod(count, n_styles)
    selected = []
    style_counts = {}
    for style_index, style in enumerate(dataset.s_list):
        take = base + int(style_index < remainder)
        available = sorted(groups[style], key=lambda item: item[0])
        if len(available) < take:
            raise ValueError(
                f"Style {style!r} has {len(available)} samples, fewer than {take}."
            )
        order = torch.randperm(len(available), generator=generator)[:take].tolist()
        selected.extend(available[position][1] for position in order)
        style_counts[style] = take

    return sorted(selected), style_counts


def make_subset_loader(loader, subset_size, seed=0, strategy="random"):
    indices, style_counts = select_subset_indices(
        loader.dataset, subset_size, seed=seed, strategy=strategy
    )
    kwargs = {
        "batch_size": loader.batch_size,
        "shuffle": False,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": False,
    }
    if loader.num_workers > 0:
        kwargs["persistent_workers"] = loader.persistent_workers
        if loader.prefetch_factor is not None:
            kwargs["prefetch_factor"] = loader.prefetch_factor

    subset_loader = DataLoader(Subset(loader.dataset, indices), **kwargs)
    metadata = {
        "subset_size": len(indices),
        "subset_seed": int(seed),
        "subset_strategy": strategy,
        "style_counts": style_counts,
    }
    return subset_loader, metadata
