"""Permanent snapshot schedule, independent of current/best rotation."""


def validate_snapshot_schedule(config):
    interval = config.get("snapshot_every_n_epochs", 0)
    epochs = config.get("snapshot_epochs", [])
    if type(interval) is not int or interval < 0:
        raise ValueError("snapshot_every_n_epochs must be a nonnegative integer")
    if not isinstance(epochs, list) or any(type(e) is not int or e < 0 for e in epochs):
        raise ValueError("snapshot_epochs must contain zero-based nonnegative integer epochs")
    if len(set(epochs)) != len(epochs):
        raise ValueError("snapshot_epochs must not contain duplicates")


def snapshot_due(config, epoch):
    interval = config.get("snapshot_every_n_epochs", 0)
    return epoch in config.get("snapshot_epochs", []) or (
        interval > 0 and (epoch + 1) % interval == 0)
