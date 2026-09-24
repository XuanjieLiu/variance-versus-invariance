"""Opt-in decoder normalization; historical constructors remain unchanged."""
from torch import nn


def configure_backbone_normalization(decoder, kind='batch', groups=8, scope='decoder'):
    if scope not in ('encoder', 'decoder'):
        raise ValueError(f'Unknown normalization scope: {scope!r}')
    if kind not in ('batch', 'group', 'none') or (scope == 'encoder' and kind == 'group'):
        raise ValueError(f'Unknown {scope}_normalization: {kind!r}')
    if kind == 'batch':
        return []
    if kind == 'group' and (isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0):
        raise ValueError('decoder_groupnorm_groups must be a positive integer')
    targets = [(name, module) for name, module in decoder.named_modules()
               if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not targets:
        raise ValueError('Decoder GroupNorm requested, but decoder has no BatchNorm layers')
    # Validate all targets before mutating any module. Never silently pick new groups.
    for name, module in targets:
        if not name:
            raise ValueError('Expected a decoder containing BatchNorm, not a bare BatchNorm layer')
        if kind == 'group' and module.num_features % groups:
            raise ValueError(f'decoder.{name}: channels={module.num_features} not divisible by groups={groups}')
    converted = []
    for name, module in targets:
        parent_name, _, child_name = name.rpartition('.')
        parent = decoder.get_submodule(parent_name) if parent_name else decoder
        replacement = (nn.Identity() if kind == 'none' else
                       nn.GroupNorm(groups, module.num_features, eps=module.eps, affine=module.affine))
        # Reuse affine parameters exactly, including dtype/device/requires_grad.
        # GroupNorm construction initializes constants, not random numbers.
        if kind == 'group' and module.affine:
            replacement.weight = module.weight
            replacement.bias = module.bias
        replacement.train(module.training)
        setattr(parent, child_name, replacement)
        converted.append({'name': f'{scope}.{name}', 'channels': module.num_features,
                          'groups': groups if kind == 'group' else None,
                          'eps': module.eps, 'affine': module.affine,
                          'replacement': 'GroupNorm' if kind == 'group' else 'Identity'})
    return converted


def configure_decoder_normalization(decoder, kind='batch', groups=8):
    """Backward-compatible decoder entry point; omission still means BN."""
    return configure_backbone_normalization(decoder, kind, groups, 'decoder')
