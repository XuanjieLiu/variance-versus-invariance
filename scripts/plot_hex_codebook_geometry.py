"""Render native Hex, PhoneNums or Uppercase atoms without changing model/data.

Run on a ws-* compute node. Labels come from a hash-verified existing full-test
Hungarian mapping where available, otherwise an encoder-only full-test pass.
Labels are never used to fit any dimensionality reduction.
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import sys
from importlib import import_module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import scipy
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE, trustworthiness
from sklearn.neighbors import kneighbors_graph
import torch
import yaml


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or not np.isfinite(counts).all() or np.any(counts < 0) or counts.sum() <= 0:
        raise ValueError('Invalid code/content counts')
    code, label = linear_sum_assignment(-counts)
    return dict(zip(code.tolist(), label.tolist())), float(counts[code, label].sum() / counts.sum())


def get_mapping(run, checkpoint, digest, config, state, n_codes, alphabet):
    cache = run / 'codebook_geometry' / f'mapping__{checkpoint.stem}__test-full.json'
    candidates = sorted(run.glob(f'acceptance_*/codebook_confusion_matrix__{checkpoint.stem}__test-full*.json'))
    if cache.is_file(): candidates.insert(0, cache)
    for path in candidates:
        info = json.loads(path.read_text())
        if info.get('checkpoint_sha256') != digest or info.get('split') != 'test':
            continue
        if 'raw_counts' not in info or not info.get('pages'):
            continue
        counts = np.asarray(info['raw_counts'])
        if counts.shape != (n_codes, len(alphabet)) or info.get('content_labels') != alphabet:
            raise ValueError(f'Mapping dimensions/content order mismatch: {path}')
        mapping, accuracy = mapping_from_counts(counts)
        return mapping, accuracy, counts, int(info['pages']), path, False

    # Older evaluation JSONs saved the score but not counts/mapping. Regenerate
    # only assignments; do not decode, evaluate losses, train or alter BN/VQ.
    from model.factory import get_model
    from run_codebook_health import build_loader
    from utils.codebook_metrics import batch_confusion_counts
    if not torch.cuda.is_available():
        raise RuntimeError('A GPU is required to recover a missing mapping.')
    torch.manual_seed(0)
    model = get_model(config['dataloader'], config['model_config']).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    loader, content_names, _ = build_loader(config)
    assert list(content_names) == alphabet
    counts = torch.zeros(n_codes, len(alphabet), device='cuda', dtype=torch.int64)
    pages = 0
    with torch.inference_mode():
        for images, content, _ in loader:
            zc, _ = model.encode(images.cuda())
            _, indices, _ = model.quantize(zc, freeze_codebook=True)
            counts += batch_confusion_counts(indices, content, n_codes, len(alphabet))
            pages += len(images)
    assert pages == len(loader.dataset)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), f'Model mutated during mapping: {key}'
    counts = counts.cpu().numpy()
    mapping, accuracy = mapping_from_counts(counts)
    old = run / 'vis' / f'codebook_confusion_matrix__{checkpoint.stem}__test-full.json'
    previous = None
    if old.is_file():
        previous = json.loads(old.read_text())
        if previous.get('checkpoint_sha256') == digest:
            if abs(previous['metrics']['one_to_one_accuracy'] - accuracy) > .002:
                raise ValueError('Recovered mapping score differs materially from historical evaluation')
    cache.parent.mkdir(exist_ok=True, parents=True)
    if cache.exists(): raise FileExistsError(cache)
    cache.write_text(json.dumps({'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
        'split': 'test', 'pages': pages, 'content_labels': alphabet, 'raw_counts': counts.tolist(),
        'code_to_label': mapping, 'one_to_one_accuracy': accuracy,
        'scope': 'full test encoder-only frozen model; existing dataset split unchanged',
        'config_sha256': sha256(run/'config.yaml'),
        'historical_accuracy': previous['metrics']['one_to_one_accuracy'] if previous else None}, indent=2)+'\n')
    del model, before
    torch.cuda.empty_cache()
    return mapping, accuracy, counts, pages, cache, True


def draw(ax, coordinates, labels, colors, method, dims, subtitle):
    ax.scatter(*coordinates.T, c=colors, s=105 if dims == 2 else 70,
               edgecolors='white', linewidths=.9, **({'depthshade': False} if dims == 3 else {}))
    span = np.ptp(coordinates, axis=0)
    half = max(float(span.max()) * .68, 1e-6)
    center = (coordinates.max(0) + coordinates.min(0)) / 2
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    if dims == 2:
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    else:
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=23, azim=-58)
        ax.set_zlabel('Component 3', labelpad=7)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(f'{method} | {dims}D\n{subtitle}', fontsize=12, pad=15)
    if dims == 3:
        px, py, _ = proj3d.proj_transform(*coordinates.T, ax.get_proj())
        displayed = np.column_stack([px, py])
    else:
        displayed = coordinates
    ordering = np.argsort(displayed[:, 0])
    ranks = np.empty(len(ordering), dtype=int)
    ranks[ordering] = np.arange(len(ordering))
    singular = np.linalg.svd(coordinates - coordinates.mean(0), compute_uv=False)
    almost_line = singular[0]**2 / np.sum(singular**2) > .98
    for i, (point, label) in enumerate(zip(displayed, labels)):
        # Offsets/leader lines affect labels only, never the plotted atoms.
        dy = (22, -32, 52, -60)[ranks[i] % 4] if almost_line else (12, -24)[ranks[i] % 2]
        text = ax.annotate(label.replace(' [', '\n['), point, xytext=(0, dy),
                           xycoords='data', textcoords='offset points', fontsize=9,
                           ha='center', va='center', color='#172333', annotation_clip=False,
                           arrowprops={'arrowstyle': '-', 'color': '#82909a', 'lw': .6},
                           bbox={'boxstyle': 'round,pad=.12', 'fc': 'white', 'ec': 'none', 'alpha': .85})
        text.set_path_effects([pe.withStroke(linewidth=2, foreground='white', alpha=.9)])


def separate_labels(fig):
    """Repel text rectangles in display space; never move any atom or fit.

    Use bbox patches, not annotation extents (which also include leader lines).
    The deterministic placement uses no RNG and works on projected 3D labels.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        labels = [text for text in ax.texts if isinstance(text, matplotlib.text.Annotation)]
        if not labels:
            continue
        boxes = np.array([text.get_bbox_patch().get_window_extent(renderer).extents
                          for text in labels])
        original = boxes.copy()
        bounds = ax.get_window_extent(renderer).extents
        margin = 4.
        for _ in range(300):
            overlaps = 0
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    overlap = np.minimum(boxes[i, 2:], boxes[j, 2:]) - np.maximum(boxes[i, :2], boxes[j, :2]) + margin
                    if np.all(overlap > 0):
                        overlaps += 1
                        axis = int(np.argmin(overlap))
                        direction = 1 if boxes[j, axis] >= boxes[i, axis] else -1
                        shift = direction * (overlap[axis] / 2 + .1)
                        boxes[i, [axis, axis + 2]] -= shift
                        boxes[j, [axis, axis + 2]] += shift
            # Keep labels inside each panel, away from adjacent panels/titles.
            for box in boxes:
                for axis in range(2):
                    shift = max(bounds[axis] + margin - box[axis], 0)
                    shift -= max(box[axis + 2] + shift - bounds[axis + 2] + margin, 0)
                    box[[axis, axis + 2]] += shift
            if not overlaps:
                break
        for text, shift in zip(labels, (boxes[:, :2] - original[:, :2]) * 72 / fig.dpi):
            text.set_position(np.asarray(text.xyann) + shift)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--perplexity', type=float, default=4)
    parser.add_argument('--neighbors', type=int, default=5)
    parser.add_argument('--output-tag', default='geometry-v3')
    args = parser.parse_args()
    if not socket.gethostname().startswith('ws-'):
        raise RuntimeError('All geometry computation must run on a ws-* compute node.')
    torch.set_num_threads(4)
    run = Path(args.run)
    if not run.is_dir():
        run = ROOT / 'logs' / args.run
    run = run.resolve(strict=True)
    metadata = json.loads((run / 'best_macro_atom_purity.json').read_text())
    checkpoint = run / metadata['checkpoint']
    digest = sha256(checkpoint)
    config = yaml.safe_load((run/'config.yaml').read_text())
    alphabet = list(import_module('dataloader.' + config['dataloader']).C_LIST)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    keys = [k for k in state['model'] if k.endswith('vq._codebook.embed')]
    assert len(keys) == 1, keys
    atoms = state['model'][keys[0]].detach().double().numpy()
    assert atoms.ndim == 3 and atoms.shape[0] == 1, atoms.shape
    n_codes, native_dim = atoms.shape[1:]
    assert n_codes == len(alphabet) == config['model_config']['n_atoms'], 'This plotter requires K=content count'
    assert native_dim == config['model_config'].get('vq_codebook_dim', config['model_config']['d_emb_c'])
    mapping, accuracy, counts, pages, mapping_file, recomputed = get_mapping(
        run, checkpoint, digest, config, state, n_codes, alphabet)
    assert sorted(mapping) == sorted(mapping.values()) == list(range(n_codes))
    assert state['epoch'] == metadata['epoch'] and np.isfinite(atoms).all()
    # Sort only for presentation; the algorithms do not see semantic labels.
    code_ids = np.array(sorted(mapping, key=mapping.get))
    x = atoms[0, code_ids].copy()
    assert np.min(pdist(x)) > 0
    if not 0 < args.perplexity < n_codes or not 1 <= args.neighbors < n_codes:
        raise ValueError('Invalid perplexity/neighbors for codebook size')
    graph = kneighbors_graph(x, args.neighbors, mode='connectivity', include_self=False)
    components = connected_components(graph.maximum(graph.T), directed=False, return_labels=False)
    if components != 1:
        raise ValueError(f'Isomap graph has {components} components; choose more neighbors explicitly.')
    if not args.output_tag or Path(args.output_tag).name != args.output_tag or args.output_tag in ('.', '..'):
        raise ValueError('output-tag must be a single directory suffix')
    out = run / 'codebook_geometry' / f'{checkpoint.stem}__native-vq__seed{args.seed}__p{args.perplexity:g}__k{args.neighbors}__{args.output_tag}'
    out.mkdir(parents=True, exist_ok=False)
    report = {
        'run': run.name, 'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
        'epoch': state['epoch'], 'config_sha256': sha256(run / 'config.yaml'),
        'mapping_file': str(mapping_file), 'mapping_file_sha256': sha256(mapping_file),
        'mapping_scope': f'full{pages}-page test Hungarian, annotation only',
        'mapping_accuracy': accuracy, 'mapping_recomputed': recomputed,
        'code_to_label': mapping, 'content_names': alphabet,
        'code_ids_in_content_order': code_ids.tolist(),
        'matched_fraction_by_code': {str(k): float(counts[k, v] / counts[k].sum()) if counts[k].sum() else None for k,v in mapping.items()},
        'representation': keys[0], 'native_shape': [n_codes, native_dim],
        'preprocessing': 'raw Euclidean atoms; no whitening, unit normalization or per-feature scaling',
        'node': socket.gethostname(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'script_sha256': sha256(__file__),
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__,
                     'sklearn': sklearn.__version__, 'torch': torch.__version__,
                     'matplotlib': matplotlib.__version__},
        'seed': args.seed, 'isomap_connected_components': int(components), 'projections': {},
        'caveats': [f'{n_codes} atoms only, not an image embedding cloud.',
                    'Character annotations are Hungarian assignments, not claims of perfect code purity.',
                    'Separate fits across runs; axes/orientations are not aligned.',
                    't-SNE/Isomap geometry depends on hyperparameters; do not infer arithmetic from a layout.',
                    '2D and3D nonlinear fits are separate; PCA2D is the first two PCA3D axes.'],
    }
    pca = PCA(n_components=3, svd_solver='full')
    pca_xyz = pca.fit_transform(x)
    singular = np.linalg.svd(x - x.mean(0), compute_uv=False)
    eigen = singular ** 2
    report['native_centered_singular_values'] = singular.tolist()
    report['participation_ratio_effective_dimension'] = float(eigen.sum() ** 2 / (eigen ** 2).sum())
    report['native_pairwise_distance_min_median_max'] = [float(f(pdist(x))) for f in (np.min, np.median, np.max)]
    np.save(out / 'native_atoms_by_content.npy', x)
    np.savetxt(out / 'native_euclidean_distances.csv', squareform(pdist(x)), delimiter=',',
               header=','.join(alphabet), comments='')
    colors = plt.get_cmap('turbo')(np.linspace(.03, .97, n_codes))
    labels = [f'{char} [c{code}]' for char, code in zip(alphabet, code_ids)]
    overview = plt.figure(figsize=(22, 14))
    arm = run.name.split('__', 1)[-1]
    overview.suptitle(f'{arm} | macro-best epoch {state["epoch"]}\n'
                      f'{n_codes} raw {native_dim}D atoms | Hungarian character [code ID] | mapping {accuracy:.2%}', fontsize=18)
    all_coords = []
    for column, method in enumerate(('PCA', 't-SNE', 'Isomap')):
        for dims in (2, 3):
            params = {}
            if method == 'PCA':
                coordinates = pca_xyz[:, :dims]
                explained = float(pca.explained_variance_ratio_[:dims].sum())
                params = {'svd_solver': 'full', 'explained_variance_ratio': pca.explained_variance_ratio_[:dims].tolist(),
                          'cumulative_explained_variance': explained}
                subtitle = f'Explained variance {explained:.5%}'
            elif method == 't-SNE':
                params = dict(n_components=dims, perplexity=args.perplexity, init='pca',
                              learning_rate=50.0, early_exaggeration=12.0, method='exact',
                              metric='euclidean', random_state=args.seed)
                iteration_key = 'max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
                params[iteration_key] = 3000
                fitted = TSNE(**params).fit(x)
                coordinates = fitted.embedding_
                params.update(kl_divergence=float(fitted.kl_divergence_), fitted_iterations=int(fitted.n_iter_))
                subtitle = f'Perplexity {args.perplexity:g} | seed {args.seed} | exact'
            else:
                params = dict(n_components=dims, n_neighbors=args.neighbors, eigen_solver='arpack',
                              path_method='D', metric='euclidean')
                # ARPACK can use NumPy RNG for its initial vector.
                np.random.seed(args.seed)
                fitted = Isomap(**params).fit(x)
                coordinates = fitted.embedding_
                params['reconstruction_error'] = float(fitted.reconstruction_error())
                subtitle = f'{args.neighbors}-neighbor graph | connected'
            assert coordinates.shape == (n_codes, dims) and np.isfinite(coordinates).all()
            rho = float(spearmanr(pdist(x), pdist(coordinates)).statistic)
            trust = float(trustworthiness(x, coordinates, n_neighbors=3))
            stem = method.lower().replace('-', '') + f'_{dims}d'
            params.update(native_distance_spearman=rho, trustworthiness_k3=trust)
            report['projections'][stem] = params
            projection = '3d' if dims == 3 else None
            fig = plt.figure(figsize=(11, 10))
            ax = fig.add_subplot(111, projection=projection)
            draw(ax, coordinates, labels, colors, method, dims, subtitle)
            fig.suptitle(f'{arm}\n{checkpoint.name} | Hungarian mapping {accuracy:.2%}', fontsize=13)
            fig.text(.5, .02, f'{n_codes} raw {native_dim}D atoms | distance Spearman={rho:.3f}; neighborhood trust(k=3)={trust:.3f}\n'
                     'Independent projection; nonlinear distances are not the original distances.', ha='center', fontsize=10)
            fig.tight_layout(rect=(0, .065, 1, .93))
            separate_labels(fig)
            for ext in ('png', 'svg'):
                fig.savefig(out / f'{stem}.{ext}', dpi=190, facecolor='white')
            plt.close(fig)
            ax = overview.add_subplot(2, 3, column + 1 + (dims - 2) * 3, projection=projection)
            draw(ax, coordinates, labels, colors, method, dims, subtitle)
            for char, code, point in zip(alphabet, code_ids, coordinates):
                all_coords.append([method, dims, char, int(code), *point.tolist(), *([''] if dims == 2 else [])])
    overview.text(.5, .016, 'Top: 2D | Bottom: 3D. Colors identify characters, not learned clusters.\n'
                  'PCA preserves linear variance; nonlinear layouts depend on perplexity / neighbor graph. Compare raw distances too.',
                  ha='center', fontsize=12)
    overview.tight_layout(rect=(0, .06, 1, .92), h_pad=3, w_pad=3)
    separate_labels(overview)
    overview.savefig(out / 'overview.png', dpi=155, facecolor='white')
    overview.savefig(out / 'overview.svg', facecolor='white')
    plt.close(overview)
    with (out / 'coordinates.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['method', 'dimensions', 'content', 'code_id', 'x', 'y', 'z'])
        writer.writerows(all_coords)
    (out / 'geometry_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    assert sha256(checkpoint) == digest, 'Source checkpoint unexpectedly changed'
    assert len(list(out.glob('*_2d.png'))) == len(list(out.glob('*_3d.png'))) == 3
    print('GEOMETRY_COMPLETE', out, flush=True)
    print(json.dumps({'epoch': state['epoch'], 'pc1_variance': float(pca.explained_variance_ratio_[0]),
                      'mapping_accuracy': accuracy, 'pca2_variance': float(pca.explained_variance_ratio_[:2].sum()),
                      'pca3_variance': float(pca.explained_variance_ratio_.sum()),
                      'effective_dimension': report['participation_ratio_effective_dimension']}), flush=True)


if __name__ == '__main__':
    main()
