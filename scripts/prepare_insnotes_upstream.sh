#!/usr/bin/env bash
set -euo pipefail

: "${SLURM_JOB_ID:?Run this preparation as a Slurm job.}"
if [[ $# != 1 ]]; then
    echo "Usage: prepare_insnotes_upstream.sh RUN_NAME" >&2
    exit 2
fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run="$repo/logs/$1"
capsule="$run/reproduction"
upstream="$capsule/upstream"
data="$repo/../data/InsNotesUpstreamSTFTV1"
revision=bbbc7c9c26bda7170612814a407b76633cd22875
soundfont_deb=fluid-soundfont-gm_3.1-5.3_all.deb
soundfont_sha=6f531493ac4e4d9772fd96b2488ea1790af81c196135fbdd25997da0781fc60e
mkdir -p "$capsule"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate xuanjie
export PATH="$HOME/miniconda3/envs/v3_audio_tools/bin:$PATH"
command -v fluidsynth
fluidsynth --version
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

if [[ ! -d "$upstream/.git" ]]; then
    git clone --no-checkout https://github.com/Irislucent/variance-versus-invariance.git "$upstream"
fi
git -C "$upstream" checkout --detach "$revision"
git -C "$upstream" diff --exit-code
git -C "$upstream" rev-parse HEAD > "$capsule/upstream_git_head.txt"

if [[ ! -x "$capsule/venv/bin/python" ]]; then
    python -m venv "$capsule/venv"
fi
py="$capsule/venv/bin/python"
"$py" -m pip install --disable-pip-version-check --timeout 45 --retries 3 \
    'torch==2.0.1' 'torchaudio==2.0.2' 'torchvision==0.15.2' \
    'vector-quantize-pytorch==1.18.0' 'einops==0.8.0' 'einx==0.3.0' \
    'numpy==1.23.5' 'scipy==1.15.2' 'scikit-learn==1.2.2' \
    'librosa==0.10.1' 'midi2audio==0.1.1' 'pretty-midi==0.2.10' 'soundfile==0.12.1' \
    'numba==0.60.0' 'matplotlib==3.7.2' 'seaborn==0.13.2' \
    'Pillow==11.1.0' 'PyYAML==6.0.2' 'tqdm==4.67.0' 'h5py==3.10.0'
"$py" -m pip check
"$py" -m pip freeze > "$capsule/python_packages.txt"

work=$(mktemp -d "$repo/../data/.InsNotesUpstreamSTFT.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
curl --fail --location --retry 2 --max-time 300 \
    "https://deb.debian.org/debian/pool/main/f/fluid-soundfont/$soundfont_deb" \
    --output "$work/$soundfont_deb"
printf '%s  %s\n' "$soundfont_sha" "$work/$soundfont_deb" | sha256sum --check
ar p "$work/$soundfont_deb" data.tar.xz |
    tar -xJOf - ./usr/share/sounds/sf2/FluidR3_GM.sf2 > "$capsule/FluidR3_GM.sf2"
sha256sum "$capsule/FluidR3_GM.sf2" > "$capsule/soundfont.sha256"
cd "$upstream"
ln -sfn "$capsule/FluidR3_GM.sf2" dataset/insnotes/soundfonts/FluidR3_GM.sf2

upstream_module() {
    "$py" -c 'import random, runpy, sys; import numpy as np; random.seed(0); np.random.seed(0); runpy.run_module(sys.argv.pop(1), run_name="__main__", alter_sys=True)' "$@"
}
if [[ ! -e "$data" ]]; then
    upstream_module dataset.insnotes.gen_midi --save_dir "$work/midi"
    upstream_module dataset.insnotes.render_audio --data_dir "$work/midi" --save_dir "$work/wav"
    upstream_module dataset.insnotes.audio_feature_extraction --data_dir "$work/wav" --save_dir "$work/npy"
    upstream_module dataset.insnotes.split_train_val --data_dir "$work/npy" --save_dir "$work/dataset"
    "$py" "$repo/scripts/run_insnotes_upstream.py" audit "$work/dataset" \
        "$capsule/soundfont.sha256" "$revision"
    mv -T -- "$work/dataset" "$data"
else
    "$py" "$repo/scripts/run_insnotes_upstream.py" audit "$data" \
        "$capsule/soundfont.sha256" "$revision"
fi
cp "$data/manifest.json" "$capsule/dataset_manifest.json"
echo "UPSTREAM_INSNOTES_READY $data"
