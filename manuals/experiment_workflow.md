# VVI 实验与 Slurm 操作手册

所有命令默认从项目根目录执行：

```bash
cd /home/xuanjie.liu/Projects/variance-versus-invariance
```

## 1. 计算节点规则

`lo-*` 是 login node，只允许编辑文件、查看 Git/Slurm 状态和提交作业。禁止直接
运行 Python、测试、训练、评估或重计算。

交互资源：

```bash
salloc --partition=ws-ia -N 1 --gres=gpu:1 --mem=32G --cpus-per-task=4
source /home/xuanjie.liu/miniconda3/etc/profile.d/conda.sh
conda activate xuanjie
```

完成后必须执行 `exit`。正式实验优先使用 `sbatch`，作业终止时资源会自动释放。

## 2. 实验台账

正式实验使用三个文件：

- `experiment_log/SETTINGS.md`：setting、假设、固定条件和 decision gate。
- `experiment_log/ACTIVE_RUNS.md`：已提交但尚未验收的正式 runs。
- `experiment_log/ARCHIVE.md`：成功、失败、超时或取消后已验收的 runs。

Smoke run 不进入台账。正式提交前先写 ACTIVE；提交成功后补 Slurm Job ID。
验收时必须把同一 Run ID 从 ACTIVE 删除并写入 ARCHIVE。

## 3. GPU smoke

Smoke 使用正式 config，但 debug 模式只抽取 `debug_portion` 数据并强制训练一个
epoch：

```bash
SMOKE_NAME="smoke-$(date +%Y%m%d-%H%M)-VVI-S1-C4S512-S0"
srun --partition=ws-ia -N 1 --gres=gpu:1 --mem=32G --cpus-per-task=4 \
  --time=00:20:00 \
  bash -lc '
    source /home/xuanjie.liu/miniconda3/etc/profile.d/conda.sh
    conda activate xuanjie
    cd /home/xuanjie.liu/Projects/variance-versus-invariance
    python run_training.py \
      --config cfg_v3_uppercase_s1_c4_s512_seed0.yaml \
      --debug --name "'"$SMOKE_NAME"'"
  '
```

检查 `logs/<smoke-name>/` 中存在 `config.yaml`、checkpoint、
`loss_history.csv`、`loss_epoch_history.csv` 和 `loss_curves.png`。Smoke
目录保留为普通 artifact，不写入实验台账。

## 4. 正式提交

```bash
RUN_ID="VVI-S1-C4S512-S0"
RUN_NAME="$(date +%Y%m%d-%H%M)__${RUN_ID}"
JOB_ID="$(
  scripts/submit_slurm.sh \
    cfg_v3_uppercase_s1_c4_s512_seed0.yaml \
    "$RUN_NAME" \
    "$RUN_ID"
)"
echo "$RUN_NAME $JOB_ID"
```

默认资源为 `ws-ia`、1 GPU、32G 内存、4 CPU、8 小时。作业头部会记录代码
SHA、dirty 状态、config checksum、hostname 和 GPU 信息。

## 5. 监控、取消和失败处理

```bash
squeue -j <job-id> -o "%.18i %.15P %.30j %.8T %.10M %.9l %R"
tail -f slurm_logs/vvi-s1-c4s512-s0-<job-id>.out
tail -f slurm_logs/vvi-s1-c4s512-s0-<job-id>.err
scancel <job-id>
```

失败、超时或人工取消也属于正式终态：从 ACTIVE 移除，在 ARCHIVE 写明状态、
最后 checkpoint、可用指标和失败原因。

## 6. 验收评估

训练完成后在 GPU 节点运行：

```bash
srun --partition=ws-ia -N 1 --gres=gpu:1 --mem=32G --cpus-per-task=4 \
  --time=01:00:00 \
  bash -lc '
    source /home/xuanjie.liu/miniconda3/etc/profile.d/conda.sh
    conda activate xuanjie
    cd /home/xuanjie.liu/Projects/variance-versus-invariance
    python run_codebook_health.py \
      --config logs/<run-name>/config.yaml \
      --run-dir logs/<run-name>
  '
```

未指定 checkpoint 时，评估工具会从 `loss_epoch_history.csv` 中选择 validation
total loss 最低且文件仍存在的 checkpoint。结果写入 run 目录下的
`codebook_health.json`。验收时用相同命令和同一 test 范围重新评估历史
`4/4`、`512/512` checkpoint，再按 `SETTINGS.md` 的 decision gate 归档。
