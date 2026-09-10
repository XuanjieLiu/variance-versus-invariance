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

研究台账分三层：

- `experiment_log/README.md`：三个研究方向的 dashboard 和下一决策。
- `experiment_log/directions/*.md`：跨 run 的假设、证据综合和 backlog。
- `experiment_log/SETTINGS.md`：setting、固定条件和 decision gate。
- `experiment_log/ACTIVE_RUNS.md`：已提交但尚未验收的正式 runs。
- `experiment_log/ARCHIVE.md`：成功、失败、超时或取消后已验收的 runs。

Smoke run 不进入台账。验证完成后立即删除对应的 `logs/smoke-*` 目录，避免
checkpoint 长期占用空间。正式提交前先写 ACTIVE；提交成功后补 Slurm Job ID。
验收时必须把同一 Run ID 从 ACTIVE 删除并写入 ARCHIVE。

ACTIVE/ARCHIVE 只写简洁生命周期信息；长结论只更新到对应 direction 文件，
避免三处互相漂移。每个 setting/run 都要标明 Direction 和 Hypothesis ID。

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

检查 `logs/<smoke-name>/` 中存在 `config.yaml`、checkpoint、loss、codebook
和 V3 ratio 三套 CSV/PNG。启用 `save_best_macro_atom_purity` 时还必须生成
`cp_best_macro_atom_purity_epoch<N>.pt` 和 `best_macro_atom_purity.json`。
验证通过或失败原因确认后，立即精确删除这个 smoke 目录；不要写入实验台账。

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

### Checkpoint 保留规则

新 run 使用 `checkpoint_policy: current_and_best_macro`，最多保留两个模型文件：

- `cp_current_epoch<N>.pt`：每轮替换上一个 current，并由
  `current_checkpoint.json` 指向。
- `cp_best_macro_atom_purity_epoch<N>.pt`：validation macro atom purity 提升时
  替换；同分时 validation total loss 更低者胜出，并由
  `best_macro_atom_purity.json` 指向。

即使 current 和 best 来自同一 epoch，也保留两个独立物理文件，便于稳定续训
和验收。历史 run 不追溯清理；旧 checkpoint 续训时可在 config 用
`resume_macro_atom_purity` 和 `resume_macro_val_loss` 初始化已有最佳值。

### Evaluation 路径解析

`run_evaluation.py` 默认从 run 目录内读取训练时备份的 `config.yaml`，无需再去
根目录匹配原 config：

```bash
# run 名；默认评估 current_checkpoint.json 指向的 checkpoint
python run_evaluation.py --run <run-name> --confusion_mtx

# run 名配 current/best 别名，或配 run 内的具体文件名
python run_evaluation.py --run <run-name> --active_checkpoint best --confusion_mtx
python run_evaluation.py --run <run-name> \
  --active_checkpoint cp_current_epoch150.pt --confusion_mtx

# 完整 checkpoint 路径；从父目录推断 config.yaml
python run_evaluation.py \
  --active_checkpoint logs/<run-name>/cp_current_epoch150.pt \
  --confusion_mtx
```

`--run` 也接受 run 目录的相对或绝对路径。显式 `--config` 仍兼容且优先级
最高；同时给出 run 和 run 外 checkpoint 会报错，防止模型与 config 错配。

`macro_atom_purity` 是每个 active atom 独立选择 dominant content 后的 purity
均值，允许多个 atom 指向同一个 content；旧字段 `legacy_codebook_accuracy`
是它的兼容别名。后续实验以 macro atom purity 为 checkpoint 选择主指标，
Hungarian one-to-one accuracy 仅作为诊断。扩大码本时必须同时检查 active
codes、usage perplexity、coverage 和 usage-weighted purity，避免低频 atom
造成表面高分。

日常快速检查使用固定的 512-sample、style-stratified test 子集：

```bash
srun --partition=ws-ia -N 1 --gres=gpu:1 --mem=32G --cpus-per-task=4 \
  --time=00:20:00 \
  bash -lc '
    source /home/xuanjie.liu/miniconda3/etc/profile.d/conda.sh
    conda activate xuanjie
    cd /home/xuanjie.liu/Projects/variance-versus-invariance
    python run_evaluation.py \
      --run <run-name> \
      --active_checkpoint best \
      --confusion_mtx \
      --test_subset_size 512 \
      --test_subset_seed 0 \
      --test_subset_strategy style_stratified
  '
```

输出使用 checkpoint 和评估范围唯一命名，同时保存 SVG、PNG、JSON，并在
run 根目录追加 `evaluation_history.csv`。训练 validation 复用完整 validation
前向，将指标写入 `codebook_epoch_history.csv` 和 `codebook_metrics.png`；四个
V3 原始 ratio 另写入 `v3_ratio_epoch_history.csv` 和 `v3_ratios.png`。

快速子集只用于趋势监控。正式验收不传任何 subset 参数，始终使用完整 test：

需要逐 fragment 诊断重构时，可以先用完整 test 建立 Hungarian code-label
mapping，再从每个 style 取一个样本生成原图/重构并排网格：

```bash
srun --partition=ws-ia -N 1 --gres=gpu:1 --mem=32G --cpus-per-task=4 \
  --time=00:30:00 \
  bash -lc '
    source /home/xuanjie.liu/miniconda3/etc/profile.d/conda.sh
    conda activate xuanjie
    cd /home/xuanjie.liu/Projects/variance-versus-invariance
    python run_reconstruction_grid.py \
      --run <run-name> \
      --active_checkpoint best
  '
```

输出 PNG、Hungarian mapping CSV 和逐格 JSON。重构格标注
`quantized code -> Hungarian mapped label`；绿色表示与该行真实 content 一致，
红色表示不一致。

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
      --run-dir logs/<run-name> \
      --selection-metric macro_atom_purity
  '
```

`--selection-metric macro_atom_purity` 会读取 `best_macro_atom_purity.json`；
默认的 `val_loss` 模式仍从 `loss_epoch_history.csv` 选择 validation total loss
最低且文件存在的 checkpoint。结果写入 run 目录下的 `codebook_health.json`。
验收时用相同 test 范围评估历史 checkpoint，再按 decision gate 归档。

## 7. 从 checkpoint 续训

checkpoint 中的 epoch 表示已经完成的 epoch；例如加载 `cp_epoch15.pt` 会从
epoch 16 开始。`epochs` 保持“额外训练轮数”的语义，因此从 epoch 16 续到
epoch 149 时设置 `epochs: 134`。新 checkpoint 同时保存 optimizer、scheduler
和 GradScaler；旧 checkpoint 没有后两项时会按完成 epoch 重建 scheduler。
启用 macro-best 的新 checkpoint 还会保存当前最佳 macro purity、对应 validation
loss 和 epoch，续训时恢复这些状态。

## 8. Objective schedule 与码本扩展

`loss_schedules` 支持 `relativity`、`weights.recon_loss` 和
`weights.commit_loss` 的绝对 epoch 分段线性 knots。Resume 不会把 epoch 重新从
零计；每轮实际值写入 `objective_schedule_epoch_history.csv` 并画到
`objective_schedules.png`。

`checkpoint_transform.type: expand_ema_codebook` 目前只允许单个 EMA Euclidean
codebook 的 2x 扩展。它保存 `initialization.json`（源 checkpoint 哈希、shape、
jitter 和 EMA mass），配置 `validate_before_training: true` 时还会在第一次更新前
写入完整 `initialization_metrics.json`。变换后 macro-best 状态必须重置。

`checkpoint_transform.type: pca_project_and_expand_ema_codebook` 先对源 atoms
执行无标签、非加权 float64 PCA，再在 native D 维空间做 2x 对称拆分。PCA
component 的符号固定，`project_in/project_out` 可通过
`freeze_vq_projection: true` 全程冻结。使用
`reset_training_state: true` 时只复制模型语义状态，optimizer、scheduler、scaler、
epoch 和 macro-best 都从头开始。初始化 validation 也可成为健康门槛内的
macro-best，以免第一轮更新破坏优秀初始化。

冗余码本验收除 purity、usage 和 dominant-code balance 外，还要检查 native VQ
空间的 alias within/between ratio 与 nearest-same margin。随后在 v5 使用
`operation_space: native_vq`；加法器直接读取 D 维 atoms，512D lifted content
只供 VVI decoder 使用。正式 addition probe 冻结 VVI，训练目标只来自观测到的
`x_c` VQ code，数字标签仅用于 number-accuracy 诊断。

## 9. Page-mean decoder、泄漏 probes 与周期快照

`model_config.decoder_style_mode` 支持 `fragment`、`page_mean` 和
`page_mean_warmup`；最后一种配合 `decoder_style_warmup_epochs: 100`，
epoch0–99使用page mean，epoch100起恢复逐fragment。只改decoder输入，
不改变原始`emb_s`、V3 loss输入、EMA、optimizer或scheduler。新增run的
model state保存`_decoder_epoch`，手工evaluation加载ckpt即可恢复对应模式。
旧config/旧checkpoint不增加该字段，继续兼容。

启用`disentanglement_probes.enabled`和`loss_config.monitor_raw_mpd`后，
每轮复用full validation记录detached probes，不增加模型前向。协议及固定
fit/score/grid文件名写入`disentanglement_probe_protocol.json`；结果写入
`disentanglement_probe_history.csv`和`disentanglement_probes.png`。
raw/whitened都保留；随机水平不是“无信息”的证明，要看style→style正向对照。

VVI-RQ2-S3按用户要求设置`snapshot_every_n_epochs: 25`，完成第25/50/…/200
轮时保留`cp_snapshot_epoch24.pt`/`cp_snapshot_epoch49.pt`/…/
`cp_snapshot_epoch199.pt`。不参与current/best轮换删除，每个run最多
8个周期快照+current+best共10个ckpt；不对旧run追溯创建或删除快照。

`reconstruction_diagnostics/`保存同epoch的26行×16列原图/recon配对图、
逐格JSON和Hungarian mapping CSV，mapping来自当轮完整validation。epoch0和100
额外输出图；epoch99的周期图为切换前最后一轮图。三个run使用相同图片。
切换和checkpoint保存均用绝对epoch，resume不能重新开始warmup。
MEAN解码要求页内fragment共享style，不得用于V5混合style triplet。
