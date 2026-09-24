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

经本次协议明确批准，PhoneNums PAPER/REPO 将时限设为24小时：
`VVI_TIME=24:00:00 bash scripts/submit_slurm.sh <config> <run-name> <run-id>`。
W50仍使用默认8小时。不得因此修改其他实验的资源预算。

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

已登记的200轮实验每25个completed epochs额外保留永久snapshot（24…199），
因此W50最多10个文件。仅PN-S1启用`save_best_validation_loss: true`，再保留一个
`cp_best_validation_loss_epoch<N>.pt`和`best_validation_loss.json`，最多11个文件。
best-val只接受更低的有限validation total loss，同分保留较早者，无label/health
gate；续训从checkpoint恢复最佳值并保留实际最佳权重，旧格式缺字段时兼容。

### Evaluation 路径解析

`run_evaluation.py` 默认从 run 目录内读取训练时备份的 `config.yaml`，无需再去
根目录匹配原 config：

```bash
# run 名；默认评估 current_checkpoint.json 指向的 checkpoint
python run_evaluation.py --run <run-name> --confusion_mtx

# run 名配 current/best 别名，或配 run 内的具体文件名
python run_evaluation.py --run <run-name> --active_checkpoint best --confusion_mtx
python run_evaluation.py --run <phone-run> --active_checkpoint best_val --confusion_mtx
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
红色表示不一致。新版标注使用不透明白底与深绿/深红文字。新版列归一化矩阵
使用普通字重，`.88`替代`0.88`；两位小数守恒舍入后为零的格子留白（原概率
可能非零，原始CSV/JSON不截断）。手工新版图使用`__viz-v2`后缀，不覆盖旧周期图。

PhoneNums自动产生10×16配对重构图、10×10均衡矩阵。最终验收在同一GPU命令中加
`--figure17`，例如`python run_reconstruction_grid.py --run <phone-run>
--active_checkpoint best_val --figure17`，额外生成10个code×8个style均值的重组图。
均值来自完整test中每种style的原始style embedding；它不是配对重构，不参与
foreground颜色误差计算。此接口目前限定原生VQ维度与decoder content维度一致。

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
默认的 `val_loss` 优先读`best_validation_loss.json`，没有时才从
`loss_epoch_history.csv`选择validation total loss最低且文件存在的checkpoint。
结果写入run目录下的`codebook_health.json`。
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

## 10. 新run目录继承暂停实验

续训config设置正确的`load_checkpoint`及增量`epochs`；例如完成37后，
`epochs: 162`会从38继续到199。`inherit_resume_history: true`用于新run目录：
复制旧CSV中不超过已完成epoch的记录，排除中断轮；原始快照/图保留在父run。
若checkpoint记有macro-best，必须复制那个真实best文件及metadata，不能把
current权重冒充旧best。`resume_lineage.json`记录源路径/hash、历史行数和旧快照。
已有目标history不会被静默覆盖，缺少源checkpoint或best会明确报错。

所有optimizer/scheduler/scaler状态按checkpoint恢复。当前旧checkpoint未保存
RNG状态，因此不能宣称数据shuffle或完整训练轨迹逐位连续。W100切换仍按绝对
epoch100执行。准备新run时检查probe fit/score/grid协议一致，并在GPU smoke中
验证实际首轮和学习率；smoke后清理产物，正式提交前更新ACTIVE。

## 11. UppercaseLettersV2 与颜色/均衡矩阵

仅在 ws-* allocation 内生成数据（默认输出目录已存在时拒绝覆盖）：

```bash
python -m dataset.uppercase_letters.generate_v2 \
  --output_root ../data/UppercaseLettersV2 \
  --generation_seed 0 --split_seed 0 --workers 4
```

该命令必须放在前述`srun`/`sbatch`的conda环境中，不能在login直接运行。
生成到临时兄弟目录，校验后整体发布；失败的staging保留待检查，不自动覆盖。
V2修复signed float加噪clip后转uint8，以及blur末strip遗漏；legacy默认保持。
manifest包含26,000张PNG路径/hash、逐页seed/字母顺序、split/style计数、生成器
与字体hash、依赖版本和Git状态。`dataset_manifest_sha256`使Trainer和验收程序
检查manifest版本；loader按manifest固定顺序读取，train仍按原seed shuffle。
`record_initial_model_hash: true`记录前向前完整参数/buffer hash，核对匹配初始化。

`disentanglement_probes`新增两个开关：`foreground_color_metrics: true`与
`balanced_confusion_matrix: true`。每次完整validation复用输出记录颜色CSV/PNG，
日志及WandB；每个重构节点同时输出mask检查图和列归一化SVG/PNG/CSV/JSON。
V2全validation/test各2600页，每个(letter,style)325个fragment；数量不均衡就报错。
矩阵为P(code|content)，每列和1；不改变旧row-normalized绘图与purity定义。
图中两位小数用列内守恒舍入，原始counts/probabilities/mapping存JSON/CSV。

mask仅从实际输入估计：外沿2px通道中位数背景、内部最大通道差>.15、至少16px。
RGB MAE/chroma RMSE先逐fragment算，再对有效fragment等权平均并分style报告。
空mask记invalid/NaN；预测不clamp，另报越界比例。不是clean-target误差或Lab ΔE，
仍受噪声与mask估计影响。颜色不进入loss、scheduler或best排名。

验收V2 best/current时，在GPU节点分别运行现有health命令（新JSON包含颜色）及：

```bash
python run_reconstruction_grid.py --run <run-name> --active_checkpoint best
python run_reconstruction_grid.py --run <run-name> --active_checkpoint current
```

这会输出完整test的均衡矩阵和颜色指标，以及固定8style×26字母重构/mask图。
每个checkpoint有独立文件前缀；mapping与重构注释共用。不要以quick子集替代验收。
S5保留原BN/optimizer/预算，完成后先归档两run，再更新RQ2与dashboard。

## 12. PhoneNumsV2 双协议

只能在ws-* GPU allocation内生成：

```bash
python -m dataset.phonenums.generate_v2 \
  --output_root ../data/PhoneNumsV2 \
  --generation_seed 0 --split_seed 0 --workers 4
```

目录已存在则拒绝覆盖。100,000页320×48 RGB，每页0–9各一次；每style
train10,000、val1,250、test1,250页，每个split内content/style严格均衡。
NumberOffice原字体布局、浅背景245和修正版renderer；排列在每style内无放回
抽取，图像独立RNG与split RNG分离。发布前检查唯一性、尺寸、RGB、逐PNG SHA256。
manifest固定顺序，保存seed、计数、源码/字体hash与环境；历史PhoneNums不改动。
当前manifest SHA256：
`1e54fa7ce5c28255c0e4c7a22f428a8aae3005147fac8faa2dd1ab7d9f2c9fce`。

PAPER/REPO配置在`configs/phonenums/v2/`；必须同时固定该manifest、初始化seed
和训练顺序。分别使用Adam/AdamW，四个V3子项权重.1/1，commit .01/.1，
EMA .95/.98，dead-code threshold3.2/16；不是单因素对照。两版200轮预算，
与上游10,000轮上限区分。验收主要读`best_val`，另外报告`best`和`current`；
完整test为10,000页，不得用512快速子集替代。Table8的89.2%按macro定义对照，
同时保留weighted/Hungarian，不能混称三者为同一个accuracy。

## 13. S7 训练 usage 与只读 BN 诊断

两个 C512 W0/W50 config 位于 `configs/uppercase/rq2/v2/`，从零训练200轮。
提交时显式使用 `VVI_TIME=12:00:00 bash scripts/submit_slurm.sh <config> <run> <id>`；
其余仍是 ws-ia/1GPU/32GB/4CPU。先登记 ACTIVE，提交后回填 Job ID/节点/hash。
不恢复已归档的 C128-W50，不替换训练 BN，也不改变 loss/optimizer。

可选配置（不设置则旧 run 行为不变）：

```yaml
training_usage_monitor:
  enabled: true
  every_n_steps: 100
bn_diagnostic:
  enabled: true
  every_n_epochs: 1
  pages_per_style: 4
  seed: 0
```

`training_codebook_usage.csv/.png` 记录在线模型每100步窗口及整轮的 active、
perplexity、最大 code 占比；轮尾不足100步也写入。不是固定 checkpoint 的统计，
不要与 validation purity 混称。`training_codebook_usage_protocol.json` 说明口径。

`bn_diagnostic_protocol.json` 固定32页（每style4页）及 hash；从已有 validation
批次收集，不重新采样。`bn_diagnostic_epoch_history.csv` / `bn_diagnostics.png`
每轮比较正常 eval、encoder-only、decoder-only、both BN batch statistics。
图中 fixed-mapping accuracy 使用当轮**完整 validation**的 Hungarian mapping；
code_changed_fraction 相对正常 eval，decoder-only 必须严格为0。
每个模式记录重构、前景颜色、usage，CSV还包含分style颜色和最大 code 占比。
模型整体 eval、VQ 冻结，只切换指定 BN；异常时也恢复所有 buffer、mode 和 RNG。
模型参数和 optimizer 不更新。小批量干预不参与正式指标或 checkpoint 选择。

S7 每25轮快照，加 current、macro-best 和可选 best-val，200轮最多11文件。
额外0/10/11/12/15/20/50/100重构/矩阵节点不多存永久 checkpoint。
`best_val` 别名用于重构对照；验收仍报告正常 full-test2600页的三类checkpoint，
宏纯度与 usage 为语义主指标，颜色及 BN 差异单列。若诊断显示明显 BN 敏感性，
下一轮注册独立归一化对照，不在当前 run 中途静默切换架构。

## 14. S8 decoder BN / GN 独立对照

`model_config.decoder_normalization` 默认为 `batch`，旧 config 完全兼容。
设为 `group` 时配合 `decoder_groupnorm_groups: 8`：只替换 decoder 内所有
BN（含 shortcut），encoder 不变。组数必须整除每层通道数，否则报错；不静默
更改组数。保留 eps 和 affine 参数初值，但丢弃 BN running buffers，故不能把
训练好的 BN checkpoint 直接按 GN 加载；S8 两组均从零训练。

两组共同使用 W50 和 lr1e-4，配对只改变 decoder normalization。S8 BN 对比历史
S7 的差异是 lr，不能把两种对比混为一个因素。每个 run 的
`normalization_config.json` 记录层配置；`initial_model_state.json` 同时记录完整
state hash 和 `parameters_sha256`。BN/GN 的全 state hash 应不同，可训练参数
hash 应相同；固定采样顺序也必须核对。

新增诊断选项 `bn_diagnostic.report_normalization_layers: true` 只对新 run 启用，
在 CSV 写入实际 BN 层数及切换数量。GN 组的 decoder-only BN 模式为明确的
no-op，图例注明；both 模式只切 encoder BN。零 decoder-mode gap 是设计属性，
不等于更好重构。必须比较正常 eval 的颜色/重构与语义指标，不能替换验收口径。
其余保持 S7 的200轮、11 checkpoint 上限及12h提交配额，smoke 验证后立即删除。

## 15. S9 NoNorm 消融与尺度/逐 style 诊断

新配置 `model_config.encoder_normalization: batch|none`，decoder 在原有
`batch|group` 外支持 `none`。`none` 使用 Identity 替换该 backbone 中所有 BN，
包括 shortcut；不加其他归一化、残差缩放或新初始化。encoder 的两个未使用
BN head 也移除，但真实消融必须覆盖实际 CNN。固定图像[-1,1]缩放和 V3 比值
公式不变。旧配置默认 BN，不允许用新架构静默加载旧 BN checkpoint。

S9 两组为 Decoder-NoNorm 和 All-NoNorm，共同从零训练200轮/W50/lr1e-4。
构造完整原模型后才移除 norm，保持公共 Conv/Linear/VQ 初始化与 RNG 一致。
BN affine 参数随层移除，所以比较 `initial_model_state.json` 内的
`non_normalization_parameters_sha256`，不能要求全参数 hash 相同。

`optimization_diagnostics.enabled: true`，默认每100 global steps及每轮最后
一步采样 activation/latent RMS、max、L2，以及 unscaled encoder/decoder
gradient L2/max；写 `optimization_scale_history.csv`、
`optimization_scale_epoch_history.csv`、`optimization_scales.png`。epoch统计是
采样步的均值，不冒充全轮统计。forward hooks 只读 detached 值，不改输出、
梯度、optimizer 或 RNG；不自动裁剪/调整学习率。

`per_style_codebook_monitor.enabled: true` 复用完整 validation indices，验证
每个(content,style)325个fragment，所有style共用全局 Hungarian mapping。
输出 `per_style_codebook_epoch_history.csv`、`per_style_codebook_metrics.png`，
以及 `per_style_codebook/counts_epochNNN__val.json` 的原始 counts 与 mapping。
单独记录各style的macro/weighted、usage、最大code占比，防止聚合macro掩盖
单个style整体塌缩。此诊断不改变macro-best门槛或历史指标。

All-NoNorm 四种 BN 干预全部是 no-op，Decoder-NoNorm 仅 decoder-only 是
no-op；图例显示无匹配 BN。归一化被移除不代表语义或颜色已经恢复。
正式任务仍各1GPU/32GB/4CPU/12h，保留每25轮快照和current/macro-best/best-val，
上限11。GPU预检通过并清理smoke后才提交，不在login执行Python或训练。

## 16. HexDigitsV2 与永久 page-mean 对照

数据生成入口（只能在已分配的 GPU compute node 内运行）：

```bash
python -m dataset.hex_digits.generate_v2 --output_root ../data/HexDigitsV2 --generation_seed 0 --split_seed 0 --workers 4
```

先通过数据测试，再生成；已有目录拒绝覆盖。每页16个小写十六进制字符，标签
用整数0..15，不用十进制 `int(c)` 解析 a–f。新 loader 为 `hex_digits_dataloader`，
按 manifest 排序，严格核对RGB/512×48/完整16个fragment。历史PhoneNums不变。
生成器使用O(页数)的无放回 rank采样，不枚举16!。正式数据100k页、8style均衡，
每个(content,style)在val/test各1250次。正式提交前将manifest SHA写入两份config；
`pending-generation` 占位值不能用于训练。完整数据校验及预览保存在diagnostics。

HEX-S1配置位于 `configs/hex/v2/`，REPO普通fragment与MEAN永久page_mean唯一
训练差异是decoder style输入；均恢复BN/lr.001/C512/K16，原始fragment z_s仍
用于V3。没有epoch50切换，也不把旧BN/NoNorm checkpoint混用。

分组诊断 `content_group_epoch_history.csv` / `content_group_metrics.png` 使用
**同一个完整validation Hungarian mapping**比较0–9与a–f；绝不能对子集重新
匹配。手工evaluation和reconstruction/matrix JSON保存同样的group/per-content
指标；evaluation_history.csv在Hex run额外保存两个group mapping accuracy列。
宏纯度仍用于checkpoint选择，group/readout/BN指标不参与loss或选择。

GPU smoke：`python scripts/smoke_hex.py --arm repo` / `--arm mean`，每个完整
2500训练步+10000页validation，验证后检查16×16配对图/矩阵并立即删除对应的
`logs/smoke-hex-s1-*`，不得入正式台账。正式任务通过原提交入口，设置
`VVI_TIME=24:00:00`，各1GPU/32GB/4CPU；保留current/macro-best/best-val加8快照。
完整test160000fragments，三个checkpoint角色均验收；最佳过关而current失败
只记短暂成功，不能据此直接部署单图解码或十六进制加法器。

## 17. Hex验收与S10字母复跑

所有命令仍必须位于GPU compute allocation。`python scripts/accept_hex.py`
验收HEX-S1两组best/best-val/current：完整10k页test，保存hash、逐style/字符/
分组指标、彩色重构与均衡矩阵；可用 `--arm repo` 或 `--arm mean` 单独处理。
它只写新的acceptance目录，不改原checkpoint。报告已存在时核对hash后复用。

`python scripts/audit_hex_uppercase.py` 比较S7与HEX保存的配置/代码和实际输入，
避免把两组误称为“只差content数量”：UppercaseLettersV2是25px字形、20,800
训练页/650步每轮；Hex是48px、80,000训练页/2500步每轮。相同200轮不是相同
更新预算；scheduler按epoch衰减也不是按step匹配。

S10 config：`configs/uppercase/rq2/v2/*hexrepo*.yaml`，普通fragment与永久mean，
复用旧UppercaseLettersV2，不重新生成或覆盖。默认200轮/130k步；若批准延长
预算，必须先更新setting/config和checkpoint保留策略，不能只把epoch改大却
宣称与Hex的LR进度相同。GPU smoke用 `python scripts/smoke_s10.py --arm repo`
和 `--arm mean`：每组完整650步+2600页validation，验收后立即删除对应smoke。
正式各1GPU/32GB/4CPU/24h，current/macro-best/best-val加每25轮快照，200轮max11。

2026-09-17用户批准S10延长：使用`*hexrepo*_seed0_resume1.yaml`从current199
续570轮至769/累计500500steps，保持原epoch-based LR，不重置训练状态。
GPU预检入口为`python scripts/smoke_s10_resume.py --arm repo`和`--arm mean`。
记录parent checkpoint hash和无RNG恢复的限制；R1继承所有诊断CSV和两个best，
不复制父快照。新目录保留22个周期快照加current/macro/best-val，最多25文件。
ws-ia的MaxTime为24h；若超时只能按实际完成epoch归档，再从current恢复剩余
预算，不覆盖原run，不将目标epoch写成已完成。提交仍用`VVI_TIME=24:00:00`
和现有入口；不自动取消其他任务。
