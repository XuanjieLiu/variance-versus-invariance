# V3 Lowercase Letters 操作手册

本文档记录 lowercase letters 数据集的生成、切分、训练和评估流程。

所有命令默认在项目根目录运行：

```bash
cd /home/xuanjie.liu/Projects/variance-versus-invariance
```

## 1. 环境依赖检查

```bash
python -m pip install vector_quantize_pytorch==1.18.0
```

作用：安装 V3 模型需要的向量量化依赖。如果当前环境已经安装，可以跳过。

```bash
python -m py_compile dataset/lowercase_letters/letter_office.py dataset/lowercase_letters/split_train_val.py dataloader/lowercase_letters_dataloader.py model/factory.py
```

作用：快速检查新增代码是否有 Python 语法错误。

## 2. 生成 lowercase letters 原始图片

如果之前已经用旧代码生成过数据，且图片里字母边缘被裁掉，先换一个新的输出目录，或者手动清空旧目录后重新生成。当前生成器会按 `a-z` 的最大字符 bbox 自动选择统一字号并居中，避免宽字母或带长笔画的字母被切边。

```bash
python -m dataset.lowercase_letters.letter_office --save_dir ../data/lowercase_letters_ --n_pages 26000
```

作用：生成原始 PNG 数据。每张图片包含一次完整的 26 个小写字母 `a-z`，顺序随机打乱；同一张图使用同一个颜色作为 style。

输出目录：

```text
../data/lowercase_letters_
```

文件名格式示例：

```text
knixdtcpbgyvzhswmquefrjoal_black.png
```

其中下划线前是字母序列，下划线后是 style label。

## 3. 切分 train / val / test

```bash
python -m dataset.lowercase_letters.split_train_val --data_dir ../data/lowercase_letters_ --save_dir ../data/LowercaseLetters --val_percentage 0.1 --test_percentage 0.1
```

作用：把原始 PNG 数据切成训练、验证、测试三份。

输出目录：

```text
../data/LowercaseLetters/train
../data/LowercaseLetters/val
../data/LowercaseLetters/test
```

## 4. 训练前 smoke test

```bash
python run_training.py --config cfg_v3_lowercase_letters.yaml --debug --epochs 1 --batch_size 2 --num_workers 0 --name lowercase_letters_smoke
```

作用：先跑一个很小的训练任务，检查 dataloader、模型尺寸、loss、checkpoint 写入是否正常。

输出目录：

```text
./logs/lowercase_letters_smoke
```

## 5. 正式训练

```bash
python run_training.py --config cfg_v3_lowercase_letters.yaml --name lowercase_letters_run1
```

作用：使用 `cfg_v3_lowercase_letters.yaml` 正式训练 lowercase letters 版本的 V3。

输出目录：

```text
./logs/lowercase_letters_run1
```

checkpoint 会保存为：

```text
./logs/lowercase_letters_run1/cp_epoch<N>.pt
```

## 6. 断点续训

```bash
python run_training.py --config cfg_v3_lowercase_letters.yaml --name lowercase_letters_run1_resume --load_checkpoint ./logs/lowercase_letters_run1/cp_epoch<N>.pt
```

作用：从已有 checkpoint 继续训练。把 `<N>` 替换成实际 epoch 编号。

## 7. 评估

```bash
python run_evaluation.py --config cfg_v3_lowercase_letters.yaml --active_checkpoint ./logs/lowercase_letters_run1/cp_epoch<N>.pt --pr_metrics --confusion_mtx
```

作用：评估指定 checkpoint 的 content/style disentanglement 表现，并输出 precision-recall metrics 和 confusion matrix。

```bash
python run_evaluation.py --config cfg_v3_lowercase_letters.yaml --active_checkpoint ./logs/lowercase_letters_run1/cp_epoch<N>.pt --vis_tsne
```

作用：生成 t-SNE 可视化。这个命令可能比较慢，建议单独运行。

## 8. 快速检查 dataloader 输出

```bash
python - <<'PY'
from dataloader.lowercase_letters_dataloader import get_dataloader, C_LIST, S_LIST

loader = get_dataloader(
    "../data/LowercaseLetters/train",
    batch_size=2,
    n_fragments=26,
    fragment_len=32,
    num_workers=0,
    shuffle=False,
)
batch, c_labels, s_labels = next(iter(loader))
print("C_LIST:", len(C_LIST), C_LIST[:3], C_LIST[-3:])
print("S_LIST:", len(S_LIST), S_LIST)
print("batch:", tuple(batch.shape), batch.dtype)
print("content:", tuple(c_labels.shape), c_labels.min().item(), c_labels.max().item())
print("style:", tuple(s_labels.shape), s_labels.min().item(), s_labels.max().item())
PY
```

期望输出形状：

```text
batch: (2, 26, 3, 48, 32)
content: (2, 26), label range 0..25
style: (2, 26), label range 0..7
```

## 9. 本地实验记录目录

训练记录默认全部保存在本地，不会自动上传云端。每个实验对应一个独立目录：

```text
./logs/<experiment_name>/
```

`<experiment_name>` 来自训练命令里的 `--name`；如果没有传 `--name`，则使用 yaml 里的 `name` 字段。

例如：

```bash
python run_training.py --config cfg_v3_lowercase_letters.yaml --name lowercase_letters_run1
```

会写入：

```text
./logs/lowercase_letters_run1/
```

目录内容：

```text
logs/lowercase_letters_run1/
  config.yaml          # 本次运行实际使用的配置备份
  log.txt              # 训练/验证 loss 和 checkpoint 日志
  cp_epoch<N>.pt       # checkpoint
  vis/                 # 评估可视化输出，跑 evaluation 后才会出现
  ood/                 # OOD 评估输出，跑 OOD evaluation 后才会出现
```

checkpoint 文件里包含：

```text
epoch
model
optimizer
```

其中 `model` 是模型参数，`optimizer` 是优化器状态。

## 10. 查看单个实验

查看 loss 日志：

```bash
less logs/lowercase_letters_run1/log.txt
```

实时观察正在训练的实验：

```bash
tail -f logs/lowercase_letters_run1/log.txt
```

只看训练 loss：

```bash
grep "TRAIN" logs/lowercase_letters_run1/log.txt
```

只看验证 loss：

```bash
grep "VALIDATION" logs/lowercase_letters_run1/log.txt
```

查看本次实际配置：

```bash
less logs/lowercase_letters_run1/config.yaml
```

列出 checkpoint：

```bash
ls -lh logs/lowercase_letters_run1/cp_epoch*.pt
```

查看 checkpoint 内容：

```bash
python - <<'PY'
import torch

path = "logs/lowercase_letters_run1/cp_epoch<N>.pt"
ckpt = torch.load(path, map_location="cpu")
print("keys:", ckpt.keys())
print("epoch:", ckpt["epoch"])
print("model tensors:", len(ckpt["model"]))
print("optimizer keys:", ckpt["optimizer"].keys())
PY
```

把 `<N>` 替换成实际 epoch 编号。

## 11. 对比多个实验

列出所有实验目录：

```bash
find logs -maxdepth 1 -mindepth 1 -type d | sort
```

对比两个实验的配置差异：

```bash
diff -u logs/lowercase_letters_run1/config.yaml logs/lowercase_letters_run2/config.yaml
```

快速抽取每个实验最后一次 validation loss：

```bash
for d in logs/lowercase_letters_*; do
  [ -f "$d/log.txt" ] || continue
  echo "$d"
  grep "VALIDATION" "$d/log.txt" | tail -1
done
```

快速抽取每个实验最小 validation loss：

```bash
python - <<'PY'
import glob
import os
import re

pattern = re.compile(r"VALIDATION - Epoch \\[(\\d+)/.*Loss: ([0-9.]+)")
for log_path in sorted(glob.glob("logs/lowercase_letters_*/log.txt")):
    vals = []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                vals.append((int(m.group(1)), float(m.group(2))))
    if vals:
        best_epoch, best_loss = min(vals, key=lambda x: x[1])
        print(f"{os.path.dirname(log_path)} best_epoch={best_epoch} best_val_loss={best_loss:.6f}")
PY
```

## 12. 可视化 loss 曲线

把某个实验的 train/validation loss 画成 PNG：

```bash
python - <<'PY'
import os
import re

import matplotlib.pyplot as plt

exp = "logs/lowercase_letters_run1"
train_pat = re.compile(r"TRAIN - Epoch \\[(\\d+)/.*Step \\[(\\d+)/.*Loss: ([0-9.]+)")
val_pat = re.compile(r"VALIDATION - Epoch \\[(\\d+)/.*Loss: ([0-9.]+)")

train_x, train_y = [], []
val_x, val_y = [], []

with open(os.path.join(exp, "log.txt")) as f:
    for line in f:
        m = train_pat.search(line)
        if m:
            train_x.append(float(m.group(1)) + float(m.group(2)) / 100000)
            train_y.append(float(m.group(3)))
        m = val_pat.search(line)
        if m:
            val_x.append(int(m.group(1)))
            val_y.append(float(m.group(2)))

plt.figure(figsize=(8, 4))
plt.plot(train_x, train_y, label="train")
plt.plot(val_x, val_y, label="val")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
out = os.path.join(exp, "loss_curve.png")
plt.savefig(out, dpi=160)
print(out)
PY
```

输出：

```text
logs/lowercase_letters_run1/loss_curve.png
```

画多个实验的 validation loss 到同一张图：

```bash
python - <<'PY'
import glob
import os
import re

import matplotlib.pyplot as plt

val_pat = re.compile(r"VALIDATION - Epoch \\[(\\d+)/.*Loss: ([0-9.]+)")

plt.figure(figsize=(8, 4))
for log_path in sorted(glob.glob("logs/lowercase_letters_*/log.txt")):
    epochs, losses = [], []
    with open(log_path) as f:
        for line in f:
            m = val_pat.search(line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    if losses:
        plt.plot(epochs, losses, label=os.path.basename(os.path.dirname(log_path)))

plt.xlabel("epoch")
plt.ylabel("validation loss")
plt.legend(fontsize=8)
plt.tight_layout()
out = "logs/lowercase_letters_val_loss_compare.png"
plt.savefig(out, dpi=160)
print(out)
PY
```

输出：

```text
logs/lowercase_letters_val_loss_compare.png
```

## 13. 查看评估可视化

生成 confusion matrix：

```bash
python run_evaluation.py --config cfg_v3_lowercase_letters.yaml --active_checkpoint ./logs/lowercase_letters_run1/cp_epoch<N>.pt --confusion_mtx
```

输出：

```text
logs/lowercase_letters_run1/vis/codebook_confusion_matrix.svg
```

生成 t-SNE：

```bash
python run_evaluation.py --config cfg_v3_lowercase_letters.yaml --active_checkpoint ./logs/lowercase_letters_run1/cp_epoch<N>.pt --vis_tsne
```

输出：

```text
logs/lowercase_letters_run1/vis/tsne_c_label_c.svg
logs/lowercase_letters_run1/vis/tsne_c_label_s.svg
logs/lowercase_letters_run1/vis/tsne_s_label_c.svg
logs/lowercase_letters_run1/vis/tsne_s_label_s.svg
```

打开可视化文件：

```bash
xdg-open logs/lowercase_letters_run1/vis/codebook_confusion_matrix.svg
```

如果在远程机器上没有图形界面，可以把 SVG 转成 PNG 后查看：

```bash
python - <<'PY'
import cairosvg

cairosvg.svg2png(
    url="logs/lowercase_letters_run1/vis/codebook_confusion_matrix.svg",
    write_to="logs/lowercase_letters_run1/vis/codebook_confusion_matrix.png",
)
PY
```

需要先安装：

```bash
python -m pip install cairosvg
```

注意：`--pr_metrics` 当前只会把结果打印到终端，不会保存成文件。如果需要长期记录，建议这样运行：

```bash
python run_evaluation.py --config cfg_v3_lowercase_letters.yaml --active_checkpoint ./logs/lowercase_letters_run1/cp_epoch<N>.pt --pr_metrics --confusion_mtx 2>&1 | tee logs/lowercase_letters_run1/eval_cp_epoch<N>.txt
```

## 14. 用不同参数再训一个实验

推荐做法：不要覆盖旧实验，给每次实验一个新的 `--name`。

只临时改一个参数：

```bash
python run_training.py --config cfg_v3_lowercase_letters.yaml --name lowercase_letters_lr3e-4 --optimizer_config.lr 3.0e-4
```

临时改多个参数：

```bash
python run_training.py --config cfg_v3_lowercase_letters.yaml --name lowercase_letters_atoms32_rel10 --model_config.n_atoms 32 --loss_config.relativity 10
```

常用可调参数：

```text
--optimizer_config.lr
--batch_size
--epochs
--model_config.n_atoms
--model_config.n_channels
--model_config.d_emb_c
--model_config.d_emb_s
--loss_config.relativity
--random_seed
```

如果要系统性保留参数设置，也可以复制一份配置文件：

```bash
cp cfg_v3_lowercase_letters.yaml cfg_v3_lowercase_letters_lr3e-4.yaml
```

然后编辑新 yaml，再运行：

```bash
python run_training.py --config cfg_v3_lowercase_letters_lr3e-4.yaml --name lowercase_letters_lr3e-4
```

建议命名规则：

```text
lowercase_letters_<关键改动>_<日期或编号>
```

例子：

```text
lowercase_letters_lr3e-4_seed0
lowercase_letters_atoms32_rel10
lowercase_letters_bs8_ch512
```

## 15. W&B 说明

当前配置默认 `wandb: False`，所以训练记录只在本地。

如果要启用 W&B：

```bash
export WANDB_API_KEY="<your_wandb_key>"
python run_training.py --config cfg_v3_lowercase_letters.yaml --wandb 1 --name lowercase_letters_wandb
```

当前 W&B 只记录训练/验证 loss、learning rate 和 config；checkpoint、评估指标、confusion matrix、t-SNE 不会自动上传。

## 16. 相关文件

```text
cfg_v3_lowercase_letters.yaml
dataloader/lowercase_letters_dataloader.py
dataset/lowercase_letters/letter_office.py
dataset/lowercase_letters/split_train_val.py
model/factory.py
```
