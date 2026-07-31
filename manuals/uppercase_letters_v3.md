# V3 Uppercase Letters 操作手册

本文档记录 uppercase letters 数据集的生成、切分、训练和评估流程。该 setting 复用 lowercase letters 的通用数据逻辑和 PhoneNums 图像 encoder/decoder。

所有命令默认在项目根目录运行：

```bash
cd /home/xuanjie.liu/Projects/variance-versus-invariance
```

## 1. 环境依赖

训练入口需要 `PyYAML`，模型需要 `vector_quantize_pytorch`：

```bash
python -m pip install PyYAML==6.0.2 vector_quantize_pytorch==1.18.0
```

如果当前 conda/env 已经装好，可以跳过。

快速检查新增代码语法：

```bash
python -m py_compile dataset/uppercase_letters/letter_office.py dataset/uppercase_letters/split_train_val.py dataloader/uppercase_letters_dataloader.py model/factory.py
```

## 2. 生成 uppercase letters 原始图片

如果之前已经用旧代码生成过数据，且图片里字母边缘被裁掉，先换一个新的输出目录，或者手动清空旧目录后重新生成。当前生成器会按 `A-Z` 的最大字符 bbox 自动选择统一字号并居中，避免 `A/W/M` 这类宽字母被切边。

```bash
python -m dataset.uppercase_letters.letter_office --save_dir ../data/uppercase_letters_ --n_pages 26000
```

作用：生成原始 PNG 数据。每张图片包含一次完整的 26 个大写字母 `A-Z`，顺序随机打乱；同一张图使用同一个颜色作为 style。

输出目录：

```text
../data/uppercase_letters_
```

文件名格式示例：

```text
CRNYFTVWQSZXGPMEIOBKHJULDA_orange.png
```

其中下划线前是大写字母序列，下划线后是 style label。

## 3. 切分 train / val / test

```bash
python -m dataset.uppercase_letters.split_train_val --data_dir ../data/uppercase_letters_ --save_dir ../data/UppercaseLetters --val_percentage 0.1 --test_percentage 0.1
```

作用：把原始 PNG 数据切成训练、验证、测试三份。

输出目录：

```text
../data/UppercaseLetters/train
../data/UppercaseLetters/val
../data/UppercaseLetters/test
```

## 4. 训练前 smoke test

```bash
python run_training.py --config cfg_v3_uppercase_letters.yaml --debug --epochs 1 --batch_size 2 --num_workers 0 --name uppercase_letters_smoke
```

作用：跑一个小训练任务，检查 dataloader、模型尺寸、loss、checkpoint 写入是否正常。

输出目录：

```text
./logs/uppercase_letters_smoke
```

## 5. 正式训练

```bash
python run_training.py --config cfg_v3_uppercase_letters.yaml --name uppercase_letters_run1
```

作用：使用 `cfg_v3_uppercase_letters.yaml` 正式训练 uppercase letters 版本的 V3。

输出目录：

```text
./logs/uppercase_letters_run1
```

checkpoint 会保存为：

```text
./logs/uppercase_letters_run1/cp_epoch<N>.pt
```

## 6. 换参数训练新实验

推荐每次使用新的 `--name`，不要覆盖旧实验：

```bash
python run_training.py --config cfg_v3_uppercase_letters.yaml --name uppercase_letters_lr3e-4 --optimizer_config.lr 3.0e-4
```

再例如改 codebook atoms 和 relativity：

```bash
python run_training.py --config cfg_v3_uppercase_letters.yaml --name uppercase_letters_atoms32_rel10 --model_config.n_atoms 32 --loss_config.relativity 10
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

## 7. 评估

```bash
python run_evaluation.py \
  --config cfg_v3_uppercase_letters.yaml \
  --active_checkpoint ./logs/uppercase_letters_run2/cp_epoch417.pt \
  --test_subset_size 1000 \
  --test_subset_seed 0 \
  --pr_metrics \
  --confusion_mtx
```

作用：评估指定 checkpoint 的 content/style disentanglement 表现，并输出 precision-recall metrics 和 confusion matrix。在 test set 里随机 sample 1000条

```bash
python run_evaluation.py \
  --config cfg_v3_uppercase_letters.yaml \
  --active_checkpoint ./logs/uppercase_letters_run2_batch32_resume2/cp_epoch472.pt \
  --confusion_mtx
```
作用：评估指定 checkpoint 的 content/style disentanglement 表现，并输出 confusion matrix。


```bash
python run_evaluation.py --config cfg_v3_uppercase_letters.yaml --active_checkpoint ./logs/uppercase_letters_run1/cp_epoch<N>.pt --vis_tsne
```

作用：生成 t-SNE 可视化。这个命令可能比较慢，建议单独运行。

如果想把评估文本也保存到本地：

```bash
python run_evaluation.py --config cfg_v3_uppercase_letters.yaml --active_checkpoint ./logs/uppercase_letters_run1/cp_epoch<N>.pt --pr_metrics --confusion_mtx 2>&1 | tee logs/uppercase_letters_run1/eval_cp_epoch<N>.txt
```

## 8. 快速检查 dataloader 输出

```bash
python - <<'PY'
from dataloader.uppercase_letters_dataloader import get_dataloader, C_LIST, S_LIST

loader = get_dataloader(
    "../data/UppercaseLetters/train",
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
C_LIST: 26 ['A', 'B', 'C'] ['X', 'Y', 'Z']
batch: (2, 26, 3, 48, 32)
content: (2, 26), label range 0..25
style: (2, 26), label range 0..7
```

## 9. 本地实验记录

训练记录默认保存在：

```text
./logs/<experiment_name>/
```

其中包括：

```text
config.yaml
log.txt
cp_epoch<N>.pt
vis/
```

实时查看训练：

```bash
tail -f logs/uppercase_letters_run1/log.txt
```

查看最后一次 validation：

```bash
grep "VALIDATION" logs/uppercase_letters_run1/log.txt | tail -1
```

对比多个 uppercase 实验的最后一次 validation：

```bash
for d in logs/uppercase_letters_*; do
  [ -f "$d/log.txt" ] || continue
  echo "$d"
  grep "VALIDATION" "$d/log.txt" | tail -1
done
```

## 10. 相关文件

```text
cfg_v3_uppercase_letters.yaml
dataloader/uppercase_letters_dataloader.py
dataset/uppercase_letters/letter_office.py
dataset/uppercase_letters/split_train_val.py
model/factory.py
```
