# S10续训：累计更新次数接近Hex

## 原始200轮阶段核对

两组均正常结束，并非作业挂掉。这里使用已有完整validation CSV和checkpoint
metadata核对；本次不将这些数值标为full-test验收，正式test留待延长阶段完成。

| Arm | 原Job / 结束时间(+04) | Current epoch199 validation macro / weighted / Hungarian | PPL / active / coverage | Val recon | Retained macro-best |
|---|---|---|---|---|---|
| fragment | 196481 / 2026-09-17 05:56:34 | .265715 / .243521 / .215710 | 23.967 / 26 / 19 | .132854 | epoch72, macro .501779 |
| permanent MEAN | 196482 / 2026-09-17 05:02:32 | .478319 / .434615 / .390355 | 24.457 / 26 / 20 | .051061 | epoch19, macro .765832 |

来源：`logs/20260916-2157__VVI-RQ2-V2-C512-K26-HEXREPO[-MEAN]-S0/`。
两组当前usage尚广，但语义混杂；不能仅凭MEAN曾有.7658 macro判为稳定成功。
GPU读取完整200轮CSV：两组均无满足原标准的稳定8/10窗口。
此前Hex普通组130k steps时Hungarian也仅.221，首个稳定窗口158–167约为
397.5k–420k steps，因此追加预算值得检验，但不保证恢复。

## 决策与控制变量

- 从各自`cp_current_epoch199.pt`恢复，新增Run ID后缀`-R1`。
- 570增量epochs，绝对epoch200–769；650steps/epoch，累计500,500steps，
  比Hex500,000多0.1%。不重置optimizer、scheduler、scaler、BN或EMA。
- 两组仍分别为fragment / 永久page mean；V3仍读取原fragment-level z_s。
- seed0、旧manifest和所有原超参数不变。保留原epoch-based LR；这不是
  从零开始按step匹配学习率，也没有消除20,800 vs80,000独立训练页、字形尺度、
  每页fragment数量等差异。最终LR约.00046，Hex最终约.000818。
- 父checkpoint没有RNG状态，续训重新按seed0初始化RNG；记录为epoch-boundary
  continuation，不能承诺逐步等价于不中断训练。
- macro-best仍为validation标签诊断oracle，不改用Hungarian排名。
- 每25 completed epochs继续保留快照和重构/均衡矩阵；R1新增22快照
  (224..749)，current+macro-best+best-val最多25文件。父run快照不复制、不删除。
- 每组1GPU/32GB/4CPU/24h。原时长普通约8h、MEAN约7.1h，按比例570轮
  约22.7h/20.1h；更长绘图历史/节点速度会影响时限，不能保证24h内完成。

## 配置与可追溯性

`configs/uppercase/rq2/v2/cfg_vvi_rq2_v2_c512_k26_hexrepo_seed0_resume1.yaml`
SHA256 `0d7b3bf05f101817f9d0618f06e08d9e5fab605da065a0f0c0de216c977b3bcb`。

`configs/uppercase/rq2/v2/cfg_vvi_rq2_v2_c512_k26_hexrepo_mean_seed0_resume1.yaml`
SHA256 `4b46eb6a2fa54495c11672b93962de71dc6c25b4c94eaa03863159dfbf1348c3`。

数据manifest SHA256：
`d455680108040b20d4880aff2bba29b4de0878f97cb0d37fd10ca91b8f1dc5b6`。
原config和所有父checkpoint保持不变。作业内保存代码SHA、dirty状态、源文件
hash/源码包、提交config；`resume_lineage.json`记录源checkpoint hash和继承历史。

## 预检与后续验收

GPU预检入口：`scripts/smoke_s10_resume.py --arm repo|mean`，各执行完整650步
和2600页validation。核对model/optimizer/scheduler/scaler逐项完整恢复、
Adamstep130000→130650、首轮epoch200、macro-best/best-val原权重继承、CSV连续、
BN decoder-only不改变code，以及实际26×8组合计数全部325。
续训历史继承补入颜色、BN、online usage、optimization scales、per-style CSV，
只涉及诊断文件，不改变训练算法。Smoke不入账，验收后精确删除其目录。

预检结果、checkpoint hashes、稳定窗口统计保存于
`logs/diagnostics/20260917_s10_resume_preflight/{repo,mean}.json`。
提交信息见ACTIVE；完成后按原uppercase标准验收full-test三个角色，更新方向结论。
如果500k预算下仍未形成稳定高purity，不能解释为“更新次数已被严格排除”，
因为学习率进度、独立样本数量和renderer差异仍在。

**预检结果2026-09-17。** 在ws-l1-002完成15项回归测试，以及两个完整epoch200
smoke；均精确恢复源model/Adam/scheduler/scaler，Adamstep130000→130650，
11类已有诊断CSV完整继承，均衡矩阵实际每(letter,style)325。两个best物理权重
与父文件hash一致；每组仅3个checkpoint。两组smoke目录验收后立即删除；
不纳入正式台账，不用其权重启动正式任务。
第一次smoke自身额外绘图配置触发严格协议检查，在任何训练前安全拒绝；
仅修正smoke脚本的强制绘图方式，未放宽正式resume保护，失败临时目录也删除。

Source checkpoint SHA256：fragment
`3f82e1c0def2e5e649e7e32908e088c0161937308aeffe0cabeb0b19273ed8af`；MEAN
`bd941b33282bae21b5b859e5349ea5411421914c9a30e5cfdf0eb136f8893950`。
Git HEAD `e0bb0c6b70b5798c0dddc69c6903f9cdf315568a`，dirty；无commit/push。

**正式提交。** 两组预先登记后，于2026-09-17 13:20+04启动：

| Arm | Run directory | Slurm Job / node |
|---|---|---|
| fragment | `20260917-1319__VVI-RQ2-V2-C512-K26-HEXREPO-S0-R1` | 197821 / ws-l1-002 |
| permanent MEAN | `20260917-1319__VVI-RQ2-V2-C512-K26-HEXREPO-MEAN-S0-R1` | 197822 / ws-l1-004 |

实际GPU均为RTX5000 Ada，CUDA可见；日志确认`200-769 (570 incremental epochs)`，
恢复原scheduler并开始epoch200。两组源码包SHA256相同：
`aaf0ec9a1e2e7c2c9e6a06198c118c2cc3d5da1006db50e6454781a736cfd4b8`。
作业结束自动释放资源；不修改集群并发限制或其他项目作业。
