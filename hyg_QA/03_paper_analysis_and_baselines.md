# 论文分析：定位、不足与完整实验脚本

---

## 1. QGF 不是 VLA 论文

QGF 是一篇**离线 RL 方法论论文**，不是 VLA（Vision-Language-Action）论文。它既不处理图像，也不处理语言。实验全部在 MuJoCo 仿真环境中用状态向量完成，没有真实机器人、没有视觉、没有指令理解。

这一点论文自身没有隐瞒——标题是 "Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning"——关键词是 RL、Flow Policies、Guidance。

---

## 2. 论文的不足 / 逻辑漏洞

### 2.1 根本局限：Sim-only & State-only

| 维度 | QGF | VLA（如 π0.5, RT-2） |
|------|-----|---------------------|
| 输入 | MuJoCo 仿真状态向量（~60 维 float） | RGB 图像 + 自然语言 |
| 泛化 | 单环境单 task | 跨环境、跨物体、跨语言 |
| 部署 | 仿真 | 真实机器人 |
| Q 函数 | 需要仿真器精确计算 reward | 需要 VLM / 人类打分 |
| 规模 | 16M 参数 | >1B 参数 |

**问题：论文在"做 RL 的人在意的指标"上证明了自己，但离实际机器人应用差得非常远。**

### 2.2 Q 函数需要显式 Reward

OGBench 的 reward 来自 `relabel_dataset`——在仿真器里精确度量每个物体离目标的物理距离。现实中：
- 没有精确的物体位姿
- 没有预定义的目标坐标
- "把红色方块推到蓝色标记处"这种语义任务无法用距离公式表达

论文说"Q 函数可替换为 VLM 打分"，但这只是设想，**没有实验**。从仿真距离 → VLM 语义打分的迁移难度是未知的。

### 2.3 Guidance Weight 需要手工调参

评估时要扫描 9 个 `guidance_weights` 取最大值。**没有自动选择 guidance weight 的机制。** 在真实场景中，你没法提前知道哪个 weight 最好——这削弱了 "test-time"（测试时即插即用）的卖点。

### 2.4 训练数据覆盖问题

100M play 数据集虽然大，但仍是**单一策略（脚本策略）**采集的。离线 RL 的经典问题是：如果数据集里没有"好 action"，Q 函数学不到好的策略。QGF 的 BC policy 受限于数据质量，Q 引导只能在数据分布周围微调。

### 2.5 实验规模 ≠ 泛化能力

论文在 4 个环境上跑，但这 4 个环境结构相似（都是 MuJoCo 桌面操作 + 拼图）。没有跨 embodiment（如从夹爪换到灵巧手）、没有跨场景（如从桌面换到厨房）、没有 sim2real。

### 2.6 方法组件拆解不完整

论文没有充分回答：
- Flow Matching vs DDPM：差多少？为什么选 FM？
- IQL vs 其他 offline RL 基座：TD3+BC、CQL 能用吗？
- Guidance 对 critic 质量有多敏感？critic 很差时引导是否反效果？

---

## 3. README 中的完整实验脚本

README §"Launching full paper experiments (via SLURM)" 给出了所有方法的脚本。分两类：

### 3.1 Train-time Baselines（各自训练 actor/critic）

```bash
python scripts/exp_cfgrl.py  && bash sbatch/cfgrl.sh     # Contrastive Flow
python scripts/exp_fql.py    && bash sbatch/fql.sh       # Flow Q-Learning
python scripts/exp_edp.py    && bash sbatch/edp.sh       # Energy Diffusion Policy
python scripts/exp_qam.py    && bash sbatch/qam.sh       # Q-aware Diffusion
python scripts/exp_dac.py    && bash sbatch/dac.sh       # Diffusion Actor-Critic
python scripts/exp_qsm_bc.py && bash sbatch/qsm_bc.sh    # QSM + BC
```

### 3.2 Test-time Methods（共享 BC+IQL 基座 + 推理时引导）

```bash
# Step 1: 训练共享基座（4 env × 5 task × 10 seed）
python scripts/bc_iql_train.py && bash sbatch/bc_iql.sh

# Step 2: 测试时方法评估（用同一个基座）
TRAIN_RUN_GROUP=bc_iql python scripts/exp_qgf_test_time_eval.py          && bash sbatch/qgf_test_time_eval.sh          # QGF
TRAIN_RUN_GROUP=bc_iql python scripts/exp_qgf_jacobian_test_time_eval.py && bash sbatch/qgf_jacobian_test_time_eval.sh # QGF-Jacobian
TRAIN_RUN_GROUP=bc_iql python scripts/exp_qfql_test_time_eval.py         && bash sbatch/qfql_test_time_eval.sh         # QFQL
TRAIN_RUN_GROUP=bc_iql python scripts/exp_robust_q.py                    && bash sbatch/robust_q.sh                    # Robust Q
TRAIN_RUN_GROUP=bc_iql python scripts/exp_grad_step_test_time_eval.py    && bash sbatch/grad_step.sh                   # Grad Step
```

### 3.3 合并训练+评估

```bash
python scripts/qgf_train_test.py && bash sbatch/qgf_train_test.sh
```

### 3.4 脚本工作原理

每个 `scripts/exp_*.py`：
1. 遍历 `(env, task, seed)` 组合
2. 为每个组合生成一条 `python main.py ...` 命令
3. 打包成 SLURM array job → `sbatch/<name>.sh`
4. 使用 GNU parallel 每 GPU 跑多个 job

**单机运行方式（不依赖 SLURM）：**

各脚本在 `--debug` 模式下只打印命令不生成 SLURM：
```bash
python scripts/exp_qgf_test_time_eval.py --debug
```
输出所有命令后，可以提取自己在本地跑。

### 3.5 完整实验规模

| 类型 | 方法 | 命令数 |
|------|------|--------|
| Train-time | CFGRL, FQL, EDP, QAM, DAC, QSM+BC | 6 × 4 env × 5 task × 10 seed = 1200 |
| Base model | BC+IQL | 4 env × 5 task × 10 seed = 200 |
| Test-time | QGF, QGF-Jacobian, QFQL, Robust Q, Grad Step | 5 × 200 = 1000 |
| **总计** | **12 种方法** | **~2400 次训练/评估** |

---

## 4. 论文贡献的真正边界

```
论文的 contribution 范围：
  ✅ 提出"在去噪过程中用 ∇Q 引导"这个机制
  ✅ 在仿真 state-based 环境上验证有效（4 envs, avg 提升 10-70%）
  ✅ 证明 Euler-step 干净估计 + stop_grad 的降低方差效果
  ✅ 证明 BC + IQL + Guidance 优于联合训练 actor-critic
  ✅ 提供完整代码库和 12 个 baseline 的实现

论文没有做到的：
  ❌ 处理视觉输入
  ❌ 处理语言指令
  ❌ 真实机器人实验
  ❌ 自动 guidance weight 选择
  ❌ 跨环境/跨任务零样本泛化
  ❌ VLM-as-critic 的实验验证
  ❌ 和 VLA 方法的直接对比
```

QGF 的正确理解：**它在离线 RL 的小世界里提出一个有效的方法**，不是要和 π0.5 竞争真实机器人任务。它解决的是"如何在不联合训练 actor-critic 的情况下，用 Q 的梯度改善 BC policy"——这是一个 RL 方法论问题，不是大模型问题。
