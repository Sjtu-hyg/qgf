# QGF 实验完整指南与结果

---

## 1. 前提准备

```bash
cd ~/QGF && source .venv/bin/activate
export OGBENCH_DATA_DIR=/mnt/data/yunyang/datasets/ogbench_data
```

数据集目录结构：
```
$OGBENCH_DATA_DIR/
├── cube-quadruple-play-100m-v0/
├── cube-triple-play-100m-v0/
├── puzzle-4x4-play-100m-v0/
└── scene-play-100m-v0/
```

额外依赖：
```bash
uv pip install moviepy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 2. 训练命令（4 环境 × 每 task 单独训）

**关键：每 task 必须单独训练，因 `relabel_dataset` 按 task 重标 reward → Q 函数是 task-specific 的。**

### scene-play (task1-5)

```bash
# 以 task2 为例，其余 task 改 --env_name 即可
CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl python main.py \
  --env_name=scene-play-singletask-task2-v0 \
  --agent=agents/qgf.py \
  --agent.denoise_steps=10 --agent.expectile=0.9 \
  --agent.action_chunking=True --agent.horizon_length=5 \
  --agent.denoised_action_approx=one_euler_step_approx \
  --agent.apply_jacobian=False \
  --agent.batch_size=1024 \
  --agent.actor_hidden_dims="(1024,1024,1024,1024)" \
  --agent.value_network_kwargs.hidden_dims="(1024,1024,1024,1024)" \
  --agent.discount=0.999 \
  --ogbench_dataset_dir=$OGBENCH_DATA_DIR/scene-play-100m-v0/ \
  --dataset_replace_interval=1000 \
  --offline_steps=500000 --eval_interval=100000 --save_interval=100000 \
  --eval_episodes=30 --video_episodes=3 \
  --guidance_weights=0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.12 \
  --wandb_run_group=bc_iql
```

### cube-triple (task1-5, 默认 task2)

```bash
CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl python main.py \
  --env_name=cube-triple-play-singletask-task2-v0 \
  ... (其余参数同上) \
  --ogbench_dataset_dir=$OGBENCH_DATA_DIR/cube-triple-play-100m-v0/
```

### cube-quadruple (task1-5, 默认 task2)

> `relabel_dataset` 在 quad 上有 bug，用 `--sparse=True` 绕过

```bash
CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl python main.py \
  --env_name=cube-quadruple-play-singletask-task2-v0 \
  ... --sparse=True \
  --ogbench_dataset_dir=$OGBENCH_DATA_DIR/cube-quadruple-play-100m-v0/
```


```bash
CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl python main.py \
  --agent=agents/qgf.py \
  --agent.denoised_action_approx=one_euler_step_approx \
  --agent.apply_jacobian=False \
  --agent.action_chunking=True \
  --agent.horizon_length=5 \
  --env_name=cube-quadruple-play-singletask-task2-v0 \
  --ogbench_dataset_dir=$OGBENCH_DATA_DIR/cube-quadruple-play-100m-v0/ \
  --offline_steps=500000 \
  --guidance_weights=0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.12
```

time-varying 调度

雅各比=True

```
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python main.py \
  --env_name=cube-quadruple-play-singletask-task2-v0 \
  --agent=agents/qgf.py \
  --agent.denoised_action_approx=one_euler_step_approx \
  --agent.apply_jacobian=True \
  --agent.action_chunking=True \
  --agent.horizon_length=5 \
  --ogbench_dataset_dir=$OGBENCH_DATA_DIR/cube-quadruple-play-100m-v0/ \
  --restore_path=exp/qgf/Debug/Debug_qgf_cube-quadruple-play-task2_seed00_e01ac582 \
  --restore_epoch=500000 \
  --eval_only=True \
  --guidance_weights=1.0 \
  --wandb_run_group=eval_timetune
```

结果：

雅各比=False

结果：

### puzzle-4x4 (task1-5, 默认 task4)

```bash
CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl python main.py \
  --env_name=puzzle-4x4-play-singletask-task4-v0 \
  ... \
  --ogbench_dataset_dir=$OGBENCH_DATA_DIR/puzzle-4x4-play-100m-v0/
```

---

## 3. 引导策略与代码版本

### 3.1 代码版本

| 文件 | 引导公式 | 说明 |
|------|---------|------|
| `agents/qgf_backup.py` | `v_bc + w · ∇Q` | **原版**：固定系数 |
| `agents/qgf.py` | `v_bc + w · (1-t)/t · ∇Q` (t>0) | **时变**：早期强引导，后期弱引导 |

恢复原版：`cp agents/qgf_backup.py agents/qgf.py`

### 3.2 Jacobian 开关

| `apply_jacobian` | ∇Q 计算 | 说明 |
|------------------|---------|------|
| `False` | `∇Q(a_approx)` 直接当作 `∇Q(a_t)` | **默认**，低方差 |
| `True` | `∇Q(a_approx) · ∂a_approx/∂a_t` | 链式法则，高方差 |

### 3.3 评估文件夹命名

| 代码 | Jacobian | wandb_run_group | 结果目录 |
|------|----------|----------------|---------|
| qgf.py (时变) | False | `eval_timetune` | `exp/qgf/eval_timetune/..._4e32d5d5` |
| qgf.py (时变) | True | `eval_timetune_jac` | `exp/qgf/eval_timetune_jac/..._<new_hash>` |
| qgf_backup.py (原版) | False | `bc_iql` | `exp/qgf/bc_iql/..._c24aba51` |
| qgf_backup.py (原版) | True | `bc_iql_jac` | `exp/qgf/bc_iql_jac/..._<new_hash>` |

---

## 4. 批量训练+评估脚本

### 4.1 目录结构

```
exp/qgf/
├── bc_iql/                           # 训练结果（原版 qgf_backup.py）
│   └── {env_name}/                   # e.g. cube-quadruple-play
│       └── seed{N}/                  # seed 0-9
│           ├── task1/                # params_500000.pkl, eval.csv, train.csv
│           ├── task2/
│           └── ...
├── eval_timetune_jac/                # 评估结果（时变 qgf.py, jac=True）
│   └── {env_name}/
│       └── seed{N}/
│           ├── task1/                # eval.csv, flags.json
│           ├── task2/
│           └── ...
```

### 4.2 用法

```bash
# ── 完整训练+评估 ──

# cube-quadruple seed=1, 全 5 task, GPU 0 (终端1)
bash scripts/run_env.sh ENV=cube-quadruple-play SEED=1 GPU=0 TASK=1-5

# puzzle-4x4 seed=1, 全 5 task, GPU 5 (终端2，可并行)
bash scripts/run_env.sh ENV=puzzle-4x4-play SEED=1 GPU=0 TASK=1-5

# scene task2 单独
bash scripts/run_env.sh ENV=scene-play SEED=0 GPU=5 TASK=2

# task 1-3 范围
bash scripts/run_env.sh ENV=cube-triple-play SEED=1 GPU=5 TASK=1-5

# task 1,3,5 逗号分隔
bash scripts/run_env.sh ENV=puzzle-4x4-play GPU=5 TASK=1-5

# ── 仅评估（不重训）──
bash scripts/eval_existing.sh ENV=cube-quadruple-play SEED=1 GPU=5
bash scripts/eval_existing.sh ENV=scene-play GPU=5 TASK=2
```

### 4.3 脚本位置

- `scripts/run_env.sh` — 训练（`qgf_backup.py`）+ 评估（`qgf.py` jac=T）
- `scripts/eval_existing.sh` — 仅评估已有 checkpoint 的 task
- 参数: `ENV=`, `SEED=`, `GPU=`, `TASK=`（TASK 可选，支持 `1-3`, `1,3,5`, `1-2,5`）

---

## 5. 评估命令

### 单 task 评估（eval_only）

```bash
CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl python main.py \
  --env_name=<ENV_NAME> \
  --agent=agents/qgf.py \
  --agent.denoise_steps=10 --agent.expectile=0.9 \
  --agent.action_chunking=True --agent.horizon_length=5 \
  --agent.denoised_action_approx=one_euler_step_approx \
  --agent.apply_jacobian=False \
  --agent.actor_hidden_dims="(1024,1024,1024,1024)" \
  --agent.value_network_kwargs.hidden_dims="(1024,1024,1024,1024)" \
  --agent.discount=0.999 \
  --ogbench_dataset_dir=$OGBENCH_DATA_DIR/<DATASET_DIR>/ \
  --restore_path=<CHECKPOINT_PATH> --restore_epoch=500000 \
  --eval_only=True --eval_episodes=100 \
  --guidance_weights=0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.12
```

### 批量评估（scene 5 task 链式）

```bash
CKPT=exp/qgf/bc_iql/bc_iql_qgf_scene-play-task2_seed00_c24aba51
DATA=$OGBENCH_DATA_DIR/scene-play-100m-v0
GPU=4
AA="--agent=agents/qgf.py --agent.denoise_steps=10 --agent.expectile=0.9 --agent.action_chunking=True --agent.horizon_length=5 --agent.denoised_action_approx=one_euler_step_approx --agent.apply_jacobian=False --agent.actor_hidden_dims='(1024,1024,1024,1024)' --agent.value_network_kwargs.hidden_dims='(1024,1024,1024,1024)' --agent.discount=0.999"
for t in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl python main.py $AA \
    --env_name=scene-play-singletask-task${t}-v0 \
    --ogbench_dataset_dir=$DATA \
    --restore_path=$CKPT --restore_epoch=500000 \
    --eval_only=True --eval_episodes=100 \
    --guidance_weights=0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.12
done
```

---

## 6. 实验结果

> 配置: batch_size=1024, hidden=4×1024, discount=0.999, 30 episodes, 500K steps  
> 原版 = `qgf_backup.py` (w·∇Q, jac=F) | 时变 = `qgf.py` (w·(1-t)/t·∇Q, jac=T)

### 6.1 Scene-play

| Task | Seed | 原版 | 时变 |
|------|------|------|------|
| task1 | seed=0 | 100% (w=0.10) | — |
| task2 | seed=0 | 100% (w=0.08) | 100% (w=0.10) |

**scene task2 详细**:

| W | 原版 | 时变 |
|---|------|------|
| 0.004 | 83% | 43% |
| 0.008 | 100% | 63% |
| 0.01 | 100% | 67% |
| 0.02 | 100% | 97% |
| 0.04 | 100% | 100% |
| 0.06 | 97% | 100% |
| 0.08 | 100% | 100% |
| 0.10 | 100% | 100% |
| 0.12 | 100% | 97% |

> scene 简单——两种方法均 100%，差异仅在 w=0.004（原版 83% > 时变 43%，低 w 时变干扰 BC）

### 6.2 Cube-triple-play

| Task | Seed | 原版 | 时变 |
|------|------|------|------|
| task2 | seed=0 | 83% (w=0.06) | 83% (w=0.12) |

**cube-triple task2 详细**:

| W | 原版 | 时变 |
|---|------|------|
| 0.004 | 7% | 0% |
| 0.008 | 30% | 0% |
| 0.01 | 37% | 0% |
| 0.02 | 67% | 0% |
| 0.04 | 73% | 53% |
| 0.06 | **83%** | 73% |
| 0.08 | 67% | 77% |
| 0.10 | 37% | 70% |
| 0.12 | 30% | **83%** |

> 持平（83%），时变最优 w 右移至 0.12；原版 w>0.06 衰退，时变 w>0.06 维持

### 6.3 Cube-quadruple-play

| Task | Seed | 原版 | 时变 | Δ |
|------|------|------|------|----|
| task1 | seed=1 | 33% (w=0.02) | **50%** (w=0.12) | +17% |
| task2 | seed=1 | 10% (w=0.01) | 13% (w=0.12) | +3% |
| task2 | seed=0 | 3% (w=0.02) | **17%** (w=0.08) | +14% |
| task3-5 | seed=1 | — | — | MuJoCo 崩溃 |

> task3-5 seed=1 因 MuJoCo 物理碰撞超限崩溃，需换 seed 重试。

**quad task1 seed1 详细**（最佳改善 +17%）:

| W | 原版 | 时变 |
|---|------|------|
| 0.004 | 0% | 0% |
| 0.008 | 3% | 0% |
| 0.01 | 13% | 0% |
| 0.02 | **33%** | 0% |
| 0.04 | 27% | 3% |
| 0.06 | 13% | 10% |
| 0.08 | 7% | 20% |
| 0.10 | 0% | 30% |
| 0.12 | 0% | **50%** |

> 时变在高 w (0.10-0.12) 完胜原版；quad Q 较弱，早期强引导更关键

### 6.4 Puzzle-4x4-play (seed=0, 5 task)

| Task | 原版 | 时变 | Δ |
|------|------|------|----|
| task1 | 63% (w=0.01) | **67%** (w=0.10) | +4% |
| task2 | 53% (w=0.01) | **57%** (w=0.04) | +4% |
| task3 | 77% (w=0.008) | **87%** (w=0.06) | +10% |
| task4 | 33% (w=0.008) | **40%** (w=0.04) | +7% |
| task5 | 43% (w=0.01) | **53%** (w=0.04) | +10% |
| **平均** | **54%** | **61%** | **+7%** |

> **5 task 全胜，无一倒退。** 时变最优 w 从 0.01 右移至 0.04-0.10。

**puzzle task3 详细**:

| W | 原版 | 时变 |
|---|------|------|
| 0.004 | 0% | 0% |
| 0.008 | 27% | 27% |
| 0.01 | 30% | 30% |
| 0.02 | 57% | 57% |
| 0.04 | 83% | 83% |
| 0.06 | 73% | **87%** |
| 0.08 | 77% | 73% |
| 0.10 | 63% | 77% |
| 0.12 | 53% | 77% |

### 6.5 总计

| 环境 | Tasks | 原版 avg | 时变 avg | Δ | 趋势 |
|------|-------|---------|---------|----|------|
| scene | 1-2 | 100% | 100% | 持平 | 任务简单 |
| cube-triple | 2 | 83% | 83% | 持平 | 最优 w 右移 |
| cube-quadruple | 1-2 | 15% | 27% | +12% | 低 Q 时受益 |
| puzzle-4x4 | 1-5 | 54% | **61%** | +7% | **** |

### 6.6 关键发现

1. **(1-t)/t + jac=T 最优 w 普遍右移**（≈0.01→0.04-0.12），Jacobian 修正抵消了早期步的 (1-t)/t 放大
2. **高 w 区间时变明显更强**：原版 w>0.06 常崩，时变 w=0.08-0.12 仍有效
3. **低 Q 环境（quad）受益最大**（+12%），弱 Q 需要早期强引导
4. **puzzle 5 task 全胜**（+7%），离散任务对 Jacobian 修正意外的适应良好
5. **低 w 区间退化**：w=0.004 时 (1-t)/t 仍放大 9×，干扰 BC 基线

## 7. Singletask 多任务机制

### 5 个 Task = 5 个不同目标

OGBench 每个环境提供 5 个 singletask（OGBenchREADME:126-149），对应同一场景的 5 种不同目标配置（物体放不同位置、按钮不同状态组合）。

### 训练数据相同，Reward 不同

100M play 数据集**同一份 s,a**，但 `relabel_dataset()` 按 `--env_name` 指定的 task 目标重标 reward：

```
--env_name=scene-...-task2 → relabel 用 task2 目标计算 reward → Q_{task2}
--env_name=scene-...-task1 → relabel 用 task1 目标计算 reward → Q_{task1}
```

BC policy 不受影响（只用 s,a），**Q 函数完全按 task 绑定**。

### 为什么跨 task 评估必崩

用 task2 的 Q 评估 task1 → ∇Q 指向 task2 目标方向 → agent 走向错误位置 → 0% success。**Q 和 task 必须配对。**

### 论文做法

论文 4 env × 5 task = 20 条独立训练命令（`bc_iql_train.py:18-24`），每 task 独立 checkpoint，评估时各自配对。

---

## 8. Wandb 使用

### 登录（一次性）
```bash
wandb login
```

### 看训练指标
打开 run 链接 → 往下滚过 SYSTEM 面板 → 训练指标在下面（training/bc_loss 等）

### 关键指标
- `evaluation/success` — 金标准
- `training/bc_loss` — 应持续下降
- `training/critic_loss` — 震荡正常

---

## 9. Debug & 排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `No module named 'ogbench'` | 环境没装好 | `uv sync` |
| `FileNotFoundError: *.npz` | 路径不对 | `echo $OGBENCH_DATA_DIR` 确认 |
| `flax.errors.ScopeParamShapeError` | eval 时网络大小不一致 | 加上 `--agent.actor_hidden_dims` 等参数 |
| `wandb.Video requires moviepy` | 缺依赖 | `uv pip install moviepy` |
| `relabel_dataset np.stack shape error` | ogbench quad bug | 加 `--sparse=True` |

---

## 10. Debug 命令（5K 步快速验证）

```bash
CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl python main.py \
  --env_name=scene-play-singletask-task2-v0 \
  --agent=agents/qgf.py \
  --agent.action_chunking=True --agent.horizon_length=5 \
  --agent.denoised_action_approx=one_euler_step_approx \
  --agent.apply_jacobian=False \
  --ogbench_dataset_dir=$OGBENCH_DATA_DIR/scene-play-100m-v0/ \
  --dataset_replace_interval=1000 \
  --offline_steps=5000 --eval_interval=2500 --save_interval=2500 \
  --eval_episodes=5 --guidance_weights="0.0,1.0,5.0" \
  --wandb_run_group=debug --debug=True
```

---

## 11. 环境速查

| 环境 | `--env_name` 示例 | 数据集 | 默认 Task |
|------|-----------------|--------|-----------|
| scene | `scene-play-singletask-task2-v0` | `.../scene-play-100m-v0/` | task2 |
| cube-triple | `cube-triple-play-singletask-task2-v0` | `.../cube-triple-play-100m-v0/` | task2 |
| cube-quadruple | `cube-quadruple-play-singletask-task2-v0` | `.../cube-quadruple-play-100m-v0/` | task2 |
| puzzle-4x4 | `puzzle-4x4-play-singletask-task4-v0` | `.../puzzle-4x4-play-100m-v0/` | task4 |
