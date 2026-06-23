# QGF Project Quick Reference

## 项目概述
Q-Guided Flow (QGF): 离线 RL 方法，BC Flow Matching policy + IQL critic/value + 推理时 ∇Q 引导去噪。
- 环境: OGBench (4 个 MuJoCo 仿真场景 × 5 task)
- 4 环境: scene-play, cube-triple-play, cube-quadruple-play, puzzle-4x4-play
- 论文: Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning

## 环境启动
```bash
cd ~/QGF && source .venv/bin/activate
export OGBENCH_DATA_DIR=/mnt/data/yunyang/datasets/ogbench_data
```

## 核心代码文件
| 文件 | 功能 |
|------|------|
| `main.py` | 统一入口，训练+评估 |
| `agents/qgf_backup.py` | 原版 QGF (w·∇Q, jac=F) |
| `agents/qgf.py` | 时变 QGF (w·(1-t)/t·∇Q, jac=T) |
| `utils/networks.py` | ActorFlowField (policy), Value (Q/V) |
| `utils/evaluation.py` | eval_with_test_time_guidance, run_episodes |
| `envs/ogbench_utils.py` | 100M 数据集加载 + relabel_dataset |

## 批量脚本
```bash
bash scripts/run_envV1.sh ENV=<env> SEED=<n> GPU=<n> TASK=<n>
# ENV: scene-play | cube-triple-play | cube-quadruple-play | puzzle-4x4-play
# TASK: 1-5 | 1,3,5 | 1-3 | 留空=1-5
# 失败不中断，记录到 scripts/failed_runs.log

bash scripts/eval_existing.sh ENV=<env> GPU=<n> TASK=<n>
# 仅评估已有 checkpoint
```

## 实验结果目录
- 原版训练: `exp/qgf/bc_iql/{env}/seed{N}/task{n}/bc_iql_qgf_.../`
- 时变评估: `exp/qgf/eval_timetune_jac/{env}/seed{N}/task{n}/eval_timetune_jac_qgf_.../`
- 文档: `hyg_QA/`（详见 00_INDEX.md）

## 核心结果 (55 对配对, 4 环境)
原版 64.5% → 时变 69.2% (+4.7%)
详见 hyg_QA/10_full_results.md

## 已知问题
- cube-quadruple task3: MuJoCo 物理崩溃（mj_maxContact 超限），eval_vecenv_size=1 无法解决，换 seed 无效
- puzzle-4x4 seed3 task3: 同样可能 MuJoCo 崩溃

## 数据类型
OGBench singletask: 5 个 task = 5 种不同目标配置，每 task 需单独训练 Q 函数
100M play 数据集: 100 个 .npz slice，每 1000 步轮换
40 维 obs (19 proprio + 9 方块 + 8 按钮 + 4 抽屉窗户)，5 维 action
