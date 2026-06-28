#!/bin/bash
# 仅评估已有 checkpoint 的 task（时变 QGF: qgf.py, (1-t)/t·∇Q, jac=False）
# 目的: 消融实验 — 分离时变权重 (1-t)/t 与 Jacobian 修正各自的贡献
#
# 用法:
#   bash scripts/eval_existing_nojac.sh ENV=scene-play SEED=2 GPU=5
#   bash scripts/eval_existing_nojac.sh ENV=scene-play SEED=2 GPU=5 TASK=2
#
# SEED 必须指定（一次只跑一个 seed 的所有 task）
# TASK 可选：留空 = 全跑 1-5，指定 n 或 n-m 或 n,m,o
#
# 错误处理: 单个 task 失败不中断，记录到 scripts/failed_runs.log
# 日志: 同时输出到终端和 exp/qgf/eval_timetune_without_jac/<env>/seed<N>/eval_nojac.log
set -uo pipefail

ENV=""; SEED=""; GPU=""; TASK=""

for arg in "$@"; do
  case $arg in
    ENV=*)  ENV="${arg#*=}" ;;
    SEED=*) SEED="${arg#*=}" ;;
    GPU=*)  GPU="${arg#*=}" ;;
    TASK=*) TASK="${arg#*=}" ;;
    *) echo "Unknown arg: $arg" && exit 1 ;;
  esac
done

if [ -z "$ENV" ] || [ -z "$GPU" ] || [ -z "$SEED" ]; then
  echo "用法: bash scripts/eval_existing_nojac.sh ENV=<env> SEED=<seed> GPU=<gpu> [TASK=<n>]"
  echo ""
  echo "必需:  ENV   环境名  (scene-play | cube-triple-play | cube-quadruple-play | puzzle-4x4-play)"
  echo "       SEED  随机种子 (必须显式指定)"
  echo "       GPU   GPU ID"
  echo "可选:  TASK  单个 task (默认全跑 1-5)"
  echo ""
  echo "示例:"
  echo "  bash scripts/eval_existing_nojac.sh ENV=cube-quadruple-play SEED=1 GPU=5"
  echo "  bash scripts/eval_existing_nojac.sh ENV=scene-play SEED=0 GPU=5 TASK=2"
  exit 1
fi

case $ENV in
  scene-play)          PREFIX="scene" ;;
  cube-triple-play)    PREFIX="cube-triple" ;;
  cube-quadruple-play) PREFIX="cube-quadruple" ;;
  puzzle-4x4-play)     PREFIX="puzzle-4x4" ;;
  *) echo "未知环境: $ENV" && exit 1 ;;
esac

if [ -n "$TASK" ]; then
  TASKS=()
  IFS=',' read -ra PARTS <<< "$TASK"
  for part in "${PARTS[@]}"; do
    if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      for ((i=${BASH_REMATCH[1]}; i<=${BASH_REMATCH[2]}; i++)); do
        TASKS+=($i)
      done
    else
      TASKS+=($part)
    fi
  done
else
  TASKS=(1 2 3 4 5)
fi

# 日志目录 & 文件
LOG_DIR=exp/qgf/eval_timetune_without_jac/${ENV}/seed${SEED}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_nojac.log"

# 记录启动信息
{
  echo "========================================"
  echo "eval_existing_nojac.sh start: $(date)"
  echo "ENV=$ENV  PREFIX=$PREFIX  SEED=$SEED  GPU=$GPU  TASKS=${TASKS[*]}"
  echo "LOG: $LOG_FILE"
  echo "========================================"
} | tee -a "$LOG_FILE"

# ── 逐 task 评估 ──
for t in "${TASKS[@]}"; do
  TRAIN_DIR=exp/qgf/bc_iql/${ENV}/seed${SEED}/task${t}
  CKPT=$(find "$TRAIN_DIR" -name "params_500000.pkl" 2>/dev/null | head -1)

  if [ -z "$CKPT" ]; then
    echo "[SKIP] $PREFIX task$t: no checkpoint at $TRAIN_DIR" | tee -a "$LOG_FILE"
    continue
  fi

  TRAIN_SUBDIR=$(dirname "$CKPT")
  EVAL_DIR=exp/qgf/eval_timetune_without_jac/${ENV}/seed${SEED}/task${t}
  mkdir -p "$EVAL_DIR"

  {
    echo ""
    echo ">>> checkpoint: $TRAIN_SUBDIR"
    echo ">>> [时变 nojac] Eval: $PREFIX task$t seed$SEED (qgf.py, (1-t)/t·∇Q, jac=False)"
    echo ">>> start: $(date)"
  } | tee -a "$LOG_FILE"

  if CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl python main.py \
    --seed=$SEED \
    --env_name=${PREFIX}-play-singletask-task${t}-v0 \
    --agent=agents/qgf.py \
    --agent.action_chunking=True --agent.horizon_length=5 \
    --agent.denoised_action_approx=one_euler_step_approx \
    --agent.apply_jacobian=False \
    --agent.actor_hidden_dims="(1024,1024,1024,1024)" \
    --agent.value_network_kwargs.hidden_dims="(1024,1024,1024,1024)" \
    --agent.discount=0.999 \
    --ogbench_dataset_dir=$OGBENCH_DATA_DIR/${PREFIX}-play-100m-v0/ \
    --restore_path=$TRAIN_SUBDIR \
    --restore_epoch=500000 \
    --eval_only=True --eval_episodes=30 \
    --guidance_weights=0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.12 \
    --save_dir=$EVAL_DIR \
    --wandb_run_group=eval_timetune_without_jac 2>&1 | tee -a "$LOG_FILE"; then
    echo "<<< Eval done: $EVAL_DIR  ($(date))" | tee -a "$LOG_FILE"
  else
    {
      echo "!!! Eval FAILED: $PREFIX task$t seed$SEED  ($(date))"
      echo "!!! Eval FAILED: $PREFIX task$t seed$SEED" >> scripts/failed_runs.log
    } | tee -a "$LOG_FILE"
  fi
done

{
  echo ""
  echo "========================================"
  echo "All done! ($(date))"
  echo "  Eval (时变 nojac): exp/qgf/eval_timetune_without_jac/${ENV}/seed${SEED}/"
  echo "  Log: $LOG_FILE"
  echo "========================================"
} | tee -a "$LOG_FILE"
