#!/bin/bash
# 仅评估已有 checkpoint 的 task（时变 QGF: qgf.py, (1-t)/t·∇Q, jac=True）
# 用法:
#   全部:  bash scripts/eval_existing.sh ENV=scene-play SEED=0 GPU=5
#   单个:  bash scripts/eval_existing.sh ENV=scene-play SEED=0 GPU=5 TASK=2
set -euo pipefail

ENV=""; SEED=0; GPU=0; TASK=""

for arg in "$@"; do
  case $arg in
    ENV=*)  ENV="${arg#*=}" ;;
    SEED=*) SEED="${arg#*=}" ;;
    GPU=*)  GPU="${arg#*=}" ;;
    TASK=*) TASK="${arg#*=}" ;;
    *) echo "Unknown arg: $arg" && exit 1 ;;
  esac
done

if [ -z "$ENV" ] || [ -z "$GPU" ]; then
  echo "用法: bash scripts/eval_existing.sh ENV=<env> GPU=<gpu> [SEED=<n>] [TASK=<n>]"
  echo ""
  echo "必需:  ENV  环境名  GPU  GPU ID"
  echo "可选:  SEED 随机种子 (默认 0)"
  echo "       TASK 单个 task (默认全跑 1-5)"
  echo ""
  echo "示例:"
  echo "  bash scripts/eval_existing.sh ENV=cube-quadruple-play SEED=1 GPU=5"
  echo "  bash scripts/eval_existing.sh ENV=scene-play GPU=5 TASK=2"
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

# ── 评估（时变 QGF: qgf.py, (1-t)/t·∇Q, jac=True）──
for t in "${TASKS[@]}"; do
  TRAIN_DIR=exp/qgf/bc_iql/${ENV}/seed${SEED}/task${t}
  CKPT=$(find "$TRAIN_DIR" -name "params_500000.pkl" 2>/dev/null | head -1)

  if [ -z "$CKPT" ]; then
    echo "Skip $PREFIX task$t: no checkpoint at $TRAIN_DIR"
    continue
  fi

  TRAIN_SUBDIR=$(dirname "$CKPT")
  EVAL_DIR=exp/qgf/eval_timetune_jac/${ENV}/seed${SEED}/task${t}
  mkdir -p $EVAL_DIR

  echo ">>> checkpoint: $TRAIN_SUBDIR"
  echo ">>> [时变 QGF] Eval: $PREFIX task$t seed$SEED (qgf.py, (1-t)/t·∇Q, jac=True)"
  CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl python main.py \
    --seed=$SEED \
    --env_name=${PREFIX}-play-singletask-task${t}-v0 \
    --agent=agents/qgf.py \
    --agent.action_chunking=True --agent.horizon_length=5 \
    --agent.denoised_action_approx=one_euler_step_approx \
    --agent.apply_jacobian=True \
    --agent.actor_hidden_dims="(1024,1024,1024,1024)" \
    --agent.value_network_kwargs.hidden_dims="(1024,1024,1024,1024)" \
    --agent.discount=0.999 \
    --ogbench_dataset_dir=$OGBENCH_DATA_DIR/${PREFIX}-play-100m-v0/ \
    --restore_path=$TRAIN_SUBDIR \
    --restore_epoch=500000 \
    --eval_only=True --eval_episodes=30 \
    --guidance_weights=0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.12 \
    --save_dir=$EVAL_DIR \
    --wandb_run_group=eval_timetune_jac

  echo "<<< Eval done: $EVAL_DIR"
done
