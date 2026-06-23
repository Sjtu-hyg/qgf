#!/bin/bash
# 训练（qgf_backup.py 原版）+ 评估（qgf.py 时变+jac=T）
# 用法:
#   全部 5 task:  bash scripts/run_env.sh ENV=scene-play         SEED=1 GPU=5
#   单个 task:    bash scripts/run_env.sh ENV=cube-quadruple-play SEED=1 GPU=5 TASK=2
set -uo pipefail

# ── 参数 ──
ENV=""   # scene-play | cube-triple-play | cube-quadruple-play | puzzle-4x4-play
SEED=0
GPU=0
TASK=""  # 留空 = 全跑 1-5

for arg in "$@"; do
  case $arg in
    ENV=*)      ENV="${arg#*=}" ;;
    SEED=*)     SEED="${arg#*=}" ;;
    GPU=*)      GPU="${arg#*=}" ;;
    TASK=*)     TASK="${arg#*=}" ;;
    *) echo "Unknown arg: $arg" && exit 1 ;;
  esac
done

if [ -z "$ENV" ] || [ -z "$GPU" ]; then
  echo "用法: bash scripts/run_env.sh ENV=<env> SEED=<seed> GPU=<gpu> [TASK=<n>]"
  echo ""
  echo "必需参数:"
  echo "  ENV   scene-play | cube-triple-play | cube-quadruple-play | puzzle-4x4-play"
  echo "  GPU   GPU ID"
  echo ""
  echo "可选参数:"
  echo "  SEED  随机种子 (默认 0)"
  echo "  TASK  指定单个 task (默认全跑 1-5)"
  echo ""
  echo "示例:"
  echo "  bash scripts/run_env.sh ENV=scene-play SEED=0 GPU=5"
  echo "  bash scripts/run_env.sh ENV=cube-quadruple-play SEED=1 GPU=5 TASK=2"
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

echo "========================================"
echo "ENV=$ENV  PREFIX=$PREFIX  SEED=$SEED  GPU=$GPU  TASKS=${TASKS[*]}"
echo "========================================"

# ── 训练 + 评估（每 task 训完立即评）──
for t in "${TASKS[@]}"; do
  # -- 训练（原版 QGF: qgf_backup.py, w·∇Q, jac=False）--
  SAVE_DIR=exp/qgf/bc_iql/${ENV}/seed${SEED}/task${t}
  mkdir -p $SAVE_DIR

  echo ""
  echo ">>> [原版 QGF] Training: $PREFIX task$t seed$SEED (qgf_backup.py, w·∇Q, jac=False)"

  if CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl python main.py \
    --seed=$SEED \
    --env_name=${PREFIX}-play-singletask-task${t}-v0 \
    --agent=agents/qgf_backup.py \
    --agent.denoise_steps=10 --agent.expectile=0.9 \
    --agent.action_chunking=True --agent.horizon_length=5 \
    --agent.denoised_action_approx=one_euler_step_approx \
    --agent.apply_jacobian=False \
    --agent.batch_size=1024 \
    --agent.actor_hidden_dims="(1024,1024,1024,1024)" \
    --agent.value_network_kwargs.hidden_dims="(1024,1024,1024,1024)" \
    --agent.discount=0.999 \
    --ogbench_dataset_dir=$OGBENCH_DATA_DIR/${PREFIX}-play-100m-v0/ \
    --dataset_replace_interval=1000 \
    --offline_steps=500000 --eval_interval=100000 --save_interval=100000 \
    --eval_episodes=30 --video_episodes=0 --eval_vecenv_size=1 \
    --guidance_weights=0.004,0.008,0.01,0.02,0.04,0.06,0.08,0.1,0.12 \
    --save_dir=$SAVE_DIR \
    --wandb_run_group=bc_iql; then
    echo "<<< Training done: $SAVE_DIR"
  else
    echo "!!! Training FAILED: $PREFIX task$t seed$SEED (will skip eval)"
    echo "!!! Training FAILED: $PREFIX task$t seed$SEED" >> scripts/failed_runs.log
    continue
  fi

  # -- 评估（时变 QGF: qgf.py, (1-t)/t·∇Q, jac=True）--
  CKPT=$(find "$SAVE_DIR" -name "params_500000.pkl" 2>/dev/null | head -1)

  if [ -z "$CKPT" ]; then
    echo "Skip eval: no checkpoint"
    continue
  fi

  TRAIN_SUBDIR=$(dirname "$CKPT")
  EVAL_DIR=exp/qgf/eval_timetune_jac/${ENV}/seed${SEED}/task${t}
  mkdir -p $EVAL_DIR

  echo ">>> checkpoint: $TRAIN_SUBDIR"
  echo ">>> [时变 QGF] Eval: $PREFIX task$t seed$SEED (qgf.py, (1-t)/t·∇Q, jac=True)"

  if CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl python main.py \
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
    --wandb_run_group=eval_timetune_jac; then
    echo "<<< Eval done: $EVAL_DIR"
  else
    echo "!!! Eval FAILED: $PREFIX task$t seed$SEED"
    echo "!!! Eval FAILED: $PREFIX task$t seed$SEED" >> scripts/failed_runs.log
  fi
done

echo ""
echo "========================================"
echo "All done!"
echo "  Train (原版): exp/qgf/bc_iql/${ENV}/seed${SEED}/"
echo "  Eval  (时变): exp/qgf/eval_timetune_jac/${ENV}/seed${SEED}/"
echo "========================================"
