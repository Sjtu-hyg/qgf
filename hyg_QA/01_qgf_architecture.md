# QGF 架构、理论与训练详解

---

## 1. QGF 是什么（不是什么）

```
QGF 不是什么：
  ❌ 不是 VLA/VLM — 不看图像，不听语言
  ❌ 没有预训练基础模型 — 不载入 π0.5 或其他 VLM
  ❌ 不需要互联网数据 — 只在 OGBench 仿真数据集上训练

QGF 是什么：
  ✅ 纯状态空间的离线 RL 方法
  ✅ BC Flow Matching policy + IQL critic/value + 推理时 ∇Q 引导
  ✅ 3 个小型 MLP，~16M 参数，173MB checkpoint
  ✅ 在 OGBench 从零训练，单 GPU 数小时完成
```

和 VLA 的关系：把 VLA 的 VLM backbone 去掉，只留 Flow Matching Action Head，在 state 空间（非 latent 空间）训练，配独立 Q 函数 → 就是 QGF。

---

## 2. State 空间设计（OGBench scene 环境源码)

### 2.1 环境描述

Scene 环境（`ogbench/manipspace/envs/scene_env.py`）包含：
- 1 个方块（cube）
- 2 个按钮（button）— 控制抽屉和窗户的锁
- 1 个抽屉（drawer）
- 1 个窗户（window）
- 1 个 UR5e 机械臂 + Robotiq 夹爪

### 2.2 5 个 Task 的真实含义

来自 `set_tasks()` 源码，每个 task 是**不同的初始状态 + 目标状态组合**：

| Task | 任务名 | 目标 |
|------|--------|------|
| task1 | `task1_open` | cube 不动，打开抽屉 + 窗户 |
| task2 | `task2_unlock_and_lock` | 解锁/锁定：cube 不动，关抽屉 + 窗户 |
| task3 | `task3_rearrange_medium` | 移动 cube + 按按钮 + 开抽屉 + 关窗户 |
| task4 | `task4_put_in_drawer` | 把 cube 放进抽屉 |
| task5 | `task5_rearrange_hard` | 把 cube 放抽屉 + 开窗户 |

### 2.3 40 维 Observation 结构

来自 `compute_observation()` 源码：

```
维度  来源                   内容                         缩放
===   ====                   ====                         ====
0-5   proprio/joint_pos      UR5e 6 个关节角度             原始
6-11  proprio/joint_vel      UR5e 6 个关节角速度           原始
12-14 proprio/effector_pos   末端执行器 XYZ (center=0.425,0,0) ×10
15-16 proprio/effector_yaw   末端执行器 yaw (cos, sin)      —
17    proprio/gripper_opening 夹爪开合度                    ×3
18    proprio/gripper_contact 夹爪接触状态                  原始
                     ---  proprio 共 19 维 ---

19-21 block_0_pos           方块 XYZ (center 偏移)         ×10
22-25 block_0_quat          方块四元数 (w,x,y,z)           原始
26-27 block_0_yaw           方块 yaw (cos, sin)            —
                     --- 单方块 9 维 ---

28-29 button_0_state        按钮 0 状态 (one-hot 2)        原始
30    button_0_pos          按钮 0 位置                     ×120
31    button_0_vel          按钮 0 速度                     原始
32-33 button_1_state        按钮 1 状态 (one-hot 2)        原始
34    button_1_pos          按钮 1 位置                     ×120
35    button_1_vel          按钮 1 速度                     原始
                     --- 双按钮 8 维 ---

36    drawer_pos            抽屉滑动位置                    ×18
37    drawer_vel            抽屉滑动速度                    原始
38    window_pos            窗户滑动位置                    ×15
39    window_vel            窗户滑动速度                    原始
                     --- 抽屉+窗户 4 维 ---

总计: 19 + 9 + 8 + 4 = 40
```

### 2.4 Action 空间

5 维连续动作：**末端执行器 3D 位移 + yaw 旋转 + 夹爪开合**，范围 [-1, 1]。

### 2.5 各环境差异

| 环境 | num_cubes | num_buttons | obs 维度 | act 维度 |
|------|-----------|-------------|---------|---------|
| scene | 1 | 2 (+抽屉+窗户) | 40 | 5 |
| cube-triple | 3 | 0 | ~46 | 6 |
| cube-quadruple | 4 | 0 | ~55 | 6 |
| puzzle-4x4 | 0 | 16 | ~68 | 2 |

### 2.6 为什么是 State-based 而非像素

`ob_type` 默认 `'states'`，直接输出低维浮点向量。虽环境也支持 `'pixels'` 模式，但 QGF 纸只用 state。这也是为什么 checkpoint 只有 173MB——没有 CNN/ViT 视觉编码器。

---

### Policy — ActorFlowField (`utils/networks.py:138-169`)

```
输入: [obs + a_noisy + time_embed] → concat → MLP(4×1024, GELU) → Dense(action_dim) → 速度向量 v

obs:   MuJoCo 状态向量（物体位姿、夹爪位置等浮点数，~60-80维）
a_noisy: 当前噪声 action (action_dim × H)
time_embed: 正弦时间编码 (16维)
v:     向量场 a_1 - a_0，指向"从噪声到真实 action"的方向
```

- 激活：GELU + LayerNorm
- 参数量：~3.3M

### Critic — Value (`utils/networks.py:284-343`)

```
输入: [obs + action] → MLP(4×1024) → 1D Q值
     双 Q 集成（ensemblize，axis=0），推理取 min 避免高估
```

- 参数量：~8.4M（双 Q）

### Value — Value（同上，num_ensembles=1）

```
输入: [obs] → MLP(4×1024) → 1D V值
```

- 参数量：~4.2M

### Checkpoint 大小（173MB）

```
总参数 ≈ 16M × 4 bytes × (1 weights + 2 Adam states) ≈ 192MB raw
+ pickle 序列化 overhead → 173MB
```

vs π0.5: VLM backbone 含 >1B 参数 → 11GB，差距 60 倍。

---

## 4. Flow Matching — Policy Loss

### 数学原理

用**直线路径**连接噪声分布和目标分布：

```
a_0 ~ N(0, I)              ← 源分布（纯噪声）
a_1                         ← 目标（数据集中真实 action）
a_t = (1-t)·a_0 + t·a_1     ← 线性插值路径
v_target = a_1 - a_0        ← 目标速度（直线方向）
```

### 代码实现 (`agents/qgf.py:56-81`)

```python
a1 = batch_actions           # 真实 action
a0 = N(0, I)                 # 噪声
t  = randint(0, denoise_steps+1) / denoise_steps  # 离散时间，10步
a_t = a0*(1-t) + a1*t        # 线性插值
vel_target = a1 - a0         # 目标速度

vel_pred = policy(obs, a_t, t)   # 网络预测
bc_loss = MSE(vel_pred, vel_target)
```

**为什么 10 步就够了：** 直线路径每步修正方向确定，不像 DDPM 的随机游走需 100-1000 步。

---

## 5. IQL — Critic & Value Loss

### Critic Loss (`agents/qgf.py:83-94`)

```python
next_v = value(next_obs)                    # 冻结 V 网络取 V(s')
target_q = rewards + discount^H * masks * next_v  # n-step SARSA 目标

qs = critic(obs, batch_actions)             # 双Q，(2,B)
critic_loss = mean((qs - target_q)^2)       # TD 回归
```

### Value Loss (`agents/qgf.py:96-106`)

```python
q = aggregate_q(target_critic(obs, actions))  # min(Q1,Q2)
v = value(obs)
diff = q - v
weight = where(diff>0, expectile, 1-expectile)  # τ=0.9
value_loss = mean(weight * diff^2)
```

expectile=0.9 推动 V ≈ Q 的上 0.9 分位数，**近似 max Q 但不要求逐 action 最大化**，自然避免 OOD action 高估。

### 训练时 Policy/Critic/Value 完全解耦

```python
@jax.jit
def update(self, batch):
    new_policy = apply_loss_fn(policy_loss)        # ① BC 更新
    new_critic = apply_loss_fn(critic_loss)        # ② Q 更新
    target_critic = polyak(critic, target, τ=0.005) # ③ 软更新
    new_value  = apply_loss_fn(value_loss)         # ④ V 更新
    return replace(policy=, critic=, target_critic=, value=)
```

三个网络独立优化，无梯度依赖。

---

## 6. 推理 — Q 梯度引导去噪 (`agents/qgf.py:157-246`)

### 核心 10 步循环

```
a_0 ~ N(0,I)

for t_idx in 0..9:
    t = t_idx / 10
    dt = 0.1

    v_bc = policy(obs, a_t, t)                    # ① BC 速度

    v_bc_sg = stop_gradient(v_bc)                 # ② 阻断 BC 梯度
    a_approx = clip(a_t + (1-t)*v_bc_sg, -1, 1)   #    Euler 步估算干净 action

    q_fn = lambda a: aggregate_q(critic(obs, a)).sum()
    ∇Q = jax.grad(q_fn)(stop_gradient(a_approx))  # ③ 在干净估计上求 ∇Q

    a_{t+1} = a_t + (v_bc + guidance_weight * ∇Q) * dt  # ④ 引导更新
```

### 为什么在 a_approx 上求 ∇Q 而不是 a_t

- a_t 是噪声，Q(obs, a_t) 无意义（OOD）
- a_approx ≈ 干净 action，接近数据分布
- ∇Q_{a_approx} 有物理含义：怎么改 action 能提高 reward

### Best-of-N Rejection Sampling

```python
if N > 1:
    actions = N 次独立去噪 → 选 Q 值最高的
```

---

## 7. Reward 设计（从 ogbench 源码）

### Scene 环境 (`ogbench.relabel_utils.relabel_dataset`)

```python
# 提取每个子目标的实际状态
cube_xyzs = dataset['qpos'][..., obj_start : obj_start+3]  # N个方块位置
target_cube_xyzs = env._data.mocap_pos                      # 方块目标
target_button_states = env._target_button_states            # 按钮目标
target_drawer_pos = env._target_drawer_pos                   # 抽屉目标
target_window_pos = env._target_window_pos                   # 窗户目标

# 逐子目标判断 4cm 阈值达标
cube_successes = ‖cube_xyzs - target_xyzs‖ ≤ 0.04
button_successes = button_states == target_states
drawer_success = |drawer_pos - target| ≤ 0.04
window_success = |window_pos - target| ≤ 0.04

# reward = 达标数 - 总目标数（负值，0 为最优）
rewards = successes.sum(axis=-1) - successes.shape[-1]

# masks: 全部达标→0（终止backup），否则→1
masks = 1.0 - np.all(successes, axis=-1)
```

**物理含义：** 不是连续距离惩罚，是**离散达标制**。每步 reward ∈ [-N, 0]，N 为子目标总数。100% 达标 → reward=0。解释了训练中 rewards ≈ -2/步（约 2 个子目标未达标）。

### Cube 环境

```python
cube_successes = ‖cube_xyzs - target_xyzs‖ ≤ 0.04  # 每方块 4cm 内
rewards = successes.sum() - num_cubes                # -3 到 0（triple），-4 到 0（quadruple）
```

### Puzzle 环境

```python
successes = button_states == target_states            # 按钮状态匹配
rewards = successes.sum() - num_buttons               # 离散达标制
```

### --sparse 模式 (`utils/datasets.py:304-308`)

```python
sparse_rewards = np.where(dense_rewards != 0.0, -1.0, 0.0)
# 所有非零 dense reward → -1（每步惩罚）
# 达标（dense reward=0）→ 0
```

---

## 8. 训练曲线解读

| 指标 | 合理趋势 | 说明 |
|------|---------|------|
| `training/bc_loss` | 下降 | BC 精度提升 |
| `training/critic_loss` | 震荡正常 | V 同步更新导致 TD target 移动 |
| `training/value_loss` | 可升可降 | 不代表 V 质量 |
| Q/V 值 | 可负 | 取决 reward scale，不重要 |
| `evaluation/success` | **唯一金标准** | 任务完成率 |

Q 值从 -300 到 -580 不是变差：训练初期 Q 高估（不够负），收敛后准确反映 -2/步 × 200 步的真实期望回报。

---

## 9. 核心模块速查

| 模块 | 位置 | 功能 |
|------|------|------|
| QGFAgent | `agents/qgf.py` | policy_loss, critic_loss, value_loss, update, sample_actions |
| ActorFlowField | `utils/networks.py:138` | Policy 网络 |
| Value | `utils/networks.py:284` | Q/V 网络，num_ensembles 控制用法 |
| TrainState | `utils/flax_utils.py` | Flax 训练状态，含 apply_loss_fn, target_update |
| Dataset | `utils/datasets.py` | 离线数据集，sample_sequence(H步采样) |
| ReplayBuffer | `utils/datasets.py:326` | 在线 RL Buffer |
| eval_with_test_time_guidance | `utils/evaluation.py` | 评估入口，遍历 guidance weights |
| relabel_dataset | ogbench 包内 | 按 task 目标重标 reward |
| SbatchGenerator | `scripts/generate.py` | 生成 SLURM 脚本 |
