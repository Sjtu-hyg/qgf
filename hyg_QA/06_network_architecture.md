# QGF 网络架构详解：MLP vs U-Net，扩散 vs Flow Matching

---

## 1. QGF 的 Flow Matching 网络架构（代码证据）

### 1.1 整体结构：三个独立 MLP，~16M 参数

QGF 包含 3 个网络，全部是纯 MLP，无卷积、无注意力、无 U-Net：

| 网络 | 类 | 输入 | 隐藏层 | 输出 | 参数量 |
|------|-----|------|--------|------|--------|
| Policy | `ActorFlowField` | obs + a_noisy + t_embed | 4×1024, GELU, LayerNorm | action_dim (5) | ~3.3M |
| Critic (双Q) | `Value(num_ensembles=2)` | obs + action | 4×1024, GELU, LayerNorm | 1 (Q值) | ~8.4M |
| Value | `Value(num_ensembles=1)` | obs | 4×1024, GELU, LayerNorm | 1 (V值) | ~4.2M |

### 1.2 Policy 网络：`ActorFlowField` (`utils/networks.py:138-169`)

```python
class ActorFlowField(nn.Module):
    hidden_dims: Any          # 默认 (512,512,512,512)，实验用 (1024,1024,1024,1024)
    action_dim: int           # 5 (scene) 或 6 (cube)
    mlp_kwargs: dict          # activation=gelu, layer_norm=True
    time_embedding: str = "sinusoidal"

    @nn.compact
    def __call__(self, obs, noised_action, t=None, is_encoded=False):
        parts = [obs, noised_action]
        if t is not None:
            parts.append(embed_time(t, self.time_embedding))  # 16维正弦编码

        concat_input = jnp.concatenate(parts, axis=-1)        # [B, 40+5+16]
        outputs = MLP(self.hidden_dims, activate_final=True)(concat_input)
        v = nn.Dense(self.action_dim)(outputs)                 # [B, 5]
        return v  # 速度向量 field
```

**关键设计决策：**

1. **时间编码** (`utils/networks.py:110-121`)：正弦编码（同 Transformer positional encoding），16 维
   ```python
   def timestep_embedding(t, emb_size=16, max_period=10000):
       t = t * max_period
       half = dim // 2
       freqs = exp(-log(max_period) * arange(0, half) / half)  # 对数间隔频率
       args = t[:, None] * freqs[None]
       embedding = concat([cos(args), sin(args)])              # 16维
   ```
   - 选正弦而非 `raw`（原始标量）因为：多频率捕捉不同时间尺度特征，类比 Transformer 位置编码

2. **输入拼接而非交叉注意力**：将 `[obs, a_noisy, t_embed]` 直接 concat → MLP，而非像 U-Net 中用 cross-attention 注入时间条件。低维数据下 concat 已足够。

3. **`activate_final=True`**：最后一层 MLP 后仍过 GELU 激活，再经 `nn.Dense(action_dim)` 输出（无激活，速度向量可正可负）

### 1.3 Critic/Value 网络：`Value` (`utils/networks.py:284-343`)

```python
class Value(nn.Module):
    num_ensembles: int = 2        # Q: 2, V: 1
    network_class: str = "MLP"    # 或 "BroNet"
    network_kwargs: Dict = dict(
        hidden_dims=(1024,1024,1024,1024),
        layer_norm=True,
        activation=gelu,
    )

    def setup(self):
        network = MLP(hidden_dims=(..., 1), activate_final=False)
        if self.num_ensembles > 1:
            network = ensemblize(network, self.num_ensembles)  # vmap 创建双Q
        self.value_net = network

    def __call__(self, observations, actions=None):
        inputs = concat([observations, actions]) if actions is not None else [observations]
        return self.value_net(inputs).squeeze(-1)  # [B] or [2,B]
```

**双 Q 实现** (`utils/networks.py:17-27`)：通过 `nn.vmap` 而非 for 循环：
```python
def ensemblize(cls, num_qs, ...):
    return nn.vmap(cls,
        variable_axes={"params": 0},      # 参数在第0维堆叠
        split_rngs={"params": True},       # 独立初始化
        axis_size=num_qs,                  # 2
    )
```

### 1.4 配置证据 (`agents/qgf.py:334-370`)

```python
def get_config():
    config = ConfigDict(dict(
        actor_hidden_dims=(512, 512, 512, 512),      # 默认4×512
        activation="gelu",
        use_layer_norm=1,
        num_qs=2,
        value_network_class="MLP",
        value_network_kwargs=dict(
            hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
        ),
        time_embedding="sinusoidal",                  # 正弦时间编码
    ))
```

实验启动时通过 CLI 覆盖为 4×1024：`--agent.actor_hidden_dims=1024,1024,1024,1024`

---

## 2. 为什么扩散模型通常用 U-Net？

### 2.1 U-Net 结构回顾

```
Encoder (下采样)          Bottleneck        Decoder (上采样)
  输入 256×256
  ├─ Conv 64×64  ──────────────────────────→ Conv 64×64    (skip)
  ├─ Conv 32×32  ────────────────────────→ Conv 32×32      (skip)
  ├─ Conv 16×16  ────────────────────→ Conv 16×16          (skip)
  ├─ Conv 8×8    ────────────────→ Conv 8×8                (skip)
  └─ Conv 4×4 ──── [Self-Attn] ────→ Conv 4×4
```

核心三要素：**下采样 → 瓶颈 → 上采样 + 跳跃连接 + 时间条件注入**

### 2.2 为什么 U-Net 适合扩散模型

**① 去噪是逐像素预测任务（image-to-image translation）**

扩散模型的训练目标：给定噪声图 x_t，预测添加的噪声 ε。输入和输出在**同一空间**（像素空间），尺寸相同。U-Net 的 encoder-decoder 结构正是为此设计的——从损坏的输入中恢复干净的输出，本质是 pixel-wise regression。

**② 多尺度特征对去噪至关重要**

去噪需要同时理解：
- **全局结构**（这个物体是猫还是狗？整体布局是什么？）→ 深层、低分辨率特征
- **局部细节**（边缘在哪里？纹理是什么？）→ 浅层、高分辨率特征

U-Net 的层次化结构天然提供这种多尺度表示：deep bottleneck 捕捉语义/全局信息，shallow layers 保留空间细节。

**③ 跳跃连接保留高频信息**

下采样会丢失精确的空间位置信息。跳跃连接将 encoder 的浅层特征直接送入 decoder 对应层，保留了边缘、纹理等高频细节。对去噪来说，这些细节是恢复清晰图像的关键——纯 bottleneck 无法重建。

**④ 时间条件的灵活注入**

扩散模型需要知道当前噪声水平 t。U-Net 中常用两种注入方式：
- **FiLM / AdaGN**：`x = scale(t) * x + shift(t)`，对每层 feature map 做条件归一化
- **Cross-attention**：用 t 的 embedding 作为 query，对 feature map 做交叉注意力

这些方法可以在**所有尺度**注入时间信息，让网络在不同抽象层次都知道当前噪声水平。

**⑤ 归纳偏置匹配噪声预测任务**

噪声 ε ~ N(0,I) 是空间均匀的——每个像素独立同分布。这意味着去噪网络需要对图像做**等变性处理**：平移输入应导致平移输出。CNN 的平移等变性恰好匹配这一需求。Vision Transformer (ViT) 则没有这种内置偏置，需要更多数据/训练来学习。

### 2.3 DiT (Diffusion Transformer) 的挑战

最近 DiT 用 ViT 替代 U-Net 也取得了成功，但：
- 需要更大的数据量（ImageNet 级别的百万级）
- 依赖 adaLN-zero（自适应层归一化）注入时间条件
- 参数量通常更大（DiT-XL/2 ≈ 675M）

U-Net 在中小规模数据和计算预算下仍然高效。

---

## 3. Flow Matching 用什么架构？

### 3.1 FM 与扩散的核心区别决定架构选择

| | 扩散模型 (DDPM) | Flow Matching |
|---|---|---|
| 路径 | 随机游走（Markov chain） | **直线插值** a_t = (1-t)·a_0 + t·a_1 |
| 目标 | 预测噪声 ε（间接） | **直接预测速度** v = a_1 - a_0 |
| 步数 | 100-1000 步 | **5-10 步**（更光滑的 vector field） |
| 难度 | 每步小修正，随机性强 | 每步大步修正，确定性路径 |

FM 的直线路径意味着 vector field 比扩散的 score function **更光滑、更好学**，对网络容量的要求理论上更低。

### 3.2 分层选择：数据维度决定架构

```
高维数据（图像/视频）                低维数据（action/分子/参数）
       │                                      │
  U-Net / DiT                               MLP
  - 空间结构存在                     - 无空间结构
  - 多尺度特征必不可少               - 维度低（5-100），无需层次化
  - 参数多（100M-1B+）              - 参数少（1-10M）
  - 例: Stable Diffusion, Sora       - 例: QGF, 分子生成, 机器人 action
```

**FM 文献案例：**
- **Lipman et al. (2023) "Flow Matching for Generative Modeling"**：2D toy data 用 MLP，图像用 U-Net
- **Tong et al. (2024) "Improving and generalizing flow-based generative models"**：低维实验用 MLP
- **RFM (Rectified Flow Matching)**：低维生成用 MLP，图像生成用 DiT/U-Net
- **扩散策略 (Diffusion Policy)**：机器人 action 用 1D CNN（本质是 MLP 的时间序列版本），**不用 U-Net**
- **QGF**：5D action，纯 MLP

### 3.3 QGF 用 MLP 的合理性

QGF 的 5 维 action 向量没有空间结构——它只是 5 个标量（dx, dy, dz, dyaw, dgripper）。对这样的数据：

- **U-Net 没有意义**：没有空间维度可以下采样/上采样。强行加 1D U-Net 会在 5 个元素上做卷积，本质退化为 MLP。
- **Attention 没有意义**：5 个元素之间的 self-attention 过度参数化，5×5 的注意力矩阵无法学到有用模式。
- **MLP 足够**：4 层 1024 维足以逼近任意光滑函数（Universal Approximation Theorem），且 3.3M 参数用 100M 样本训练完全充分。

**这个选择与 FM 文献一致，也与扩散策略领域（Diffusion Policy 用 1D CNN/MLP 而非 U-Net）一致。**

---

## 4. 总结对比

| | 扩散 (DDPM) | Flow Matching |
|------|-------------|---------------|
| **图像数据** | U-Net / DiT | U-Net / DiT |
| **低维数据** | MLP / 1D CNN | **MLP**（QGF 的选择） |
| **时间编码** | Sinusoidal + 多尺度注入 | **Sinusoidal + concat**（QGF） |
| **条件注入** | Cross-attn / FiLM / AdaGN | **concat**（简单有效） |
| **参数量** | 100M+（图像）| 1-10M（低维） |

QGF 的网络设计遵循了一个简单原则：**数据没有空间结构 → 不需要 U-Net。5 维 action → 纯 MLP 足够。** 这与 Flow Matching 文献的主流做法一致，也与扩散策略（Diffusion Policy）领域的设计共识一致——低维控制信号不需要层次化的视觉架构。

---

## 5. QGF 实验中的 FM 输入输出（逐维度拆解）

### 5.1 Policy 网络前向：`ActorFlowField.__call__`

`agents/qgf.py:156-170` + `utils/networks.py:156-169`：

```python
# sample_actions() 中的调用（denoising 循环内）
v_bc = self.policy(observations, a, ti)

# ActorFlowField.__call__:
parts = [obs, noised_action]                     # obs: [B, 40], a: [B, 5]
parts.append(sinusoidal_embedding(t))            # t_embed: [B, 16]
concat_input = jnp.concatenate(parts, axis=-1)   # → [B, 61]
outputs = MLP([1024,1024,1024,1024], GELU, LN)(concat_input)  # → [B, 1024]
v = nn.Dense(5)(outputs)                         # → [B, 5]  速度向量
```

### 5.2 输入维度（以 scene-play 为例）

| 分量 | 维度 | 含义 | 来源 |
|------|------|------|------|
| `observations` | **40** | MuJoCo 状态向量 | 环境 `compute_observation()` |
| `noised_action` | **5** | 当前噪声 action a_t | 随机采样/上一步积分 |
| `t_embedding` | **16** | 正弦时间编码 | `timestep_embedding(t, emb_size=16)` |
| **总输入** | **61** | concat 拼接 | — |
| **输出 v** | **5** | 速度向量 (a₁-a₀ 方向) | 指向干净 action |

### 5.3 四个环境的输入差异

| 环境 | obs 维度 | act 维度 | 总输入维 | 策略输入形状 |
|------|---------|---------|---------|-------------|
| scene-play | 40 | 5 | 61 | `[B, 40+5+16]` |
| cube-triple-play | ~46 | 6 | ~68 | `[B, 46+6+16]` |
| cube-quadruple-play | ~55 | 6 | ~77 | `[B, 55+6+16]` |
| puzzle-4x4-play | ~68 | 2 | ~86 | `[B, 68+2+16]` |

### 5.4 输入为什么没有图像？

- `ob_type` 默认 `'states'`（`utils/datasets.py`），直接使用 MuJoCo 底层状态向量
- 环境支持 `'pixels'` 模式但 QGF 论文未使用
- 这是 QGF 区别于 VLA（如 π0.5）的根本原因：**纯 state-based，无视觉编码器**

### 5.5 Critic/Value 输入

```python
# Critic (Q): 输入 [obs + action] → 1D Q值
critic(observations=[B,40], actions=[B,5]) → [2,B]  # 双Q

# Value (V): 输入 [obs] → 1D V值  
value(observations=[B,40]) → [B]
```

---

## 6. VLA 中的 Flow Matching：π0.5 架构对比

### 6.1 总体架构：三段式，~3.3B 参数

```
┌──────────────┐    ┌────────────────┐    ┌─────────────────────┐
│ SigLIP ViT   │    │  Gemma 2B      │    │  Action Expert      │
│ (~400M)      │───▶│  (~2.6B)       │───▶│  (~300M, 18层 Trans)│
│              │    │                 │    │                     │
│ 图像→patch   │    │ VLM 语义理解    │    │ Flow Matching 动作  │
│ token (256)  │    │ width=2048     │    │ width=1024          │
│              │    │ depth=18       │    │ depth=18            │
│              │    │ mlp_dim=16384  │    │ mlp_dim=4096        │
└──────────────┘    └────────────────┘    └─────────────────────┘
    图像编码             语言+视觉融合            连续动作生成
```

### 6.2 Action Expert 网络规格（代码证据）

来自 `openpi/models/gemma.py`：

```python
action_expert_config = Config(
    width=1024,          # 隐藏层维度
    depth=18,            # Transformer 层数
    mlp_dim=4096,        # FFN 中间维度
    num_heads=8,         # 注意力头数
    num_kv_heads=1,      # GQA: 8 query 头共享 1 个 KV 头
    head_dim=256,        # 每头维度 (8×256=2048 → 实际用了不同的投影)
)
# 总参数 ≈ 311M
```

**时间条件注入：AdaRMSNorm**（非 concat！）

```python
# π0.5 每层 Transformer：
x = AdaRMSNorm(x, flow_timestep_embedding)   # 用去噪步数 τ 调制归一化参数
x = x + SelfAttention(x)                      # Blockwise Causal Mask
x = AdaRMSNorm(x, flow_timestep_embedding)
x = x + FFN(x)
x = CrossAttention(x, kv_from_gemma_2b)       # 交叉注意力：从 VLM 提取特征
```

### 6.3 π0.5 FM 的输入输出

| 分量 | 维度 | 含义 |
|------|------|------|
| **Action Expert 输入** | `[B, 50, 1024]` | 50 步噪声 action 块（经 embedding） |
| **VLM KV cache** | `[B, prefix_len, 2048]` | Gemma 2B 输出（图像+文本+状态理解） |
| **Flow timestep τ** | 标量 | 当前去噪进程 τ ∈ [0, 1] |
| **Action Expert 输出** | `[B, 50, action_dim]` | 50 步干净 action 块（~1 秒动作） |

**条件机制**：
- **QGF**：`concat(obs, a_noisy, t_embed)` → MLP，简单粗暴
- **π0.5**：VLM 输出通过 **Cross-Attention** 注入每个 Transformer 层，时间 τ 通过 **AdaRMSNorm** 注入每层

### 6.4 QGF vs π0.5 FM 全面对比

| 维度 | QGF | π0.5 Action Expert |
|------|-----|-------------------|
| **数据模态** | 纯状态向量（40维） | 图像 + 语言 + 状态 → VLM latent |
| **网络架构** | **MLP** (4×1024) | **Transformer Decoder** (18层×1024) |
| **参数量** | **3.3M** | **311M** |
| **时间编码** | Sinusoidal 16维 | Sinusoidal → AdaRMSNorm |
| **条件注入** | **concat** (obs+a+t) | **Cross-Attn** (VLM KV) + **AdaRMSNorm** (t) |
| **动作表示** | 5维连续向量 | 50步 action chunk |
| **FM 路径** | 直线插值: a_t = (1-t)·a₀ + t·a₁ | 线性高斯: a_t^τ = τ·a + (1-τ)·ε |
| **去噪步数** | 10 | 10 |
| **步长** | dt = 0.1 | δ = 0.1 |
| **时间采样** | 均匀离散 (0/10, 1/10, ...) | Shifted Beta(1.5, 1)：偏向高噪声区域 |
| **推理耗时** | ~ms（单 GPU） | ~27ms（×10 action forward on RTX 4090） |
| **总参数量** | **~16M**（3网络合计） | **~3.3B**（含 VLM backbone） |
| **训练数据** | 100M transitions（仿真） | 10k+ 小时真实机器人数据 + 网络数据 |
| **适用场景** | 固定环境离线 RL | 开放世界通用机器人操控 |

### 6.5 为什么架构差异如此巨大？

核心原因只有一条：**输入数据的复杂度不同**。

```
QGF 的输入: 40 个浮点数 → MLP 直接拟合映射
π0.5 的输入: 多视角图像 + 自然语言指令 → 需要 VLM 先理解语义

QGF 只需学习: "状态 → 动作"的映射
π0.5 需要学习: "看到杯子→理解'拿起杯子'→生成抓取轨迹"的完整管线
```

**Flow Matching 本身只负责最后一步**——从噪声生成连续动作。QGF 和 π0.5 在 FM 这一环的核心数学完全一致（直线路径、10步积分、速度预测），差异全在**条件信号的来源**上：

- QGF：条件来自 40 维状态 → concat → MLP
- π0.5：条件来自 2.6B VLM 的语义理解 → Cross-Attention → Transformer

**这恰好印证了 §3.2 的结论：FM 的架构选择取决于数据，不取决于方法本身。**
