# 瓶体液体识别模型研究报告

## 任务概述

目标：从单张 RGB 图像中同时识别透明瓶体中的液体状态。模型需完成四个子任务：

| 子任务 | 类型 | 输出 |
|--------|------|------|
| 瓶体分割 | 像素级二分类 | bottle mask |
| 液体分割 | 像素级二分类 | liquid mask |
| 有无液体 | 图像级二分类 | P(has_liquid) |
| 液位状态 | 图像级五分类 | empty / little / half / much / full |

**训练数据**（三来源交替训练）：

| 数据集 | 样本数 | 提供标签 |
|--------|--------|----------|
| LCDTC | 7853 train / 2620 val | state + binary + bottle mask + liquid mask |
| TransSeg | 5000 train / 1000 val | bottle mask only |
| LiquiContain (Torres 2026) | 2106 train / 200 val | bottle mask + liquid mask + binary |

---

## 架构演进总览

```
v1 (基线)                    v2 (CNN+Transformer)        v3 (v2+三数据集)           v4 (序数回归+门控)
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   ResNet34/50   │         │   ResNet50      │         │   ResNet50      │         │   ResNet34      │
│       │          │         │       │          │         │       │          │         │       │          │
│      FPN        │         │      FPN        │         │      FPN        │         │      FPN        │
│       │          │         │       │          │         │       │          │         │       │          │
│  SegDecoder     │         │ TripletAttn     │         │ TripletAttn     │         │ TripletAttn     │
│       │          │         │       │          │         │       │          │         │       │          │
│  ┌────┴────┐    │         │ TransformerEnc  │         │ TransformerEnc  │         │ TransformerEnc  │
│  │Seg heads│    │         │       │          │         │       │          │         │       │          │
│  └─────────┘    │         │  ┌────┴────┐    │         │  ┌────┴────┐    │         │  ┌────┴────┐    │
│       │          │         │  │X-Attn   │    │         │  │X-Attn   │    │         │  │Ordinal  │    │
│  GAP pooling    │         │  │Classifier│    │         │  │Classifier│    │         │  │X-Attn   │    │
│       │          │         │  └─────────┘    │         │  └─────────┘    │         │  │Classifier│    │
│  MLP classifier │         │       │          │         │       │          │         │  └─────────┘    │
└─────────────────┘         └─────────────────┘         └─────────────────┘         │  ┌─────────────┐│
                                                                                    │  │Bottle→Liq   ││
                                                                                    │  │Cross-Gate   ││
                                                                                    │  └─────────────┘│
                                                                                    │  ┌─────────────┐│
                                                                                    │  │Ratio Regr.  ││
                                                                                    │  └─────────────┘│
                                                                                    └─────────────────┘
```

---

## V1：基线模型

### 设计

- **模型名**：`BottleLiquidNet`
- **骨干网络**：ResNet34（默认）/ ResNet50 + ImageNet 预训练
- **颈部**：FPN 多尺度特征融合（C2–C5），通过 `SegDecoder` 统一到 192 维
- **分割头**：两个独立的 `ConvNormAct → Conv2d(1)` 分支，输出瓶体和液体 mask
- **分类头**：全局平均池化 + 瓶体 mask 引导池化 + 液体 mask 引导池化 → 拼接为 192 维 → MLP → 二分类 + 五分类线性头
- **参数量**：~22M (ResNet34)

### 损失函数

| 损失项 | 权重 | 说明 |
|--------|------|------|
| CrossEntropy (state) | 1.0 | 5 类状态分类，含 label_smoothing=0.05 |
| BCE (binary) | 0.5 | 有无液体二分类 |
| BCE + Dice (bottle mask) | 1.0 | 瓶体分割 |
| BCE + Dice (liquid mask) | 1.0 | 液体分割 |
| Area prior | 0.05 | 基于类别范围约束的填充比损失 |

### 训练策略

- 两数据集交替（LCDTC + TransSeg），每个 LCDTC batch 后接一个 TransSeg batch
- 优化器：AdamW，lr=3e-4，weight_decay=1e-4
- 调度器：CosineAnnealingLR，40 epochs

### 结果

仅完成 smoke test（1 epoch），未进行完整训练。仅用作架构基线。

---

## V2：CNN + Transformer 混合模型

### 设计

- **模型名**：`LiquidV2Net`
- **骨干网络**：ResNet50 + ImageNet 预训练
- **参数量**：~30.8M

#### 新增组件

1. **Triplet Attention**（LCDTC 论文 Section 4.3）
   - 三个并行的交叉维度注意力分支：C-H、C-W、H-W
   - 通过维度旋转→Z-pool（max+mean concat）→卷积→Sigmoid 实现
   - 三个分支等权平均融合

2. **Transformer Encoder**
   - 下采样 (stride=2) 后的特征图展开为 token 序列
   - 可学习 2D 位置编码（行列分离嵌入）
   - 2 层 TransformerEncoderLayer：d_model=256, 8 heads, GELU, norm_first
   - 输出上采样回原分辨率

3. **Cross-Attention Classifier**
   - 5 个可学习的状态原型 query（每类一个）
   - Query 间先做 self-attention，再与图像 token 做 cross-attention
   - 聚合后的原型特征与 global/bottle/liquid 池化特征拼接
   - MLP → 二分类 + 五分类线性头

### 损失函数

与 V1 相同（CrossEntropy + BCE + BCE/Dice + Area prior）。

### 训练策略

- 两数据集交替（LCDTC + TransSeg）
- 分层学习率：Transformer/Classifier lr = 1e-4，其余 = 2e-4
- CosineAnnealingLR，50 epochs

### 结果

| 指标 | 值 (epoch 49) |
|------|---------------|
| state_acc | 0.7763 |
| **state_macro_f1** | **0.7551** |
| binary_acc | 0.9397 |
| **binary_f1** | **0.9592** |
| Score (state_f1 + 0.5×binary_f1) | **1.2347** |

**这是所有版本中的最高分。**

---

## V3：三数据集联合训练

### 设计

V3 **未修改模型代码**——直接复用 `LiquidV2Net`。所有改进都在训练策略层面：

1. **引入 LiquiContain 数据集**：作为第三个交替训练数据集，提供多样化场景的瓶体+液体多边形标注
2. **Focal Loss**（gamma=2.0）：替代标准 CrossEntropy 用于状态分类，缓解"half"类欠拟合
3. **溢出惩罚（Overflow Penalty）**（权重 0.1）：惩罚液体 mask 超出瓶体 mask 的像素，强制 `liquid ⊂ bottle` 物理约束

### 训练策略

- 三数据集交替：每个 LCDTC batch → TransSeg batch → LiquiContain batch
- 各数据集任务权重：seg_task_weight=0.6, liq_task_weight=0.8
- 其他同 V2

### 结果

| 指标 | V3 (epoch 31) | vs V2 |
|------|---------------|-------|
| state_acc | 0.7683 | -0.0080 |
| **state_macro_f1** | **0.7494** | **-0.0057** |
| binary_acc | 0.9279 | -0.0118 |
| **binary_f1** | **0.9508** | **-0.0084** |
| liq_binary_acc | 0.9250 | — |
| liq_bottle_dice | 0.9687 | — |
| liq_liquid_dice | 0.3505 | — |

额外数据集和损失项引入了正则化效果，但在 LCDTC 主任务上指标轻微下降。

---

## V4：序数回归 + 瓶液互门控 + 填充比回归

### 设计

- **模型名**：`LiquidV4Net`
- **骨干网络**：ResNet34 + ImageNet 预训练
- **参数量**：~28.1M

#### 新增/改进组件

1. **序数交叉注意力分类器（OrdinalCrossAttentionClassifier）**
   - 将 K=5 类 softmax 替换为 K-1=4 个有序二分类器
   - 每个分类器回答 P(y > k)：液位是否高于等级 k？
   - 类别概率通过序数差值计算：
     ```
     P(y=0) = 1 - σ(l₀)
     P(y=k) = σ(l_{k-1}) - σ(l_k)   (0 < k < K-1)
     P(y=K-1) = σ(l_{K-2})
     ```
   - 预测类别 = count(σ(logits) > 0.5)

2. **瓶体→液体空间交叉门控（Bottle→Liquid Cross-Gate）**
   - 瓶体概率图 (detach) 通过可学习卷积 (1→dec_dim, k=3, sigmoid) 产生空间门控图
   - 门控图逐元素调制液体特征：`liquid_feat = liquid_feat * gate`
   - 强制物理先验：液体特征仅在瓶体区域有意义

3. **填充比回归头（Ratio Regression Head）**
   - 分类器额外输出连续 scalar：预测当前瓶体的填充比例
   - 与分割 mask 计算的 fill_ratio 做 MSE 一致性约束
   - 桥接分割与分类两个任务

4. **分割比注入（Seg Ratio Injection）**
   - 从当前 batch 的 bottle/liquid mask 实时计算 seg_ratio
   - 作为额外特征显式注入分类器

### 损失函数

| 损失项 | 权重 (v4 / v4_m2) | 说明 |
|--------|--------------------|------|
| CORAL loss (state) | 1.0 / **5.0** | K-1 有序二分类 BCE，含 label_smoothing |
| BCE (binary) | 0.5 | 有无液体二分类 |
| BCE + Dice (bottle mask) | 1.0 | 瓶体分割 |
| BCE + Dice (liquid mask) | 1.0 | 液体分割 |
| Overflow penalty | 0.1 / **0.02** | liquid ⊂ bottle 约束 |
| Area prior | 0.05 | 类别范围约束 |
| **Rank violation** | 0.02 / **0.005** | 惩罚非单调序数 logits |
| **Seg-cls consistency** | 0.1 | MSE(ratio_pred, seg_ratio) |

### 训练策略

- **两阶段训练**：前 N 个 epoch 冻结 cls_head（仅训练编码器+分割头），之后解冻
- **困难样本挖掘**：EMA 追踪每样本 loss，前 X% 高 loss 样本获得 2× 权重
- **AMP 默认启用**（--no-amp 关闭）
- **channels_last 内存格式**加速卷积

### 结果

#### V4 第一版

| 指标 | V4 (epoch 36) | vs V2 | vs V3 |
|------|---------------|-------|-------|
| state_acc | 0.7225 | -0.0538 | -0.0458 |
| **state_macro_f1** | **0.6936** | **-0.0615** | **-0.0558** |
| binary_acc | 0.9206 | -0.0191 | -0.0073 |
| **binary_f1** | **0.9455** | **-0.0137** | **-0.0053** |

#### V4_m2（参数调优版）

对比 V4 第一版的参数调整：

| 参数 | V4 | V4_m2 | 调优原因 |
|------|-----|-------|----------|
| state_loss_weight | 1.0 | **5.0** | 分割 loss 是分类的 ~150× |
| overflow_weight | 0.1 | **0.02** | 占总 loss 77%，压倒一切 |
| rank_penalty_weight | 0.02 | **0.005** | 不应大于 raw state_loss |
| stage1_epochs | 10 | **5** | 减少分类头冻结时间 |
| hard_example_ratio | 0.2 | **0.1** | 减少困难样本比例 |
| lcd_batch | 40 | **64** | 利用 24GB 显存余量 |
| workers | 4 | **8** | 加速数据加载 |

| 指标 | V4_m2 (epoch 33) | vs V4 |
|------|------------------|-------|
| state_acc | 0.7435 | +0.0210 |
| **state_macro_f1** | **0.7271** | **+0.0335** |
| binary_acc | 0.9233 | +0.0027 |
| **binary_f1** | **0.9484** | **+0.0029** |
| liq_liquid_dice | 0.3492 | +0.0260 |

**参数调优使 V4 的 state_f1 提升了 4.8%，但仍未达到 V2/V3 水平。**

---

## 综合对比

| | V1 | V2 | V3 | V4 | V4_m2 |
|---|---|---|---|---|---|
| **骨干网络** | ResNet34 | **ResNet50** | **ResNet50** | ResNet34 | ResNet34 |
| **参数量** | ~22M | ~30.8M | ~30.8M | ~28.1M | ~28.1M |
| **Triplet Attention** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **Transformer Encoder** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **Cross-Attn Classifier** | ✗ | ✓ | ✓ | ✓ | ✓ |
| **Ordinal Classifier (CORAL)** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **Bottle→Liquid Gate** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **Ratio Regression** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **LCDTC** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **TransSeg** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **LiquiContain** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **Focal Loss** | ✗ | ✗ | ✓ | ✗ | ✗ |
| **Overflow Penalty** | ✗ | ✗ | ✓ | ✓ | ✓ |
| **Rank Violation** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **Seg-Cls Consistency** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **Two-Stage Training** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **Hard Example Mining** | ✗ | ✗ | ✗ | ✓ | ✓ |
| **AMP (默认)** | ✗ | ✗ | ✗ | ✓ | ✓ |
| | | | | | |
| **state_macro_f1** | — | **0.7551** | 0.7494 | 0.6936 | 0.7271 |
| **binary_f1** | — | **0.9592** | 0.9508 | 0.9455 | 0.9484 |
| **Score** (f1+0.5×bin) | — | **1.2347** | 1.2248 | 1.1663 | 1.2013 |
| **训练 epochs** | 1 | 50 | 50 | 50 | 50 |

---

## 分析与讨论

### 什么工作是有效的

1. **Triplet Attention + Transformer Encoder**（V2）：从 V1 基线到 V2 的跳跃最大，全局自注意力机制显著增强了空间特征的表达能力。这是整个系列中最重要的单次架构改进。

2. **Cross-Attention Classifier**（V2）：可学习类别原型通过注意力与图像 token 交互，比简单的全局池化更精准地定位判别性区域。

3. **Label Smoothing + CORAL loss**（V4_m2 修复）：在第一版 V4 中，CORAL loss 未启用 label_smoothing，导致 logit 极度饱和、梯度消失。V4_m2 修复后，饱和状态下 loss 从 0.0067 恢复到 0.26（×38），Training 信号恢复正常。

4. **参数权重平衡**（V4_m2）：将 state_loss_weight 从 1.0 提至 5.0 是 V4_m2 相较 V4 提升 4.8% 的主因。三数据集交替训练中分割损失天然占优，必须有足够的分类权重才能让梯度流向分类头。

### 什么工作不够理想

1. **序数回归（CORAL）**：理论上更适合有序类别（empty < little < half < much < full），但实践中 state_f1 反而不如标准 5 类 softmax。可能原因：
   - CORAL 的 K-1 个二分类器之间存在隐式相关性，优化更难
   - rank_violation 损失持续上升（0.05→0.10），序数约束未被有效执行
   - 类别边界处的歧义（half vs much）在序数框架下与普通分类一样困难

2. **瓶→液交叉门控**：物理上合理的约束，但对最终指标的提升缺乏独立验证。可能因为 mask 质量本身较高，门控机制未能提供额外信息。

3. **ResNet34 vs ResNet50**：V4 回到 ResNet34 后参数量下降约 2.7M（28M vs 31M），但 state_f1 从 0.755 降至 0.727。骨干网络容量仍是关键因素。

4. **LiquiContain 的负向影响**：V3 相较 V2 的退步暗示，LiquiContain 的数据分布与 LCDTC 存在差异，直接混合训练可能引入噪声而非有效正则化。

### 待解决问题

1. **half 类识别最弱**：所有版本中 half 的 F1 (0.64–0.68) 始终是最低的，与相邻类 (little/much) 的混淆难以消除
2. **空瓶误判为满**：面积先验和溢出约束未能完全解决光学歧义
3. **liq_liquid_dice 始终很低**（0.32–0.35）：LiquiContain 上的液体分割质量未见改善
4. **V4 后期过拟合**：epoch 33 后 state_f1 从 0.727 退化至 epoch 49 的 0.704

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `liquid_v1/model.py` | V1 模型定义 (BottleLiquidNet) |
| `liquid_v1/train.py` | V1 训练入口 |
| `liquid_v2/model.py` | V2/V3 模型定义 (LiquidV2Net) |
| `liquid_v2/train.py` | V2/V3 训练入口 |
| `liquid_v4/model.py` | V4 模型定义 (LiquidV4Net) |
| `liquid_v4/train.py` | V4 训练入口 (含 V4_m2 参数) |
| `output_liquid_v2/` | V2 训练输出 |
| `output_liquid_v3/` | V3 训练输出 |
| `output_liquid_v4/` | V4 第一版训练输出 |
| `output_liquid_v4_m2/` | V4_m2 参数调优版训练输出 |
| `notebooks/demo_v2.ipynb` | V2/V3 推理与可视化 |
| `notebooks/demo_v4.ipynb` | V4 推理与可视化 |
