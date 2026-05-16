# Liquid V2 → V3 改进方案

## 基于自定义图片测试的诊断结果（66 张图片，2026-05-14）

| 预测分布 | 数量 | 比例 |
|----------|------|------|
| full | 35 | 53.0% |
| much | 19 | 28.8% |
| little | 7 | 10.6% |
| empty | 5 | 7.6% |
| half | 0 | 0% |

定量异常指标：
- **12/66 图片 liquid_mask 面积 > bottle_mask 面积**（液体分割溢出瓶体边界）
- **55/66 被判为 full 或 much**，模型系统性地高估液体量
- **binary classifier 几乎总是预测 "有液体"**（binary_prob > 0.99 的比例极高）
- 模型输出呈现极端 overconfidence（state probs 往往 >0.95 集中于某一类）

---

## 问题一：Half 几乎无法识别

### 诊断

LCDTC 验证集上 half recall = 0.554、half F1 = 0.654 已经是最差类别。在自定义图片（不同瓶型、不同饮料颜色、不同光照背景）上模型完全没有预测 half。混淆矩阵显示 half→much（71 例）和 half→little（39 例）是非对称混淆，说明 "half" 的视觉边界在 LCDTC 标注中本身就模糊。

### 解决方案

1. **类别重加权损失**：将 `state_loss_weight` 从统一的 1.0 改为 per-class 权重，half 类给予 2-3× 权重。
2. **Focal Loss 替代 CrossEntropy**：让模型聚焦难分类样本（half 类样本），减少对 easy 样本（empty/full）的过度拟合。
3. **数据增强针对 half**：对 half 类样本做更强的颜色抖动和几何增强，模拟不同饮料颜色和瓶型。
4. **引入新的标注数据**：见问题二的方案。

---

## 问题二：Empty/Little 被系统性地错判为 Full

### 诊断

这是当前最严重的问题。根源有三层：

**A. 训练数据域差距（Domain Gap）**
- LCDTC 的训练图片多为透明/半透明液体、简单背景
- 用户拍摄的图片：有色饮料（茶、可乐）、复杂室内背景、不同瓶型
- 模型从未见过 "有色液体装在透明瓶子里" 的视觉模式，它将瓶壁的反光/折射纹理误认为液体

**B. 透明瓶空 vs 满的物理歧义**
- 透明玻璃瓶 + 透明液体（水）：空瓶和满瓶在二维图像上几乎无法区分
- 人类也需要通过折射率差异（背景变形程度）来判断
- 对于有色饮料这个问题不存在（可以直接看到液面），但模型因为没见过有色液体，把颜色当成了噪声

**C. 模型过度依赖 binary classifier**
- binary 任务（有无液体）过于简单（F1=0.96），模型学会了 "只要看到瓶子就预测有液体"
- binary head 的梯度几乎为零（loss=0.0009），不再提供有用的训练信号
- state classifier 在 binary 已经判 "有液体" 的前提下只能选 little/half/much/full 之一

### 解决方案

**方案 2a（必须）：引入 Roboflow 多类别液体数据集**

Torres (2026) 在 Roboflow 上发布了透明容器液体监测数据集：
`https://universe.roboflow.com/liquidfy/`

该数据集特点：
- 2404 张标注图片，4 个类别：liquid、glass、bottle 等
- 使用 YOLOv11 格式标注，包含检测框
- 涵盖多种液体颜色和容器类型
- 可直接作为 LCDTC 之外的第二个分类数据集加入训练

具体做法：
- 从 Roboflow 下载数据集，转换为与 LCDTC 兼容的 SampleRecord 格式
- 新增 `RoboflowLiquidDataset` 类，与 `LCDTCCropDataset` 和 `TransparentObjectSegDataset` 并列
- 训练循环中三个数据集交替采样，增加数据多样性

**方案 2b（建议）：基于折射分析的物理辅助特征**

针对透明瓶空 vs 满的物理歧义，借鉴 Phys-Liquid (Ma et al. 2025) 的思路：

- 核心原理：光在空气（n≈1.0）、水（n≈1.33）、玻璃（n≈1.5）中的折射率不同。瓶子区域内的背景图案变形程度可以提示内部是空气还是液体。
- 具体实现：在 bottle_mask 区域内计算背景梯度的局部方差。如果瓶内是液体（尤其是有色液体），背景透过的纹理会有明显的折射扭曲；如果是空的，扭曲程度较轻。
- 这可以作为一个**辅助输入通道**或**辅助损失项**，而非替代视觉特征。

简化的工程实现：
1. 在预处理阶段计算瓶体区域的局部梯度幅值
2. 在瓶体 mask 内统计梯度幅值的均值和方差
3. 将这两个标量作为额外特征 concat 到分类器的输入中

**方案 2c（可选）：颜色先验增强**

对于有色饮料（茶色、咖啡色、可乐色），液面以下区域的 HSV 分布与液面以上有明显差异。可以在 bottle_mask 内计算上下两半的颜色直方图距离，作为一个 light-weight 辅助特征。

---

## 问题三：Liquid Mask 溢出瓶体边界

### 诊断

- 12/66 图片 liquid_area > bottle_area（液体分割大于瓶体分割）
- 最严重的两张图片 overflow ratio 分别达 23.0% 和 20.1%
- 根因：bottle_head 和 liquid_head 是两个**独立的分割头**，没有任何约束保证 liquid ⊂ bottle

### 解决方案

**方案 3a（首推）：YOLO 预检测 + 裁剪后送入当前模型**

用户提出的这条路线是正确的。Torres (2026) 论文中使用 YOLOv11 进行透明容器检测，效果良好。

两阶段流水线：
1. **阶段一**：YOLOv11s 检测瓶体 bounding box（可用 Roboflow 数据集训练或直接使用 Torres 发布的预训练权重）
2. **阶段二**：根据检测框裁剪后送入 LiquidV2Net 进行分类和分割

优点：
- 裁剪后瓶体占据画面主体，背景干扰大幅减少
- 自然地约束了 liquid mask 不会溢出（因为图像已经被裁剪到瓶体附近）
- YOLO 检测和 LiquidV2Net 分类可以独立迭代优化

**方案 3b（辅助）：添加溢出惩罚损失**

在训练时增加一个 loss 项，惩罚 liquid mask 中超出 bottle mask 的像素：

```python
def overflow_loss(liquid_logits, bottle_logits):
    liquid_prob = torch.sigmoid(liquid_logits)
    bottle_prob = torch.sigmoid(bottle_logits)
    outside = liquid_prob * (1 - bottle_prob)
    return outside.mean()
```

**方案 3c（辅助）：推理时后处理**

在推理阶段直接对 liquid_mask 做约束：

```python
liquid_mask = liquid_mask * (bottle_mask > 0.5).float()
```

这是最轻量的改动，但不能解决根本问题（模型内部的特征学习）。

---

## 实施路线图

### 第一步（紧急）：推理后处理补丁

- 在 demo notebook 和推理代码中加入 `liquid_mask = liquid_mask * bottle_mask` 的后处理
- 加入 `compute_area_ratio_loss` 的输出，作为定性参考
- **预计工作量**：0.5h

### 第二步（核心）：引入 Roboflow 数据集

- 从 Roboflow 下载 Torres (2026) 数据集
- 编写 `RoboflowLiquidDataset` 适配器
- 修改训练循环支持三个数据集交替
- **预计工作量**：3-4h

### 第三步（模型改进）：损失函数优化

- 将 state loss 从 CrossEntropy 改为 Focal Loss（gamma=2.0）
- 加入 per-class 权重（half 类 3×）
- 加入 overflow 惩罚损失
- **预计工作量**：2h

### 第四步（架构增强）：折射特征 + 颜色先验

- 实现瓶体区域梯度方差特征提取
- 作为额外通道或标量特征注入分类器
- **预计工作量**：4-5h

### 第五步（两阶段流水线）：YOLO 预检测

- 训练/获取 YOLOv11s 瓶体检测器
- 构建 crop → classify 的两阶段推理
- **预计工作量**：5-6h

---

## 依赖的外部资源

| 资源 | 来源 | 用途 |
|------|------|------|
| Roboflow liquid 数据集 | Torres (2026) / `universe.roboflow.com/liquidfy/` | 多类别液体检测数据 |
| Phys-Liquid 模拟数据 | Ma et al. (2025) / `dualtransparency.github.io/Phys-Liquid/` | 可选：物理模拟数据增强 |
| YOLOv11s 预训练权重 | Ultralytics | 瓶体检测预裁剪 |
