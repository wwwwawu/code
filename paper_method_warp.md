# WaRP 方法草稿

## 1. 整体命名

建议将整体方法命名为 **WaRP**，全称为 **Waterlogging-aware Residual Patch Refinement**，中文名称为 **积水感知残差补丁细化网络**。

这个名字突出三个核心点：

- **Waterlogging-aware**：方法面向城市道路积水检测与分割任务，而不是通用异常检测。
- **Residual**：通过轻量残差适配器对冻结 ViT 视觉骨干进行任务适配。
- **Patch Refinement**：方法从 patch 级积水响应图出发，并通过细化头恢复更精细的像素级积水区域。

论文中可以将 WaRP 定义为一个面向积水分割的 ViT 适配框架。它使用冻结的视觉基础模型作为特征提取器，只训练少量任务相关模块，从而在保留预训练语义能力的同时提升积水区域定位和边界分割能力。

## 2. 方法总览

给定一张道路图像，WaRP 首先通过统一的 ViT 表征编码器提取多层 patch tokens。为了让通用视觉特征更适合道路积水场景，模型在若干 transformer block 中插入轻量的积水残差适配器。随后，模型引入一对可学习的任务语义 token，分别表示积水区域和背景区域，并通过层自适应空间 token 交互模块，让这两个语义 token 感知不同层级的 patch 信息。

在得到任务感知 token 和多层 patch tokens 后，WaRP 计算每个 patch 与积水 token、背景 token 的相似度差异，生成多层粗粒度积水响应图。由于该响应图本质上来自 patch 网格，上采样后边界仍然较粗糙，因此进一步引入补丁-热图细化头，将多层 patch 特征与粗热力图融合，输出最终像素级积水分割结果。

整体流程可以概括为：

```text
Input Image
  -> Unified ViT Token Encoder
  -> Waterlogging Residual Adapter
  -> Dual Water-Background Tokens
  -> Layer-Adaptive Spatial Token Interaction
  -> Patch-wise Differential Water Scoring
  -> Patch-Heatmap Refinement Head
  -> Final Waterlogging Mask
```

## 3. 模块划分与命名

### 3.1 Unified ViT Token Encoder，统一 ViT 表征编码器

该模块用于将不同 ViT 类视觉基础模型统一成相同的输出格式。对于输入图像，编码器输出若干中间层的 patch tokens，并记录 patch 网格大小、特征维度和层数等信息。

在实现中，该模块可以支持 CLIP ViT、DINOv3 ViT 和 SAM ViT-v1。不同骨干虽然预训练目标和输入预处理不同，但它们都可以被抽象为 patch token 提取器。这样，后续模块不需要关心具体骨干来源，只需要处理统一格式的多层 patch tokens。

论文中建议强调：

- WaRP 是一个 **ViT-style backbone adaptable framework**。
- SAM 部分只声明支持 **SAM ViT-v1**。
- 不声明支持 SAM2 或 SAM3，因为它们的结构已经不再是当前实现所依赖的标准 ViT block 形式。

### 3.2 Waterlogging Residual Adapter，积水残差适配器，WRA

WRA 是 WaRP 的第一个核心模块。它被插入到选定的 ViT block 中，用于对原始视觉特征进行轻量任务适配。设输入特征为 \(x\)，适配器学习一个小幅残差修正：

```text
x' = x + alpha * A(x)
```

其中 \(A(\cdot)\) 是轻量适配分支，\(alpha\) 是可学习或初始化较小的残差缩放系数。这样设计的目的是让模型在训练初期尽量保持预训练骨干的原始表达能力，只逐渐学习与积水分割相关的特征偏移。

WRA 可以描述为两种形式：

- **MLP-WRA**：通过 LayerNorm、降维线性层、非线性激活、门控机制和升维线性层对 token 特征进行语义校正。
- **Conv-WRA**：在 MLP-WRA 的基础上增加 depthwise convolution 分支，用来补充相邻 patch 之间的局部空间关系，增强对积水边界、反光区域和湿滑纹理的建模能力。

模块作用：

- 让冻结的通用 ViT 特征适配道路积水任务。
- 降低全量微调带来的参数量和过拟合风险。
- 保留预训练语义知识，同时增强积水、道路、背景和反光区域的区分能力。

### 3.3 Dual Water-Background Tokens，积水-背景双语义 token，DWB Tokens

WaRP 引入两个可学习任务 token：

- **Water Token**：表示积水区域的任务语义原型。
- **Background Token**：表示非积水背景的任务语义原型。

这两个 token 与图像 patch tokens 一同进入视觉 token 流或在外部骨干适配结构中被维护。训练过程中，Water Token 学习积水区域的共性表达，例如水面、漫反射、高亮反光、道路淹没区域等；Background Token 学习正常道路、车辆、天空、建筑、阴影等非积水区域的表达。

论文中不要使用 anomaly token 和 normal token 这样的异常检测术语，而应统一改写为 water token 和 background token。这样方法会更自然地服务于积水分割任务，而不是显得从异常检测任务迁移而来。

### 3.4 Layer-Adaptive Spatial Token Interaction，层自适应空间 token 交互，LASTI

不同 ViT 层包含不同尺度的信息。浅层特征通常保留更多边缘、纹理和局部结构，高层特征则具有更强语义表达。WaRP 使用 LASTI 模块让 Water Token 和 Background Token 分别从每一层 patch tokens 中吸收空间上下文信息。

LASTI 的核心思想是：不是直接用单个语义 token 去关注大量 patch，而是引入少量可学习 anchor queries，对 patch tokens 进行空间聚合，再通过 token-guided gating 控制聚合信息注入到 Water Token 和 Background Token 的强度。

该模块的作用：

- 让语义 token 针对不同特征层形成不同的积水/背景判别原型。
- 利用 patch 的空间上下文增强 token 表达。
- 减少单个全局 token 对复杂积水形态表达不足的问题。

### 3.5 Patch-wise Differential Water Scoring，补丁差分积水评分，PDWS

PDWS 用于生成粗粒度积水响应图。对于每一层的 patch token \(p_i\)，分别计算其与 Water Token \(t_w\) 和 Background Token \(t_b\) 的余弦相似度，并取二者差值作为积水响应分数：

```text
s_i = cos(p_i, t_w) - cos(p_i, t_b)
```

如果一个 patch 更接近 Water Token，则其响应分数较高；如果更接近 Background Token，则响应分数较低。所有 patch 分数被重新排列为二维 patch 网格，并上采样到输入图像大小，得到单层积水响应图。多个层级的响应图可以相加融合，形成粗粒度热力图。

该模块具有较强可解释性：最终分数直接来自“更像积水还是更像背景”的差异判断。

### 3.6 Patch-Heatmap Refinement Head，补丁-热图细化头，PHR Head

PHR Head 是 WaRP 的第二个核心模块。其设计动机是：PDWS 得到的热力图来自 patch 级评分，即使经过双线性上采样，也难以恢复精细边界和小区域形状。因此，WaRP 使用一个轻量细化头，将多层 patch features 与粗热力图融合，进一步输出像素级分割 logits。

PHR Head 的输入包括：

- 多层 patch features。
- 多层 PDWS 粗响应图融合得到的 coarse heatmap。

处理过程为：

1. 对每一层 patch feature 进行线性投影，统一到相同隐藏维度。
2. 将 patch tokens reshape 成二维特征图。
3. 将不同层的特征图与 coarse heatmap 在通道维度拼接。
4. 通过轻量卷积解码器输出单通道 refined logits。
5. 将 refined logits 上采样到输入图像大小，作为最终积水分割结果。

PHR Head 的作用：

- 修复 patch 上采样导致的边界粗糙问题。
- 利用多层特征补充纹理、边界和局部结构信息。
- 减少湿滑道路、阴影、反光和背景区域造成的局部误检。
- 改善小面积积水区域的完整性。

## 4. 训练目标

WaRP 采用图像级和像素级联合监督。

### 4.1 图像级积水存在性监督

粗热力图或最终热力图通过 Top-K pooling 转换为图像级积水分数。Top-K pooling 只聚合响应最高的一小部分区域，可以减少大面积背景对图像级判断的干扰。该分数通过 BCE loss 与图像级标签进行监督。

### 4.2 像素级积水分割监督

对于 patch 差分评分得到的粗响应图，使用 Focal Loss 和 Dice Loss 进行像素级监督。Focal Loss 用于缓解前景积水区域与背景区域的不平衡，Dice Loss 用于直接优化区域重叠。

对于 PHR Head 输出的 refined mask，同样使用 Focal Loss 和 Dice Loss 进行监督，使细化分支直接学习更准确的积水区域。

### 4.3 语义 token 区分约束

为了避免 Water Token 和 Background Token 学到相似表示，训练中加入 token-level contrastive constraint，使两个任务语义原型保持区分性。这有助于提升 patch 差分评分的稳定性。

总体损失可以写为：

```text
L = L_img + L_coarse_seg + L_refine_seg + L_token
```

其中：

- \(L_img\)：图像级 BCE loss。
- \(L_coarse_seg\)：粗响应图的 Focal + Dice loss。
- \(L_refine_seg\)：细化输出的 Focal + Dice loss。
- \(L_token\)：Water Token 与 Background Token 的区分约束。

当前主实验可以不把 IoU Loss 作为核心损失描述，因为实现中的主要训练配置默认关闭 IoU Loss。

## 5. 推理流程

推理时，输入图像经过 WaRP 得到多层 patch tokens、Water Token 和 Background Token。模型先通过 PDWS 生成多层粗响应图，再由 PHR Head 输出最终 refined logits。随后可以对 logits 进行高斯平滑，并通过 sigmoid 与阈值得到二值积水 mask。

图像级积水分数由最终热力图的 Top-K pooling 得到。像素级指标使用最终 refined logits 或其平滑结果计算。

## 6. 数据集命名建议

如果论文同时贡献数据集，建议将数据集命名为 **RoadWater-Seg**，中文为 **城市道路积水分割数据集**。

备选名称：

- **UW-Seg**：简洁，适合方法论文中的数据集名称。
- **RoadWater-Seg**：更直观，强调道路积水分割任务。
- **UrbanWater-Seg**：更偏城市积水场景，但容易和已有 benchmark 名称接近。

推荐使用 **RoadWater-Seg**。它比 UW-Seg 更容易让读者直接理解任务，也与论文题目和应用场景更匹配。

数据集部分可以强调：

- 包含正常道路和积水道路图像。
- 提供图像级积水标签和像素级积水掩码。
- 覆盖不同道路环境、光照、反光、天气或场景变化。
- 可用于训练和评估道路积水检测、定位和分割模型。

## 7. 论文方法部分推荐结构

论文中的 Method 章节可以按如下结构组织：

```text
3. Method
3.1 Overview
3.2 Unified ViT Token Encoder
3.3 Waterlogging Residual Adapter
3.4 Dual Water-Background Token Learning
3.5 Layer-Adaptive Spatial Token Interaction
3.6 Patch-wise Differential Water Scoring
3.7 Patch-Heatmap Refinement Head
3.8 Training Objectives
```

如果篇幅有限，也可以压缩为：

```text
3. Method
3.1 Overview
3.2 Waterlogging-aware ViT Adaptation
3.3 Token-guided Patch Water Scoring
3.4 Patch-Heatmap Mask Refinement
3.5 Training and Inference
```

## 8. 写作口径注意事项

论文中建议统一使用以下术语：

- anomaly token -> **Water Token**
- normal token -> **Background Token**
- anomaly map -> **Waterlogging Response Map** 或 **Coarse Water Heatmap**
- residual adapter -> **Waterlogging Residual Adapter**
- refined mask head -> **Patch-Heatmap Refinement Head**
- patch similarity score -> **Differential Water Score**

避免在论文正文中出现“异常检测原方法”“工业异常检测改进”“VisualAD”等表述。可以把 WaRP 写成一个独立的道路积水分割框架，其技术动机来自积水任务自身的三个问题：

1. 通用 ViT 表征与道路积水分割之间存在任务差异。
2. 单纯 patch 级响应图上采样后边界粗糙。
3. 积水区域容易与反光、湿滑道路、阴影和复杂背景混淆。

WaRP 的三个对应解决方案是：

1. 用 WRA 对冻结 ViT 表征进行轻量任务适配。
2. 用 PDWS 生成可解释的积水响应图。
3. 用 PHR Head 融合多层 patch 特征和粗热力图，得到精细 mask。

