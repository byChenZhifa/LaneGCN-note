# 1.以下是对`layers.py`脚本的解析与核心代码注释：

### 核心代码注释

```python
# -*- coding: utf-8 -*-
# 功能：实现自定义神经网络层，包含多种基础模块与特征提取方法

### 核心模块解析 ###

# -------------------------- 1. 基础卷积模块 --------------------------
class Conv(nn.Module):
    """2D卷积+归一化+ReLU三件套
    原理：通过组归一化(GN)或批归一化(BN)稳定训练过程
    应用：用于处理二维特征图（如BEV特征）
    """
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm="GN", ng=32, act=True):
        super().__init__()
        # 卷积层定义
        self.conv = nn.Conv2d(n_in, n_out, kernel_size,
                            padding=(kernel_size-1)//2,  # 保持特征图尺寸
                            stride=stride, bias=False)

        # 归一化选择
        if norm == "GN":
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)  # 组归一化
        elif norm == "BN":
            self.norm = nn.BatchNorm2d(n_out)                # 批归一化

        self.relu = nn.ReLU(inplace=True) if act else None   # 激活函数

class Conv1d(nn.Module):
    """1D卷积变体，结构与Conv类似
    应用：处理序列数据（如车辆轨迹）
    """

# -------------------------- 2. 残差模块 --------------------------
class PostRes(nn.Module):
    """标准残差块（ResNet风格）
    原理：通过跳跃连接解决梯度消失问题，结构：
        x → conv1 → norm1 → relu → conv2 → norm2 → + → relu
        |__________________________________________|
    """
    def __init__(self, n_in, n_out, stride=1, norm="GN", ng=32, act=True):
        super().__init__()
        # 主路径
        self.conv1 = nn.Conv2d(n_in, n_out, 3, stride, 1, bias=False)
        self.conv2 = nn.Conv2d(n_out, n_out, 3, 1, 1, bias=False)

        # 跳跃连接（维度匹配时使用Identity，否则用1x1卷积）
        self.downsample = nn.Sequential(
            nn.Conv2d(n_in, n_out, 1, stride, bias=False),
            nn.GroupNorm(gcd(ng, n_out), n_out) if norm=="GN"
            else nn.BatchNorm2d(n_out)
        ) if (stride!=1 or n_in!=n_out) else None

class Res1d(nn.Module):
    """1D残差块
    应用：在ActorNet中处理车辆轨迹序列
    """

class LinearRes(nn.Module):
    """线性残差块
    原理：将残差思想应用于全连接层
        x → linear1 → norm1 → relu → linear2 → norm2 → + → relu
        |_____________________________________________|
    """

# -------------------------- 3. 特征采样模块 --------------------------
def linear_interp(x, n_max):
    """双线性插值核心函数
    输入：归一化坐标x ∈ [0,1]，映射到n_max维度的绝对坐标
    输出：左右相邻点的权重和索引
    算法步骤：
        1. 将x从[0,1]映射到[-0.5, n_max-0.5]
        2. 计算左右相邻整数坐标
        3. 根据距离计算权重
    """
    x = x * n_max - 0.5  # 坐标变换
    n = torch.floor(x)    # 左侧索引
    rw = x - n           # 右侧权重
    li = n.long()        # 左侧索引
    ri = li + 1          # 右侧索引

def get_roi_feat(fm, bboxes, roi_size, pts_range):
    """ROI特征提取（Rotated ROI Align）
    原理：根据旋转框参数生成采样网格，通过双线性插值获取特征
    流程：
        1. 根据bbox参数生成采样点网格（考虑旋转角度）
        2. 将采样点坐标归一化到特征图范围
        3. 使用linear_interp进行双线性插值
        4. 聚合有效区域的特征
    应用：用于从BEV特征图中提取旋转区域特征
    """
    # 生成旋转采样网格
    rot_mat = ...  # 根据角度计算旋转矩阵
    offset = ...   # 生成相对采样点
    x, y = ...     # 应用旋转矩阵后的绝对坐标

    # 坐标归一化
    x = (x - x_min) / (x_max - x_min)
    y = (y_max - y) / (y_max - y_min)

    # 双线性插值
    lw, li, rw, ri = linear_interp(...)
    feat = (lw * fm[li] + rw * fm[ri])  # 加权求和

### 其他辅助模块 ###
class Null(nn.Module):
    """空操作层（占位符）"""

### 模块应用场景 ###
"""
+----------------+-----------------------------+
|     模块       |         主要应用场景        |
+----------------+-----------------------------+
| Res1d         | ActorNet轨迹特征提取        |
| MapNet        | 车道线图卷积网络            |
| LinearRes     | 预测头的特征变换            |
| get_roi_feat  | 目标检测中的旋转ROI提取     |
+----------------+-----------------------------+
"""

### 关键设计思想图解 ###
"""
Res1d 残差连接示意图：
Input
  │
  ├─→ Conv1d → BN → ReLU → Conv1d → BN →
  │                                    ⊕
  └────────────────────────────────────┘
                                       │
                                    Output

Rotated ROI Align 流程：
1. 定义旋转框参数 (cx, cy, w, h, θ)
2. 生成采样网格（考虑旋转角度）
   ┌──────────────┐
   │  bbox中心    │
   │    ┌───────┐ │
   │    │  ROI  │ │ → 通过旋转矩阵计算采样点
   │    └───────┘ │
   └──────────────┘
3. 双线性插值获取特征值
"""
```

### 核心模块原理解释

1. **组归一化 vs 批归一化**

```mermaid
graph LR
    A[输入特征图] --> B[GN: 按通道分组归一化]
    A --> C[BN: 按批次维度归一化]

    B --> D[更适合小batch/分布式训练]
    C --> E[需要足够大的batch size]
```

2. **残差学习机制**

- **问题**：深层网络梯度消失/爆炸
- **解决**：引入跳跃连接（Shortcut Connection）
- **公式**：$H(x) = F(x) + x$
- **优势**：允许梯度直接回传，训练更深的网络

3. **旋转 ROI Align**

- **传统 ROI Pooling**：轴对齐的矩形区域，无法处理旋转目标
- **改进点**：
  - 通过旋转矩阵生成采样网格
  - 双线性插值保持亚像素精度
- **数学过程**：
  ```
  x' = (x - cx) * cosθ - (y - cy) * sinθ + cx
  y' = (x - cx) * sinθ + (y - cy) * cosθ + cy
  ```

4. **1D 卷积设计**

- **输入数据**：时序轨迹数据（n_ctrs × seq_len × feat_dim）
- **操作**：在序列长度维度进行卷积
- **优势**：捕获局部时序模式，参数效率高于全连接

### 性能优化技巧

1. **原位操作**：`ReLU(inplace=True)` 减少内存占用
2. **参数初始化**：卷积层默认无 bias（由归一化层补偿）
3. **稀疏计算**：`get_roi_feat`中通过 mask 过滤无效采样点
4. **张量视图优化**：`view()`和`transpose()`的连续化处理

# 2

根据代码实现，Conv 和 Conv1d 的主要区别如下：

```python
# 维度差异（核心区别）
class Conv(nn.Module):   # 2D卷积
    self.conv = nn.Conv2d(...)  # 处理空间特征（h x w）
    self.norm = nn.GroupNorm(...)  # GroupNorm/BatchNorm2d

class Conv1d(nn.Module): # 1D卷积
    self.conv = nn.Conv1d(...)  # 处理序列特征（length）
    self.norm = nn.GroupNorm(...)  # GroupNorm/BatchNorm1d
```

具体差异对比：

1. **输入维度**：

   - Conv：处理 2D 特征图（形状为 B×C×H×W）
   - Conv1d：处理 1D 序列数据（形状为 B×C×L）

2. **应用场景**：

   - Conv：用于处理 BEV（鸟瞰图）等二维空间特征
   - Conv1d：处理车辆轨迹等时序数据（代码注释说明其优势：多尺度特征提取 + 并行计算效率）

3. **归一化层**：

   - Conv：使用 nn.BatchNorm2d / nn.GroupNorm（二维）
   - Conv1d：使用 nn.BatchNorm1d / nn.GroupNorm（一维）

4. **计算特性**：

   - Conv1d 相比 RNN 的优势（如注释所述）：
     ✓ 能捕获局部时序模式
     ✓ 支持并行计算加速训练
     ✓ 通过堆叠实现多尺度特征提取

5. **参数差异**：
   - kernel_size 维度：Conv 用 3x3 核，Conv1d 用 3 点核
   - padding 计算：Conv 保持二维尺寸，Conv1d 保持一维长度

两种结构在模型中的典型应用场景：

```python
# Conv处理BEV特征示例
bev_feat = Conv(64, 128)(bev_input)  # 输入形状：(16,64,200,200)

# Conv1d处理轨迹示例
traj_feat = Conv1d(3, 64)(traj_input)  # 输入形状：(16,3,20) (x,y,t坐标序列)
```

# 更加完善的解释

以下是为`lanegcn.py`添加的完整注释。由于篇幅限制，我将展示部分核心模块的注释示例，您可以根据需要扩展到整个文件：

```python
# Copyright (c) 2020 Uber Technologies, Inc.
# ... [原有版权信息保留]

"""
LaneGCN 轨迹预测模型
核心功能：通过融合车辆轨迹特征与高精度地图特征，实现多模态轨迹预测
主要组成部分：
1. ActorNet - 车辆轨迹特征提取
2. MapNet - 地图拓扑结构特征提取
3. 融合模块（A2M/M2M/M2A/A2A）- 车路交互建模
4. PredNet - 多模态轨迹预测
"""

import numpy as np
import os
# ... [原有导入保留]

### 全局配置 ###
config = dict()
"""训练相关"""
config["display_iters"] = 205942  # 训练日志打印间隔
config["val_iters"] = 205942 * 2  # 验证间隔
# ... [其他配置参数保留原有注释]

class Net(nn.Module):
    """
    功能：LaneGCN 主干网络
    实现原理：
    - 通过ActorNet和MapNet分别提取车辆和地图特征
    - 使用四个融合模块进行车路交互
    - 最终通过PredNet生成预测轨迹

    输入输出：
    输入:
        data (Dict) - 包含以下键值：
            "feats": 车辆轨迹特征 [batch_size, seq_len, 2]
            "ctrs": 车辆中心坐标 [batch_size, 2]
            "graph": 地图图结构数据
    输出:
        Dict[str, List[Tensor]] - 包含预测结果：
            "cls": 各模态置信度 [batch_size, num_mods]
            "reg": 预测轨迹 [batch_size, num_mods, num_preds, 2]
    """
    def __init__(self, config):
        super(Net, self).__init__()
        # 初始化各子模块
        self.actor_net = ActorNet(config)  # 车辆特征提取器
        self.map_net = MapNet(config)       # 地图特征提取器
        # ... [其他模块初始化]

    def forward(self, data: Dict) -> Dict[str, List[Tensor]]:
        # 特征提取阶段
        actors, actor_idcs = actor_gather(gpu(data["feats"]))  # 聚合批次车辆特征 [total_actors, seq_len, 2]
        actor_ctrs = gpu(data["ctrs"])                         # 车辆中心坐标 [total_actors, 2]
        actors = self.actor_net(actors)                        # 提取车辆特征 [total_actors, n_actor]

        graph = graph_gather(to_long(gpu(data["graph"])))      # 聚合地图图结构
        nodes, node_idcs, node_ctrs = self.map_net(graph)      # 提取地图特征 [total_nodes, n_map]

        # 特征融合阶段
        nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)  # Actor->Map融合
        nodes = self.m2m(nodes, graph)                                  # Map内部传播
        actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, ...)  # Map->Actor融合
        actors = self.a2a(actors, actor_idcs, actor_ctrs)              # Actor间交互

        # 轨迹预测
        out = self.pred_net(actors, actor_idcs, actor_ctrs)  # 生成多模态预测
        # 坐标系转换
        rot, orig = gpu(data["rot"]), gpu(data["orig"])      # 旋转矩阵和平移向量
        for i in range(len(out["reg"])):
            # 将局部坐标转换为全局坐标
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(1,1,1,-1)
        return out

class ActorNet(nn.Module):
    """
    功能：车辆轨迹特征提取网络
    实现原理：
    - 使用多层1D残差卷积提取时序特征
    - 通过特征金字塔融合多尺度特征

    输入输出：
    输入:
        actors (Tensor) - 车辆轨迹 [num_actors, seq_len, 2]
    输出:
        Tensor - 提取的特征 [num_actors, n_actor]
    """
    def __init__(self, config):
        super(ActorNet, self).__init__()
        # 构建3层残差卷积块
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        # ... [具体初始化代码]

    def forward(self, actors: Tensor) -> Tensor:
        out = actors.transpose(1, 2)  # [num_actors, 2, seq_len]
        outputs = []
        # 通过各卷积层
        for i in range(len(self.groups)):
            out = self.groups[i](out)  # 每层输出 [num_actors, n_out[i], seq_len//2^i]
            outputs.append(out)
        # 特征融合
        out = self.lateral[-1](outputs[-1])  # 最深层特征
        for i in reversed(range(len(outputs)-1)):
            out = F.interpolate(out, scale_factor=2, mode="linear")  # 上采样
            out += self.lateral[i](outputs[i])  # 特征相加
        return out[:, :, -1]  # 取最后时间步特征 [num_actors, n_actor]

class Att(nn.Module):
    """
    功能：注意力机制模块（用于A2M/M2A等）
    实现原理：
    1. 计算query与context节点的空间关系
    2. 通过可学习参数融合空间关系和特征
    3. 使用index_add进行高效的特征聚合

    输入输出：
    输入:
        agts (Tensor): 目标节点特征 [num_agts, feat_dim]
        ctx (Tensor): 上下文节点特征 [num_ctx, feat_dim]
        空间坐标信息等
    输出:
        Tensor - 更新后的目标节点特征 [num_agts, feat_dim]
    """
    def __init__(self, n_agt: int, n_ctx: int):
        super(Att, self).__init__()
        # 空间编码层
        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),  # 将相对坐标映射到高维
            nn.ReLU(),
            Linear(n_ctx, n_ctx, norm="GN")
        )
        # 注意力计算层
        self.query = Linear(n_agt, n_ctx)  # 将目标特征转换为query

    def forward(self, agts, agt_idcs, agt_ctrs, ctx, ctx_idcs, ctx_ctrs, dist_th):
        # 计算节点间距离
        hi, wi = [], []
        for i in range(len(agt_idcs)):
            dist = agt_ctrs[i].unsqueeze(1) - ctx_ctrs[i].unsqueeze(0)
            dist = torch.norm(dist, dim=-1)
            mask = dist <= dist_th  # 基于距离阈值筛选相邻节点
            # 收集有效索引...

        # 计算注意力特征
        dist_feat = self.dist(agt_ctrs[hi] - ctx_ctrs[wi])  # 空间编码
        query = self.query(agts[hi])                        # query特征
        ctx_feat = torch.cat([dist_feat, query, ctx[wi]], dim=1)
        ctx_out = self.ctx(ctx_feat)  # 特征融合

        # 特征聚合
        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx_out)  # 高效聚合操作
        return agts

# ...[其他类/函数类似方式添加注释]
```

关键注释策略说明：

1. 类/函数级注释：

```python
class ClassName(nn.Module):
    """
    功能：核心功能描述
    实现原理：1-2句话说明算法原理
    输入输出：
        输入: 各参数含义及shape
        输出: 返回值含义及shape
    """
```

2. 代码段注释：

```python
# 特征聚合：使用index_add高效更新目标节点特征
agts.index_add_(0, hi, ctx_out)  # dim, index, source
```

3. 重要参数注释：

```python
config["n_actor"] = 128  # 车辆特征维度
config["actor2map_dist"] = 7.0  # 车辆到车道节点的最大交互距离
```

4. 复杂操作注释：

```python
# 相对坐标编码公式：sign(x)*log(|x|+1) 缓解数值不稳定
dist = torch.cat(
    (torch.sign(x)*torch.log(torch.abs(x)+1.0),
     torch.sign(y)*torch.log(torch.abs(y)+1.0)), 1)
```

完整注释应包含：

1. 模块功能与原理
2. 数据流维度变化
3. 关键超参数含义
4. 复杂数学操作解释
5. 各 tensor 的物理意义

建议在 IDE 中结合代码结构逐模块添加，保持注释与代码的对应关系。对于已有中文注释的部分，可以优化表述或补充实现细节。

# 33

以下是对 LaneGCN 各模块的逐层注释及功能解释：

### 一、ActorNet（车辆轨迹特征提取）

```python
class ActorNet(nn.Module):
    def __init__(self, config):
        # 输入: [batch_size, seq_len, 3] (x,y坐标+时间戳)
        # 输出: [batch_size, n_actor=128]
        groups = [
            Res1d(3, 32, norm="GN", ng=1),  # 输入3通道，输出32通道
            Res1d(32, 64, stride=2, norm="GN", ng=1),  # 下采样，特征维度64
            Res1d(64, 128, stride=2, norm="GN", ng=1)  # 再次下采样，特征维度128
        ]
        self.groups = nn.ModuleList(groups)  # 多尺度特征提取

        lateral = [
            Conv1d(32, 128),  # 32->128通道转换
            Conv1d(64, 128),
            Conv1d(128, 128)
        ]
        self.lateral = nn.ModuleList(lateral)  # 特征融合

    def forward(self, actors):
        # actors输入形状: [total_actors, 3, 20]
        # 输出形状: [total_actors, 128]
```

功能说明：

1. **Res1d 残差块**：通过 3 层 1D 卷积逐步提取多尺度轨迹特征
   - 第一层：32 通道，保持时序长度
   - 第二层：64 通道，时序长度减半（stride=2）
   - 第三层：128 通道，时序长度再减半
2. **特征融合层**：将不同尺度的特征图上采样后相加，实现多尺度特征融合
3. **输出层**：取最后一个时间步的特征作为车辆编码

---

### 二、MapNet（地图拓扑特征提取）

```python
class MapNet(nn.Module):
    def __init__(self, config):
        # 输入:
        #   ctrs: [num_nodes, 2] (车道中心点坐标)
        #   feats: [num_nodes, 2] (车道方向向量)
        # 输出: [num_nodes, n_map=128]
        self.input = nn.Sequential(
            Linear(2, 128),  # 坐标编码
            Linear(128, 128)
        )
        self.seg = nn.Sequential(
            Linear(2, 128),  # 方向向量编码
            Linear(128, 128)
        )

        # 图卷积层配置
        self.fuse = nn.ModuleDict({
            "pre0": [GroupNorm(16, 128)],  # 分组数16 (128%16=0)
            "suc0": [GroupNorm(16, 128)],
            # ...其他连接类型
        })
```

功能说明：

1. **坐标编码层**：将车道中心点坐标映射到 128 维特征
2. **方向编码层**：将车道方向向量编码为 128 维
3. **图卷积层**：通过`GroupNorm+Linear`实现多跳邻域信息聚合
   - 支持 6 个尺度（num_scales）的邻域连接
   - 使用分组归一化（16 组）稳定训练

---

### 三、Actor-Map 融合模块

#### 1. A2M（Actor 到 Map 信息传递）

```python
class A2M(nn.Module):
    def __init__(self, config):
        # 输入:
        #   actors: [N_actor, 128]
        #   nodes: [N_node, 128]
        # 输出: [N_node, 128]
        self.att = Att(n_map=128, n_actor=128)  # 注意力机制

    def forward(self, nodes, graph, actors):
        # 通过注意力机制将车辆特征融合到车道节点
```

#### 2. M2M（Map 内部信息传播）

```python
class M2M(nn.Module):
    # 结构与MapNet类似，使用图卷积层
    # 输入输出均为[N_node, 128]
    # 通过多层GN+Linear实现车道间信息传递
```

#### 3. M2A（Map 到 Actor 信息传递）

```python
class M2A(nn.Module):
    # 逆向注意力机制，将车道特征融合到车辆节点
    # 输入输出维度与A2M相反
```

#### 4. A2A（Actor 交互建模）

```python
class A2A(nn.Module):
    # 车辆间注意力机制
    # 输入输出均为[N_actor, 128]
```

---

### 四、PredNet（轨迹预测头）

```python
class PredNet(nn.Module):
    def __init__(self, config):
        # 输入: [N_actor, 128]
        # 输出:
        #   reg: [batch_size, num_modes=6, pred_steps=30, 2]
        self.pred = nn.ModuleList([
            LinearRes(128, 128) + Linear(128, 60)  # 每个模态预测30个(x,y)点
            for _ in range(6)  # 6个预测模态
        ])
```

---

### 五、关键层结构说明（layers.py）

#### 1. Res1d 残差块

```python
class Res1d(nn.Module):
    """
    输入形状: [batch, channels, seq_len]
    输出形状: 保持输入形状
    结构：
        Conv1d → GN → ReLU → Conv1d → GN → Add → ReLU
    功能：提取时序特征，残差连接避免梯度消失
    """
```

#### 2. LinearRes 线性残差

```python
class LinearRes(nn.Module):
    """
    输入形状: [..., features]
    输出形状: 保持特征维度
    结构：
        Linear → GN → ReLU → Linear → GN → Add → ReLU
    功能：全连接层的残差变体，增强特征表达能力
    """
```

#### 3. GroupNorm 配置

```python
# 修改后的GroupNorm层（解决ONNX兼容性）
groups = 16 if n_out % 16 == 0 else 1
nn.GroupNorm(groups, n_out)
```

**设计考量**：当特征维度为 128 时，使用 16 组；其他情况自动调整分组数，避免出现`num_groups=1`导致 ONNX 兼容问题

---

### 六、完整数据流维度变化

1. **输入数据**：

   - Actor 轨迹：`[batch_size, num_actors, 20, 3]` （20 帧历史轨迹）
   - 地图数据：`[batch_size, num_nodes, 2]` （车道中心点）

2. **处理过程**：

   ```
   ActorNet: [N_actor, 3, 20] → [N_actor, 128]
   MapNet: [N_node, 2] → [N_node, 128]
   A2M/M2M/M2A/A2A: 保持128维特征
   PredNet: [N_actor, 128] → [N_actor, 6, 30, 2]
   ```

3. **输出**：
   - 多模态轨迹预测：`6`种可能轨迹，每种预测`30`个时间步的`(x,y)`坐标

通过这种层级化设计，LaneGCN 实现了高效的车辆-地图交互建模，最终输出符合人类驾驶逻辑的多模态轨迹预测。

# 44

以下是为 LaneGCN 各模块 forward 函数添加的逐层输入输出注释（以关键模块为例）：

### 1. ActorNet（车辆轨迹特征提取）

```python
class ActorNet(nn.Module):
    def forward(self, actors: Tensor) -> Tensor:
        """
        输入:
            actors: 车辆轨迹特征
            - 形状: [num_actors, 3, 20]
            - 含义: [总车辆数, 特征维度(x,y,t), 历史帧数]

        处理流程:
            1. 转置输入: [num_actors, 3, 20] → [num_actors, 20, 3]
            2. 通过3层Res1d残差卷积:
                - 第1层: [num_actors, 32, 20] (保持长度)
                - 第2层: [num_actors, 64, 10] (stride=2下采样)
                - 第3层: [num_actors, 128, 5] (再次下采样)
            3. 特征融合:
                - 上采样64层特征 → [128, 10] 与第2层融合
                - 再次上采样 → [128, 20] 与第1层融合

        输出:
            Tensor - 提取的车辆特征
            - 形状: [num_actors, 128]
            - 维度说明: [总车辆数, 特征维度]
        """
        out = actors.transpose(1, 2)  # [N,3,20] → [N,20,3]
        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)  # 各层输出形状变化
            outputs.append(out)
        # 特征融合过程...
        return out[:, :, -1]  # 取最后时间步
```

### 2. MapNet（地图特征提取）

```python
class MapNet(nn.Module):
    def forward(self, graph):
        """
        输入:
            graph: 地图图结构数据
            - ctrs: [num_nodes, 2] (车道中心点坐标)
            - feats: [num_nodes, 2] (方向向量)
            - 邻接关系: pre/suc/left/right等连接信息

        处理流程:
            1. 坐标编码: [num_nodes, 2] → [num_nodes, 128]
            2. 方向编码: [num_nodes, 2] → [num_nodes, 128]
            3. 特征融合: 坐标+方向特征相加 → [num_nodes, 128]
            4. 多尺度图卷积:
                - 通过6个尺度(pre0~pre5, suc0~suc5)聚合邻域信息
                - 每个尺度使用GroupNorm+Linear处理

        输出:
            Tuple[Tensor, List[Tensor], List[Tensor]]:
                - nodes: 车道节点特征 [num_nodes, 128]
                - node_idcs: 各批次节点索引列表
                - node_ctrs: 各批次节点中心坐标列表
        """
        # 具体实现...
```

### 3. A2M（Actor 到 Map 融合）

```python
class A2M(nn.Module):
    def forward(self, nodes, graph, actors, actor_idcs, actor_ctrs):
        """
        输入:
            nodes: 车道节点特征 [num_nodes, 128]
            actors: 车辆特征 [num_actors, 128]
            actor_ctrs: 车辆中心坐标 [num_actors, 2]

        处理流程:
            1. 计算车辆与车道的空间关系（距离过滤）
            2. 注意力机制计算：
                - 线性变换生成query/key
                - 空间位置编码（相对坐标）
                - 特征拼接与融合
            3. 使用index_add聚合特征到车道节点

        输出:
            Tensor - 更新后的车道节点特征
            - 形状: [num_nodes, 128]
            - 特征包含实时交通信息
        """
        # 具体实现...
```

### 4. M2A（Map 到 Actor 融合）

```python
class M2A(nn.Module):
    def forward(self, actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs):
        """
        输入:
            actors: 车辆特征 [num_actors, 128]
            nodes: 车道特征 [num_nodes, 128]

        处理流程:
            1. 逆向注意力机制（车道→车辆）
            2. 计算车道对车辆的影响权重
            3. 特征加权聚合

        输出:
            Tensor - 更新后的车辆特征
            - 形状: [num_actors, 128]
            - 特征包含地图拓扑信息
        """
        # 具体实现...
```

### 5. PredNet（轨迹预测头）

```python
class PredNet(nn.Module):
    def forward(self, actors, actor_idcs, actor_ctrs):
        """
        输入:
            actors: 最终车辆特征 [num_actors, 128]

        处理流程:
            1. 通过6个并行预测头生成多模态轨迹
                - 每个模态: LinearRes → Linear
            2. 轨迹坐标转换（局部→全局）

        输出:
            Dict[str, List[Tensor]]:
                - reg: 预测轨迹列表
                    - 每个元素形状: [batch_actors, 6, 30, 2]
                    - 含义: [当前批次车辆数, 模态数, 预测步长, 坐标]
                - cls: 各模态置信度（可选）
        """
        # 具体实现...
```

### 6. 注意力模块（Att）

```python
class Att(nn.Module):
    def forward(self, agts, agt_idcs, agt_ctrs, ctx, ctx_idcs, ctx_ctrs, dist_th):
        """
        输入:
            agts: 目标节点特征 [num_agts, D]
            ctx: 上下文节点特征 [num_ctx, D]
            *_ctrs: 各节点坐标

        处理流程:
            1. 空间关系编码（相对坐标→高维）
            2. Query-Key乘积计算注意力权重
            3. 加权聚合上下文特征

        输出:
            Tensor - 更新后的目标节点特征
            - 形状: [num_agts, D]
            - 特征包含上下文信息
        """
        # 具体实现...
```

### 关键数据维度变化表

| 模块     | 输入形状        | 输出形状      | 说明               |
| -------- | --------------- | ------------- | ------------------ |
| ActorNet | [N, 3, 20]      | [N, 128]      | 轨迹时序特征提取   |
| MapNet   | [M, 2]          | [M, 128]      | 车道拓扑特征提取   |
| A2M      | [M,128]+[N,128] | [M, 128]      | 车辆信息融合到车道 |
| M2A      | [N,128]+[M,128] | [N, 128]      | 车道信息融合到车辆 |
| PredNet  | [N, 128]        | [B, 6, 30, 2] | 生成多模态轨迹预测 |

注：

- N：总车辆数（跨批次）
- M：总车道节点数
- B：批次车辆数
- 6：预测模态数
- 30：预测时间步数

# M# 
