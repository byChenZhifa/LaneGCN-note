# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from datetime import datetime
from importlib import import_module
import numpy as np
import os
import sys

# from fractions import gcd
from math import gcd
from numbers import Number

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# from data import ArgoDataset, collate_fn
from utils import gpu, to_long, StepLR

# from utils import gpu, to_long, Optimizer, StepLR

from layers import Conv1d, Res1d, Linear, LinearRes, Null
from numpy import float64, ndarray
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

file_path = os.path.abspath(__file__)  # 获取当前文件的绝对路径
root_path = os.path.dirname(file_path)  # 获取当前文件所在目录
model_name = os.path.basename(file_path).split(".")[0]  # 获取模型名称（文件名去掉扩展名）

### config ###
config = dict()
"""Train"""
config["display_iters"] = 205942  # 显示迭代间隔
config["val_iters"] = 205942 * 2  # 验证迭代间隔
config["save_freq"] = 1.0  # 保存频率
config["epoch"] = 0  # 当前训练轮次
config["horovod"] = True  # 是否使用Horovod分布式训练
config["opt"] = "adam"  # 优化器类型
config["num_epochs"] = 36  # 总训练轮次
config["lr"] = [1e-3, 1e-4]  # 学习率列表
config["lr_epochs"] = [32]  # 学习率调整的轮次
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])  # 学习率调度器


if "save_dir" not in config:
    config["save_dir"] = os.path.join(root_path, "results", model_name)  # 设置默认保存目录

if not os.path.isabs(config["save_dir"]):
    config["save_dir"] = os.path.join(root_path, "results", config["save_dir"])  # 确保保存目录为绝对路径

config["batch_size"] = 32  # 训练批次大小
config["val_batch_size"] = 32  # 验证批次大小
config["workers"] = 0  # 数据加载线程数
config["val_workers"] = config["workers"]  # 验证数据加载线程数


"""Dataset"""
# Raw Dataset
config["train_split"] = os.path.join(root_path, "dataset/train/data")  # 训练数据路径
config["val_split"] = os.path.join(root_path, "dataset/val/data")  # 验证数据路径
config["test_split"] = os.path.join(root_path, "dataset/test_obs/data")  # 测试数据路径

# Preprocessed Dataset
config["preprocess"] = True  # 是否使用预处理数据
config["preprocess_train"] = os.path.join(root_path, "dataset", "preprocess", "train_crs_dist6_angle90.p")  # 预处理训练数据路径
config["preprocess_val"] = os.path.join(root_path, "dataset", "preprocess", "val_crs_dist6_angle90.p")  # 预处理验证数据路径
config["preprocess_test"] = os.path.join(root_path, "dataset", "preprocess", "test_test.p")  # 预处理测试数据路径

"""Model"""
config["rot_aug"] = False  # 是否使用旋转增强
config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]  # 预测范围
config["num_scales"] = 6  # 尺度数量
config["n_actor"] = 128  # Actor特征维度
config["n_map"] = 128  # 地图特征维度
config["actor2map_dist"] = 7.0  # Actor到地图的最大距离
config["map2actor_dist"] = 6.0  # 地图到Actor的最大距离
config["actor2actor_dist"] = 100.0  # Actor之间的最大距离
config["pred_size"] = 30  # 预测时间步长
config["pred_step"] = 1  # 预测步长
config["num_preds"] = config["pred_size"] // config["pred_step"]  # 预测点数量
config["num_mods"] = 6  # 预测模式数量
config["cls_coef"] = 1.0  # 分类损失系数
config["reg_coef"] = 1.0  # 回归损失系数
config["mgn"] = 0.2  # 边缘值
config["cls_th"] = 2.0  # 分类阈值
config["cls_ignore"] = 0.2  # 分类忽略阈值
### end of config ###


class Net(nn.Module):
    """
        Lane Graph Network contains following components:
        1. ActorNet: a 1D CNN to process the trajectory input
        2. MapNet: LaneGraphCNN to learn structured map representations from vectorized map data
        3. Actor-Map Fusion Cycle: fuse the information between actor nodes and lane nodes:
            a. A2M: introduces real-time traffic information to lane nodes, such as blockage or usage of the lanes
            b. M2M:  updates lane node features by propagating the traffic information over lane graphs
            c. M2A: fuses updated map features with real-time traffic information back to actors
            d. A2A: handles the interaction between actors and produces the output actor features
        4. PredNet: prediction header for motion forecasting using feature from A2A

    Lane Graph Network 包含以下组件：
        1. ActorNet：一个 1D CNN，用于处理轨迹输入
        2. MapNet：LaneGraphCNN 从矢量化地图数据中学习结构化地图表示
        3. Actor-Map 融合循环：将 Actor 节点和 lane 节点之间的信息融合：
            a. A2M：将实时交通信息引入车道节点，例如车道的阻塞或使用
            b. M2M：通过在车道图上传播交通信息来更新车道节点特征
            c. M2A：将更新的地图功能与实时交通信息融合在一起，返回给参与者
            d. A2A：处理 Actor 之间的交互并生成输出 Actor 特征
        4. PredNet：使用 A2A 功能进行运动预测的预测标头

    LaneGCN 轨迹预测模型：
        核心功能：通过融合车辆轨迹特征与高精度地图特征，实现多模态轨迹预测
        主要组成部分：
        1. ActorNet - 车辆轨迹特征提取
        2. MapNet - 地图拓扑结构特征提取
        3. 融合模块（A2M/M2M/M2A/A2A）- 车路交互建模
        4. PredNet - 多模态轨迹预测
    """

    def __init__(self, config):
        """
        初始化 Lane Graph Network。
        :param config: 配置字典，包含模型参数。
        """
        super(Net, self).__init__()
        self.config = config

        self.actor_net = ActorNet(config)  # 初始化 ActorNet
        self.map_net = MapNet(config)  # 初始化 MapNet

        self.a2m = A2M(config)  # 初始化 Actor 到 Map 的融合模块
        self.m2m = M2M(config)  # 初始化 Map 到 Map 的融合模块
        self.m2a = M2A(config)  # 初始化 Map 到 Actor 的融合模块
        self.a2a = A2A(config)  # 初始化 Actor 到 Actor 的融合模块

        self.pred_net = PredNet(config)  # 初始化预测模块

    def forward222222(self, data: Dict) -> Dict[str, List[Tensor]]:
        """
        前向传播函数。
        :param data: 输入数据字典，包含轨迹、地图等信息。
        :return: 输出字典，包含分类和回归结果。
        """
        # 构造 Actor 特征
        actors, actor_idcs = actor_gather(gpu(data["feats"]))  # 聚合批次车辆特征 [total_actors, seq_len, 2]
        actor_ctrs = gpu(data["ctrs"])  # 车辆中心坐标 [total_actors, 2]
        actors = self.actor_net(actors)  # 提取车辆特征 [total_actors, n_actor]

        # 构造地图特征
        graph = graph_gather(to_long(gpu(data["graph"])))  # 聚合地图图结构
        nodes, node_idcs, node_ctrs = self.map_net(graph)  # 提取地图特征 [total_nodes, n_map]

        # Actor-Map 融合循环
        nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)  # Actor 到 Map 融合
        nodes = self.m2m(nodes, graph)  # Map 到 Map 融合
        actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)  # Map 到 Actor 融合
        actors = self.a2a(actors, actor_idcs, actor_ctrs)  # Actor 到 Actor 融合

        # 预测
        out = self.pred_net(actors, actor_idcs, actor_ctrs)  # 预测未来轨迹
        rot, orig = gpu(data["rot"]), gpu(data["orig"])  # 获取旋转矩阵和原点
        # 将预测结果转换到世界坐标系
        for i in range(len(out["reg"])):
            out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(1, 1, 1, -1)
        return out

    # "feats", "ctrs", "graph", "rot", "orig"
    def forward(self, feats, ctrs, graph, rot, orig) -> Dict[str, List[Tensor]]:
        """
        前向传播函数。
        :param data: 输入数据字典，包含轨迹、地图等信息。
        :return: 输出字典，包含分类和回归结果。
        """
        # 构造 Actor 特征
        actors, actor_idcs = actor_gather(feats)  # 聚合批次车辆特征 [total_actors, seq_len, 2]
        actor_ctrs = ctrs  # 车辆中心坐标 [total_actors, 2]
        actors = self.actor_net(actors)  # 提取车辆特征 [total_actors, n_actor]

        # 构造地图特征
        graph = graph_gather(to_long(graph))  # 聚合地图图结构
        nodes, node_idcs, node_ctrs = self.map_net(graph)  # 提取地图特征 [total_nodes, n_map]

        # Actor-Map 融合循环
        nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)  # Actor 到 Map 融合
        nodes = self.m2m(nodes, graph)  # Map 到 Map 融合
        actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)  # Map 到 Actor 融合
        actors = self.a2a(actors, actor_idcs, actor_ctrs)  # Actor 到 Actor 融合

        # 预测
        out = self.pred_net(actors, actor_idcs, actor_ctrs)  # 预测未来轨迹
        # rot, orig = gpu(data["rot"]), gpu(data["orig"])  # 获取旋转矩阵和原点
        # 将预测结果转换到世界坐标系
        # 修改前
        # for i in range(len(out["reg"])):
        #     out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(1, 1, 1, -1)

        # 修改后（假设rot/orig是单批次输入）
        for i in range(len(out["reg"])):
            # 使用索引0获取当前批次的旋转矩阵和原点
            out["reg"][i] = torch.matmul(out["reg"][i], rot[0]) + orig[0].view(1, 1, 1, -1)
        return out


def actor_gather(actors: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    """
    收集 Actor 特征。
    :param actors: 每个批次的 Actor 特征列表。
    :return: 合并的 Actor 特征张量和索引列表。
    """
    batch_size = len(actors)  # 批次大小
    num_actors = [len(x) for x in actors]  # 每个批次中的 Actor 数量

    actors = [x.transpose(1, 2) for x in actors]  # 转置特征维度
    actors = torch.cat(actors, 0)  # 合并所有批次的 Actor 特征

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)  # 生成索引
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs


def graph_gather(graphs):
    """
    收集地图图结构。
    :param graphs: 每个批次的地图图结构列表。
    :return: 合并的地图图结构。
    """
    batch_size = len(graphs)  # 批次大小
    node_idcs = []
    count = 0
    counts = []
    for i in range(batch_size):  # 遍历批次   batch_size = 10
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"]).to(graphs[i]["feats"][0].device)  # 生成节点索引
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["idcs"] = node_idcs
    # 修改前
    # graph["ctrs"] = [x["ctrs"] for x in graphs]
    # 修改后（直接拼接张量）
    graph["ctrs"] = torch.cat([x["ctrs"] for x in graphs], 0)

    for key in ["feats", "turn", "control", "intersect"]:
        graph[key] = torch.cat([x[key] for x in graphs], 0)  # 合并特征

    # 添加必要字段的默认值
    graph["num_nodes"] = sum(x["num_nodes"] for x in graphs)

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                graph[k1][i][k2] = torch.cat([graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0)

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0) for x in temp]
            graph[k1][k2] = torch.cat(temp)
    return graph


class ActorNet(nn.Module):
    """
    Actor 特征提取器，使用 1D 卷积网络。
    """

    def __init__(self, config):
        """
        初始化 ActorNet。
        :param config: 配置字典，包含模型参数。
        """
        super(ActorNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_in = 3
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        ######! 多层(3)残差卷积块 #####
        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        ######! 特征融合层 #####
        n = config["n_actor"]
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.output = Res1d(n, n, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        """
        前向传播函数。
        :param actors: 输入的 Actor 特征张量。
        :return: 提取后的 Actor 特征张量。
        """
        out = actors

        outputs = []
        # 通过各卷积层
        for i in range(len(self.groups)):
            out = self.groups[i](out)  # 通过残差卷积块
            outputs.append(out)

        # 特征融合
        out = self.lateral[-1](outputs[-1])  # 最后一层特征
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)  # 上采样
            out += self.lateral[i](outputs[i])  # 融合特征

        out = self.output(out)[:, :, -1]  # 输出最后一维特征  [num_actors, n_actor]
        return out


class MapNet(nn.Module):
    """
    Map Graph 特征提取器，使用 LaneGraphCNN。
    """

    def __init__(self, config):
        """
        初始化 MapNet。
        :param config: 配置字典，包含模型参数。
        """
        super(MapNet, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        self.input = nn.Sequential(
            nn.Linear(2, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )
        self.seg = nn.Sequential(nn.Linear(2, n_map), nn.ReLU(inplace=True), Linear(n_map, n_map, norm=norm, ng=ng, act=False))

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    # 修改前（lanegcn.py MapNet部分）
                    # fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                    # 修改后（显式设置num_groups=1）
                    fuse[key].append(nn.GroupNorm(1, n_map))  # 强制分组数为1
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, graph):
        """
        前向传播函数。
        :param graph: 输入的地图图结构。
        :return: 提取后的地图特征张量、节点索引和中心点。
        """
        if len(graph["feats"]) == 0 or len(graph["pre"][-1]["u"]) == 0 or len(graph["suc"][-1]["u"]) == 0:
            temp = graph["feats"]
            return (
                temp.new().resize_(0),
                [temp.new().long().resize_(0) for x in graph["node_idcs"]],
                temp.new().resize_(0),
            )

        # 修改前
        # ctrs = torch.cat(graph["ctrs"], 0)
        # 修改后（直接使用已拼接的张量）
        ctrs = graph["ctrs"]
        # ctrs = graph["ctrs"][0] # 直接使用张量，无需拼接
        # ctrs= tensor([[-0.6442, -1.4949],
        # [-0.7099, -0.2203],
        # [-0.7556, -0.3917],
        # [-0.2461, -1.0954],
        # [-1.2916, -0.1463],
        # [-0.2244, -1.4459],
        # [ 2.3021, -0.4143],
        # [-1.3379,  1.4924],
        # [ 0.0618,  1.5568],
        # [-0.3918,  0.9325],
        # [ 0.7887,  0.9624]], device='cuda:0')
        feat = self.input(ctrs)
        feat += self.seg(graph["feats"])
        feat = self.relu(feat)

        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat, graph["idcs"], graph["ctrs"]


class A2M(nn.Module):
    """
    Actor to Map Fusion:  fuses real-time traffic information from
    actor nodes to lane nodes
    """

    def __init__(self, config):
        super(A2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        """fuse meta, static, dyn"""
        self.meta = Linear(n_map + 4, n_map, norm=norm, ng=ng)
        att = []
        for i in range(2):
            att.append(Att(n_map, config["n_actor"]))
        self.att = nn.ModuleList(att)

    def forward(
        self,
        feat: Tensor,
        graph: Dict[str, Union[List[Tensor], Tensor, List[Dict[str, Tensor]], Dict[str, Tensor]]],
        actors: Tensor,
        actor_idcs: List[Tensor],
        actor_ctrs: List[Tensor],
    ) -> Tensor:
        """meta, static and dyn fuse using attention"""
        meta = torch.cat(
            (
                graph["turn"],
                graph["control"].unsqueeze(1),
                graph["intersect"].unsqueeze(1),
            ),
            1,
        )
        feat = self.meta(torch.cat((feat, meta), 1))

        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                graph["idcs"],
                graph["ctrs"],
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2map_dist"],
            )
        return feat


class M2M(nn.Module):
    """
    The lane to lane block: propagates information over lane
            graphs and updates the features of lane nodes
    """

    def __init__(self, config):
        super(M2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    # 修改前（lanegcn.py MapNet部分）
                    # fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                    # 修改后（显式设置num_groups=1）
                    fuse[key].append(nn.GroupNorm(1, n_map))  # 强制分组数为1
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat: Tensor, graph: Dict) -> Tensor:
        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat


class M2A(nn.Module):
    """
    The lane to actor block fuses updated
        map information from lane nodes to actor nodes
    """

    def __init__(self, config):
        super(M2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_map))
        self.att = nn.ModuleList(att)

    def forward(
        self,
        actors: Tensor,
        actor_idcs: List[Tensor],
        actor_ctrs: List[Tensor],
        nodes: Tensor,
        node_idcs: List[Tensor],
        node_ctrs: List[Tensor],
    ) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                nodes,
                node_idcs,
                node_ctrs,
                self.config["map2actor_dist"],
            )
        return actors


class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """

    def __init__(self, config):
        super(A2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_actor))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2actor_dist"],
            )
        return actors


class EncodeDist(nn.Module):
    def __init__(self, n, linear=True):
        super(EncodeDist, self).__init__()
        norm = "GN"
        ng = 1

        block = [nn.Linear(2, n), nn.ReLU(inplace=True)]

        if linear:
            block.append(nn.Linear(n, n))

        self.block = nn.Sequential(*block)

    def forward(self, dist):
        x, y = dist[:, :1], dist[:, 1:]
        dist = torch.cat(
            (
                torch.sign(x) * torch.log(torch.abs(x) + 1.0),
                torch.sign(y) * torch.log(torch.abs(y) + 1.0),
            ),
            1,
        )

        dist = self.block(dist)
        return dist


class PredNet(nn.Module):
    """
    Final motion forecasting with Linear Residual block
    """

    def __init__(self, config):
        super(PredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1))

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Dict[str, List[Tensor]]:
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
            reg[idcs] = reg[idcs] + ctrs

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.config["num_mods"])

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        # out["cls"], out["reg"] = [], []
        out["reg"] = []
        # for i in range(len(actor_idcs)):
        #     idcs = actor_idcs[i]
        #     ctrs = actor_ctrs[i].view(-1, 1, 1, 2)
        #     out["cls"].append(cls[idcs])
        #     out["reg"].append(reg[idcs])

        # 修改后（确保输出与输入批次一致）
        batch_size = len(actor_idcs)
        for i in range(batch_size):
            idcs = actor_idcs[i]
            out["reg"].append(reg[idcs])

        return out


class Att(nn.Module):
    """
    Attention block to pass context nodes information to target nodes
    This is used in Actor2Map, Actor2Actor, Map2Actor and Map2Map

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

    def __init__(self, n_agt: int, n_ctx: int) -> None:
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        # 空间编码层
        self.dist = nn.Sequential(
            nn.Linear(2, n_ctx),  # 将相对坐标映射到高维
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        # 注意力计算层
        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)  # 将目标特征转换为query

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        agts: Tensor,
        agt_idcs: List[Tensor],
        agt_ctrs: List[Tensor],
        ctx: Tensor,
        ctx_idcs: List[Tensor],
        ctx_ctrs: List[Tensor],
        dist_th: float,
    ) -> Tensor:
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        # 计算节点间距离
        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0
        for i in range(batch_size):
            dist = agt_ctrs[i].view(-1, 1, 2) - ctx_ctrs[i].view(1, -1, 2)
            dist = torch.sqrt((dist**2).sum(2))
            mask = dist <= dist_th  # 基于距离阈值筛选相邻节点

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])
        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        # agt_ctrs = torch.cat(agt_ctrs, 0)
        # ctx_ctrs = torch.cat(ctx_ctrs, 0)
        # 修改后：确保输入是列表形式再拼接
        agt_ctrs = torch.cat(agt_ctrs, dim=0) if isinstance(agt_ctrs, (list, tuple)) else agt_ctrs
        ctx_ctrs = torch.cat(ctx_ctrs, dim=0) if isinstance(ctx_ctrs, (list, tuple)) else ctx_ctrs

        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        query = self.query(agts[hi])

        ctx = ctx[wi]
        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)
        return agts


class AttDest(nn.Module):
    def __init__(self, n_agt: int):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts


class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(
        self,
        out: Dict[str, List[Tensor]],
        gt_preds: List[Tensor],
        has_preds: List[Tensor],
    ) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        gt_preds = torch.cat([x for x in gt_preds], 0)
        has_preds = torch.cat([x for x in has_preds], 0)

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(has_preds.device) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(torch.sqrt(((reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs]) ** 2).sum(1)))
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] += coef * (self.config["mgn"] * mask.sum() - mgn[mask].sum())
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] += coef * self.reg_loss(reg[has_preds], gt_preds[has_preds])
        loss_out["num_reg"] += has_preds.sum().item()
        return loss_out


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), gpu(data["has_preds"]))
        loss_out["loss"] = loss_out["cls_loss"] / (loss_out["num_cls"] + 1e-10) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out, data):
        post_out = dict()
        post_out["preds"] = [x[0:1].detach().cpu().numpy() for x in out["reg"]]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1].numpy() for x in data["has_preds"]]
        return post_out

    def append(
        self,
        metrics: Dict,
        loss_out: Dict,
        post_out: Optional[Dict[str, List[ndarray]]] = None,
    ) -> Dict:
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]
        return metrics

    def display(self, metrics, dt, epoch, lr=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print("************************* Validation, time %3.2f *************************" % dt)

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        print("loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade %2.4f, fde %2.4f" % (loss, cls, reg, ade1, fde1, ade, fde))
        print()


def pred_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))

    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err = err[row_idcs, min_idcs]
    ade = err.mean()
    fde = err[:, -1].mean()
    return ade1, fde1, ade, fde, min_idcs


def get_model():
    net = Net(config)
    net = net.cuda()

    # loss = Loss(config).cuda()
    # post_process = PostProcess(config).cuda()

    # params = net.parameters()
    # opt = Optimizer(params, config)

    # return config, ArgoDataset, collate_fn, net, loss, post_process, opt
    return net

    # def export_onnx(net: Net, device, dir_cache="./cache", data_in=None):
    #     os.makedirs(dir_cache, exist_ok=True)

    #     # 创建符合网络结构  dumy

    #     with torch.no_grad():
    #         test_output = net(
    #             actors=dummy_actors,
    #             actor_idcs=dummy_actor_idcs,
    #             lanes=dummy_maps,
    #             lane_idcs=dummy_map_idcs,
    #             rpe=dummy_rpe,
    #         )
    #         print("验证输出维度:", [t.shape for t in test_output[0][0]])

    #     # 导出ONNX模型
    #     time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    #     onnx_path = os.path.join(dir_cache, f"net_mtp_{time_str}.onnx")

    #     torch.onnx.export(
    #         net,
    #         (dummy_actors, dummy_actor_idcs, dummy_maps, dummy_map_idcs, dummy_rpe),
    #         onnx_path,
    #         input_names=["actors", "actor_idcs", "lanes", "lane_idcs", "rpe"],
    #         output_names=["cls", "reg", "aux"],
    #         dynamic_axes={
    #             "actors": {0: "num_actors"},
    #             "lanes": {0: "num_lanes"},
    #             "cls": {0: "batch_size"},
    #             "reg": {0: "batch_size"},
    #         },
    #         opset_version=14,
    #         do_constant_folding=True,
    #     )
    # print(f"ONNX模型已保存至: {onnx_path}")


def export_onnx(net: Net, device, output_path="lanegcn.onnx"):
    """
    Export LaneGCN model to ONNX format
    实现要点：
    1. 构造符合网络输入的虚拟数据
    2. 处理动态维度（如可变数量的actors/nodes）
    3. 保留必要特征处理逻辑
    """

    # 构造虚拟输入数据
    batch_size = 1
    num_actors = 8  # 示例车辆数
    seq_len = 20  # 轨迹序列长度
    num_nodes = 11  # 示例地图节点数

    # dummy_data = {
    #     "feats": [torch.randn(num_actors, seq_len, 3, device=device.type)],
    #     "ctrs": [torch.randn(num_actors, 2, device=device.type)],
    #     "graph": [
    #         {
    #             "ctrs": torch.randn(num_nodes, 2, device=device.type),
    #             "feats": torch.randn(num_nodes, 2, device=device.type),
    #             "turn": torch.randn(num_nodes, 2, device=device.type),
    #             "control": torch.randint(0, 2, (num_nodes,), device=device.type),
    #             "intersect": torch.randint(0, 2, (num_nodes,), device=device.type),
    #             "pre": [
    #                 {"u": torch.tensor([0], device=device.type), "v": torch.tensor([1], device=device.type)},
    #             ],
    #             "suc": [{"u": torch.tensor([1], device=device.type), "v": torch.tensor([0], device=device.type)}],
    #             "left": {"u": torch.tensor([], device=device.type), "v": torch.tensor([], device=device.type)},
    #             "right": {"u": torch.tensor([], device=device.type), "v": torch.tensor([], device=device.type)},
    #             "num_nodes": num_nodes,
    #         }
    #     ],
    #     "rot": torch.eye(2, device=device.type).unsqueeze(0).repeat(batch_size, 1, 1),
    #     "orig": torch.zeros(batch_size, 2, device=device.type),
    # }
    dummy_data = {
        "feats": [torch.randn(num_actors, seq_len, 3, device=device)],
        "ctrs": [torch.randn(num_actors, 2, device=device)],
        "graph": [
            {
                "ctrs": torch.randn(num_nodes, 2, device=device),
                "feats": torch.randn(num_nodes, 2, device=device),
                "turn": torch.randn(num_nodes, 2, device=device),
                "control": torch.randint(0, 2, (num_nodes,), device=device),
                "intersect": torch.randint(0, 2, (num_nodes,), device=device),
                "pre": [
                    {"u": torch.tensor([0], device=device), "v": torch.tensor([1], device=device)},
                    {"u": torch.tensor([0], device=device), "v": torch.tensor([1], device=device)},
                    {"u": torch.tensor([0], device=device), "v": torch.tensor([1], device=device)},
                    {"u": torch.tensor([0], device=device), "v": torch.tensor([1], device=device)},
                    {"u": torch.tensor([0], device=device), "v": torch.tensor([1], device=device)},
                    {"u": torch.tensor([0], device=device), "v": torch.tensor([1], device=device)},
                    # {"u": torch.tensor([0], device=device), "v": torch.tensor([1], device=device)},
                    # {"u": torch.tensor([0], device=device), "v": torch.tensor([1], device=device)},
                    # {"u": torch.tensor([0], device=device), "v": torch.tensor([1], device=device)},
                    # {"u": torch.tensor([0], device=device), "v": torch.tensor([1], device=device)},
                    # {"u": torch.tensor([0], device=device), "v": torch.tensor([1], device=device)},
                    # {"u": torch.tensor([0], device=device), "v": torch.tensor([1], device=device)},
                ],
                "suc": [
                    {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                    {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                    {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                    {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                    {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                    {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                    {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                    # {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                    # {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                    # {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                    # {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                    # {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                    # {"u": torch.tensor([1], device=device), "v": torch.tensor([0], device=device)},
                ],
                "left": {"u": torch.empty(0, dtype=torch.long), "v": torch.empty(0, dtype=torch.long)},
                "right": {"u": torch.empty(0, dtype=torch.long), "v": torch.empty(0, dtype=torch.long)},
                # "left": {"u": torch.tensor([], device=device), "v": torch.tensor([], device=device)},
                # "right": {"u": torch.tensor([], device=device), "v": torch.tensor([], device=device)},
                "num_nodes": num_nodes,
            }
        ],
        # "rot": [torch.eye(2, device=device).unsqueeze(0).repeat(batch_size, 1, 1)],
        # "orig": [torch.zeros(batch_size, 2, device=device)],
        "rot": [torch.eye(2, device=device)],  # 单批次旋转矩阵
        "orig": [torch.zeros(2, device=device)],  # 单批次原点
    }

    # 添加维度验证
    # assert dummy_data["graph"][0]["ctrs"].dim() == 2, "ctrs应为二维坐标"
    assert dummy_data["graph"][0]["feats"].size(1) == 2, "特征维度应为2"
    # 添加维度验证
    assert len(dummy_data["rot"]) == 1, "rot应包含单批次数据"
    assert len(dummy_data["orig"]) == 1, "orig应包含单批次数据"

    # 设备一致性验证    # 修改后的设备检查函数
    def check_device(data):
        if isinstance(data, torch.Tensor):
            # 仅检查设备类型（cpu/cuda）而非具体索引
            assert data.device.type == device.type, f"设备类型不匹配！当前:{data.device}, 要求:{device}"
        elif isinstance(data, dict):
            for v in data.values():
                check_device(v)
        elif isinstance(data, list):
            for item in data:
                check_device(item)

    print("正在验证设备一致性...")
    check_device(dummy_data)
    # for name, param in net.named_parameters():
    #     assert param.device == device, f"模型参数 {name} 在 {param.device} 设备上"

    # 运行前向传播验证
    net = net.to(device)

    with torch.no_grad():
        net.eval()
        out = net(dummy_data["feats"], dummy_data["ctrs"], dummy_data["graph"], dummy_data["rot"], dummy_data["orig"])
        assert all(t.ndim == 4 for t in out["reg"]), "输出维度异常"

        print("ONNX导出前向传播验证成功！")

    # 确保模型所有参数在目标设备上
    # device = torch.device("cpu")

    # 导出ONNX模型
    net.eval()
    torch.onnx.export(
        net,
        (dummy_data["feats"], dummy_data["ctrs"], dummy_data["graph"], dummy_data["rot"], dummy_data["orig"]),
        output_path,
        input_names=["feats", "ctrs", "graph", "rot", "orig"],
        output_names=["reg"],  # 只保留必要输出
        dynamic_axes={
            "feats": {0: "num_actors"},
            "ctrs": {0: "num_actors"},
            "graph": {"num_nodes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        },
        opset_version=14,
        do_constant_folding=True,
        custom_opsets={"": 14},  # 显式指定opset
    )
    print(f"Successfully exported ONNX model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
    parser.add_argument("-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", default="", type=str, metavar="RESUME", help="checkpoint path")
    parser.add_argument("--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path")
    # 添加导出命令
    parser.add_argument(
        "--export-onnx",
        default="/home/czf/project_czf/20241130-mine_prediction_MTP/LaneGCN-note/cache",
        type=str,
        help="Path to export ONNX model",
    )

    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    # config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model()
    net = model.get_model()

    # 添加导出逻辑
    if args.export_onnx:
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        net.to(device)
        export_onnx(net, device, args.export_onnx)
        sys.exit(0)  # 导出后退出
