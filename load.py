import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from tqdm import tqdm




class DiabetesDataset(Dataset):
    def __init__(self, filepath, num=31, row=62):
        super(DiabetesDataset, self).__init__()

        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)

        lens = xy.shape[0] // row
        # 每两组31行作为一个节点

        all_x_data = []
        all_y_data = []

        for i in range(lens):

            vec = xy[i * row,    [11,  12]][np.newaxis, :]
            magnitude = np.linalg.norm(vec[0])
            angle_rad = np.arctan2(vec[0, 0], vec[0, 1]) / np.pi
            angle_rad = angle_rad.astype(np.float32)
            features = np.array([[magnitude, angle_rad]])


            x1 = xy[i * row,    [3,  4, 5,  6,  11,  12,  18,19, 20]][np.newaxis, :]
            # x1 = xy[i * 62,    [3,  4, 5,  6,   11,  12]][np.newaxis, :]
            x2 = xy[i * row + num,   [ 18,19, 20]][np.newaxis, :]

            data = torch.from_numpy(np.concatenate([x1, x2], axis=1))
            x_data = torch.from_numpy(np.concatenate((data, features), axis=1))

            # x_data = torch.from_numpy(np.concatenate([x1, x2], axis=1))
            # x_data = torch.from_numpy(x1)

            # print(x_data)



            # x1_main = xy[i * 62, [3, 4, 5, 6, 11, 12]][np.newaxis, :]
            # x1_19_20 = xy[i * 62, [19, 20]][np.newaxis, :]
            # x2_19_20 = xy[i * 62 + num, [19, 20]][np.newaxis, :]
            #
            # # 组合数据以得到期望的顺序
            # x_data = torch.from_numpy(
            #     np.concatenate([x1_main, x1_19_20[:, 0:1], x2_19_20[:, 0:1], x1_19_20[:, 1:], x2_19_20[:, 1:]], axis=1))

            # x_data = torch.tensor(x1)

            # 使用第一个列的数值作为节点的特征
            y_data = torch.tensor([xy[i * 62, 0]], dtype=torch.float)

            all_x_data.append(x_data)
            all_y_data.append(y_data)




        all_x_data = torch.cat(all_x_data, dim=0)  # stack all x_data
        all_y_data = torch.cat(all_y_data, dim=0)  # stack all y_data

        # 创建一个全连接图的边
        total_nodes = all_x_data.size(0)

        self.num_nodes = all_x_data.size(0)
        #
        # edge_index = self._compute_edges(all_x_data)
        # # # edge_index = torch.combinations(torch.arange(self.num_nodes), r=2).t().long()
        # edge_attr = self._compute_edge_attr(all_x_data,edge_index)
        #
        # mask = edge_attr >= 0.01
        # filtered_edge_index = edge_index[:, mask]
        # filtered_edge_weight = edge_attr[mask]



        filtered_edge_index,_ = self.compute_filtered_edges_and_attrs(all_x_data)
        # filtered_edge_index,_ = self.compute_filtered_edges_and_attrs_vec(all_x_data)
        #
        # self.data = Data(x=all_x_data, edge_index=edge_index, edge_attr=edge_attr, y=all_y_data)
        # self.data = Data(x=all_x_data, edge_index=filtered_edge_index, edge_attr=filtered_edge_weight, y=all_y_data)
        self.data = Data(x=all_x_data, edge_index=filtered_edge_index, edge_attr=None, y=all_y_data)

    def compute_filtered_edges_and_attrs_vec(self, x_data, k=40, alpha=0.1, threshold=0.1):
        # 计算所有节点对之间的欧氏距离
        distance_matrix = torch.cdist(x_data[:, [0, 1]], x_data[:, [0, 1]], p=2)

        # 获取最近的k个节点的索引
        _, nearest_indices = torch.topk(-distance_matrix, k=k, largest=True)

        # 创建边索引
        source_nodes = torch.arange(x_data.size(0)).view(-1, 1).repeat(1, k).view(-1)
        target_nodes = nearest_indices.view(-1)
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)

        # 向量化提取 vec2 和 vec1 的第四和第五维构成的向量
        vec1 = x_data[edge_index[0]][:, [4, 5]]
        vec2 = x_data[edge_index[1]][:, [4, 5]]

        # 归一化这些向量
        max_norm = torch.max(vec1.norm(dim=1), vec2.norm(dim=1)).unsqueeze(1)
        normalized_vec1 = vec1 / (max_norm)  # 避免除以零
        normalized_vec2 = vec2 / (max_norm)

        # 计算点积，得到向量之间的差异
        dot_products = (normalized_vec1 * normalized_vec2).sum(dim=1)
        # edge_attr = torch.exp(-alpha * dot_products)
        edge_attr = dot_products

        # 过滤边
        mask = edge_attr >= threshold
        filtered_edge_index = edge_index[:, mask]
        filtered_edge_attr = edge_attr[mask]

        return filtered_edge_index, filtered_edge_attr

    def compute_filtered_edges_and_attrs(self , x_data, k=30, alpha=0.1, threshold=0.01):
        # 计算所有节点对之间的欧氏距离
        distance_matrix = torch.cdist(x_data[:, [0, 1]], x_data[:, [0, 1]], p=2)

        # 获取最近的k个节点的索引
        _, nearest_indices = torch.topk(-distance_matrix, k=k, largest=True)

        # 创建边索引
        source_nodes = torch.arange(x_data.size(0)).view(-1, 1).repeat(1, k).view(-1)
        target_nodes = nearest_indices.view(-1)
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)

        # 向量化计算边属性
        vec1 = x_data[edge_index[0]]
        vec2 = x_data[edge_index[1]]

        # 计算差异
        # diffs = (vec2[:, 4:6] - vec1[:, 4:6]) / (vec2[:, 0:2] - vec1[:, 0:2])
        # print(diffs)
        # mean_of_squares = torch.mean(diffs ** 2, dim=1)

        epsilon = 1e-6
        diffs0 = (vec2[:, 4] - vec1[:, 4]) / (vec2[:, 0] - vec1[:, 0] )
        diffs1 = (vec2[:, 4] - vec1[:, 4]) / (vec2[:, 1] - vec1[:, 1] )
        diffs2 = (vec2[:, 5] - vec1[:, 5]) / (vec2[:, 0] - vec1[:, 0] )
        diffs3 = (vec2[:, 5] - vec1[:, 5]) / (vec2[:, 1] - vec1[:, 1] )
        mean_of_squares = (diffs0 ** 2 + diffs1 ** 2 + diffs2 ** 2 + diffs3 ** 2) / 4

        edge_attr = torch.exp(-alpha * mean_of_squares)

        # 过滤边
        mask = edge_attr >= threshold
        filtered_edge_index = edge_index[:, mask]
        filtered_edge_attr = edge_attr[mask]

        return filtered_edge_index, filtered_edge_attr

    def _compute_edge_attr(self, all_x_data, edge_index):
        # 计算每对节点之间的边属性
        alpha = 0.1
        edge_attr = []
        for i in range(edge_index.size(1)):
            node1 = edge_index[0, i].item()
            node2 = edge_index[1, i].item()

            # 提取两个节点的特征
            vec1 = all_x_data[node1]
            vec2 = all_x_data[node2]


            diff0 = (vec2[4] - vec1[4]) / (vec2[0] - vec1[0])
            diff1 = (vec2[4] - vec1[4]) / (vec2[1] - vec1[1])

            diff2 = (vec2[5] - vec1[5]) / (vec2[0] - vec1[0])
            diff3 = (vec2[5] - vec1[5]) / (vec2[1] - vec1[1])

            # print(diff0,diff1,diff2,diff3)

            mean_of_squares = (diff0 ** 2 + diff1 ** 2 + diff2 ** 2 + diff3 ** 2) / 4
            final_value = np.exp(-alpha * mean_of_squares)
            # print(mean_of_squares , final_value)

            edge_attr.append(final_value)
        edge_attr=edge_attr
        return torch.tensor(edge_attr, dtype=torch.float)



    def similarity(self , v1, v2):
        norm_v1 = torch.norm(v1)
        norm_v2 = torch.norm(v2)

        max_norm = max(norm_v1, norm_v2)

        # 防止除以0的情况
        if max_norm == 0:
            return 0

        normalized_v1 = v1 / max_norm
        normalized_v2 = v2 / max_norm

        return torch.dot(normalized_v1, normalized_v2)


    def similarity_norm(self , v1, v2):
        norm_v1 = torch.norm(v1)
        norm_v2 = torch.norm(v2)

        min_norm = min(norm_v1, norm_v2)
        max_norm = max(norm_v1, norm_v2)

        return min_norm/max_norm

    # def _compute_edges(self, x_data, THRESHOLD = 0.2):
    #     # 提取向量
    #     vectors = x_data[:, [0,1,2,3, 6,7]]
    #
    #     edge_indices = []
    #     edge_attrs = []
    #
    #     for i in range(self.num_nodes):
    #         for j in range(i + 1, self.num_nodes):  # 避免重复计算
    #             weight = self.similarity(vectors[i], vectors[j])
    #
    #             # 基于权重的某个阈值，决定是否添加边
    #             if weight > THRESHOLD:
    #                 edge_indices.append((i, j))
    #                 edge_attrs.append(weight)
    #
    #     edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    #     edge_attr = torch.tensor(edge_attrs, dtype=torch.float32).view(-1, 1)
    #
    #     return edge_index, edge_attr


    def _compute_edges(self, x_data, k=30):
        vectors = x_data[:, [0, 1]]

        # 计算所有节点对之间的欧氏距离
        distance_matrix = torch.cdist(vectors, vectors, p=2)

        # 获取最近的k个节点的索引
        max_distance = distance_matrix.max() + 1
        _, nearest_indices = torch.topk(max_distance - distance_matrix, k=k, largest=True)

        # 创建边索引
        source_nodes = torch.arange(vectors.size(0)).view(-1, 1).repeat(1, k)
        edge_index = torch.stack([source_nodes.view(-1), nearest_indices.view(-1)], dim=0)

        # 去除重复的边
        # 确保第一个索引总是小于第二个索引
        # sorted_edge_index, _ = torch.sort(edge_index, dim=0)
        # edge_index = sorted_edge_index[:, sorted_edge_index[0] < sorted_edge_index[1]]

        return edge_index

    def __getitem__(self, index):

        return self.data

    def __len__(self):
        return 1  # 由于每个文件只有一张图，所以长度为1

#
# file_path = './tgrs/RS/both/from0_37.csv'
# dataset = DiabetesDataset(file_path)
# print( len(dataset))

import os
from multiprocessing import Pool


def load_single_dataset(path):
    dataset = DiabetesDataset(path)
    print("read ",path)
    return dataset.data


def load_datasets(paths):
    files = [os.path.join(paths, f) for f in os.listdir(paths) if f.endswith('.csv')]
    files.sort(key=lambda x: os.path.getmtime(x))

    with Pool() as pool:
        data_list = pool.map(load_single_dataset, files)

    print(f'read {paths} ok')
    return data_list

# def load_datasets(paths):
#     files = [os.path.join(paths, f) for f in os.listdir(paths) if f.endswith('.csv')]
#     files.sort(key=lambda x: os.path.getmtime(x))
#     data_list = []
#     for filename in files:
#         data_list.append(load_single_dataset(filename))
#     return data_list

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear


import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm

class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGENet, self).__init__()

        # Define GraphSAGE convolution layers
        self.conv1 = SAGEConv(in_channels,     hidden_channels)               #aggr='lstm'
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)               #aggr='lstm'
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)               #aggr='lstm'
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)               #aggr='lstm'
        # self.conv5 = SAGEConv(hidden_channels, hidden_channels)               #aggr='lstm'
        # self.conv6 = SAGEConv(hidden_channels, hidden_channels)               #aggr='lstm'

        # Define fully connected layers
        self.fc = Linear(hidden_channels, hidden_channels//2)
        self.fc1 = Linear(hidden_channels//2, hidden_channels//2)
        self.fc2 = Linear(hidden_channels//2, out_channels)

        # Define batch normalization layers
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)

        self.bn4 = BatchNorm(hidden_channels//2)
        self.bn5 = BatchNorm(hidden_channels//2)

    def forward(self, x, edge_index, edge_weight):
        # First GraphSAGE layer
        x1 = F.relu(self.conv1(x, edge_index,edge_weight))
        x1 = self.bn1(x1)

        # Second GraphSAGE layer with residual connection
        x2 = F.relu(self.conv2(x1, edge_index,edge_weight))+ x1
        x2 = self.bn2(x2)

        # Third GraphSAGE layer with residual connection
        x3 = F.relu(self.conv3(x2, edge_index,edge_weight))+ x2
        x3 = self.bn3(x3)
        # Uncomment and add a fourth SAGE layer if desired
        x4 = F.relu(self.conv4(x3, edge_index,edge_weight)) + x3


        # x5 = F.relu(self.conv4(x3, edge_index,edge_weight))
        # x6 = F.relu(self.conv4(x3, edge_index,edge_weight))

        # Fully connected layers
        x = self.fc(x4)
        x = F.relu(x)
        x = self.bn4(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn5(x)

        x = self.fc2(x)
        # x = torch.sigmoid(x)
        return x

def evaluate(outputs, labels):
    ## for cross en
    _, preds = outputs.max(1)


    ## for bce
    # outputs = outputs.view(-1)
    # preds = (outputs > 0.5).float()

    # print(outputs,preds)

    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0)

    tp = ((preds == 1) & (labels == 1)).sum().item()  # True Positive
    fn = ((preds == 0) & (labels == 1)).sum().item()  # False Negative
    fp = ((preds == 1) & (labels == 0)).sum().item()  # False Positive
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0


    return precision, recall

# Hyperparameters



# 加载数据
file_path = './both/'
# file_path = './rrs/'
file_path = "G:/resize rotaion share/last//"
# file_path = './test/'


# load_single_dataset(file_path)


# dataset = DiabetesDataset(file_path)
# loader = DataLoader(dataset, batch_size=1, shuffle=True)


