import torch
from torch import nn as nn
from torch.nn import functional as F
import networkx as nx
import numpy as np
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SWAdjLoss(nn.Module):
    """
    Small World Model Adj loss.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(SWAdjLoss, self).__init__()
        self.loss_weight = loss_weight


    def forward(self, pred, target_clustering_coefficient=0.8, **kwargs):
        """
        Args:

        tensor -> numpy -> nx -> average_path_length + clustering_coefficient -> yueshu

        """
        path_loss = 0.0
        clus_loss = 0.0

        for adj_matrix in pred:
            num_nodes = adj_matrix.size(0)
            target_avg_path_length = torch.log2(torch.tensor(num_nodes))
            adj_tensor = adj_matrix - torch.eye(num_nodes)
            if self._is_connected(adj_tensor):
                predicted_avg_path_length = self._cal_clustering_coefficient(adj_tensor)
                predicted_clustering_coefficient = self._cal_shortest_average_path(adj_tensor)
            else:
                predicted_avg_path_length = torch.tensor(num_nodes)
                predicted_clustering_coefficient = torch.tensor(0.0)

            path_loss += F.mse_loss(predicted_avg_path_length, target_avg_path_length)
            # clus_loss += F.kl_div(predicted_clustering_coefficient, torch.tensor(target_clustering_coefficient))
            clus_loss += F.mse_loss(predicted_clustering_coefficient, torch.tensor(target_clustering_coefficient))

        path_loss /= len(pred)
        clus_loss /= len(pred)
        loss = path_loss + clus_loss

        return self.loss_weight * loss

    def _is_connected(self, graph):
        def dfs(node):
            visited[node] = True
            for neighbor in range(graph.size(0)):
                if graph[node][neighbor] == 1 and not visited[neighbor]:
                    dfs(neighbor)

        num_nodes = graph.size(0)
        visited = torch.zeros(num_nodes, dtype=torch.bool)
        dfs(0)
        return visited.all()

    def _cal_clustering_coefficient(self, adjacency_matrix):
        clustering_coefficients = []
        for i in range(adjacency_matrix.size(0)):
            try:
                neighbors = torch.nonzero(adjacency_matrix[i]).squeeze()
                num_edges = torch.tensor(0.0)
                for j in neighbors:
                    for k in neighbors:
                        if adjacency_matrix[j, k] == 1:
                            num_edges += 0.5
                possible_edges = (len(neighbors) * (len(neighbors) - 1)) / 2
                clustering_coefficients.append(num_edges / possible_edges)
            except Exception:
                clustering_coefficients.append(torch.tensor(0.0))

        average_clustering_coefficient = sum(clustering_coefficients) / len(clustering_coefficients)

        return average_clustering_coefficient


    def _cal_shortest_average_path(self, adjacency_matrix):
        shortest_paths = torch.clone(adjacency_matrix)
        shortest_paths[shortest_paths == 0] = float('inf')
        for k in range(shortest_paths.size(0)):
            for i in range(shortest_paths.size(0)):
                for j in range(shortest_paths.size(0)):
                    shortest_paths[i, j] = min(shortest_paths[i, j], shortest_paths[i, k] + shortest_paths[k, j])
        num_paths = (shortest_paths < float('inf')).sum()
        total_path_length = shortest_paths.sum()
        average_shortest_path_length = total_path_length / num_paths
        return average_shortest_path_length

@LOSS_REGISTRY.register()
class FaceStructureLoss(nn.Module):
    def __init__(self, loss_weight=1.0, win_size=3):
        super(FaceStructureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.win_size = win_size

    def forward(self, pred):
        agg_loss = 0.0
        sym_loss = 0.0
        for i in range(0, len(pred), 4):
            intra_adj_v = pred[i]
            intra_adj_h = pred[i+1]
            inter_adj_v = pred[i+2]
            inter_adj_h = pred[i+3]
            agg_loss += self._calAggregation(intra_adj_v)
            agg_loss += self._calAggregation(inter_adj_h)
            sym_loss += self._calSymmetry(intra_adj_h)
            sym_loss += self._calSymmetry(inter_adj_v)

        agg_loss /= len(pred)//8
        sym_loss /= len(pred)//8
        loss = agg_loss + sym_loss

        return self.loss_weight * loss

    def _calAggregation(self, adj):
        if self.win_size == 1:
            return 0.0
        node_nums = adj.shape[0]
        agg_loss = 0.0
        count = 0.0
        for i in range(self.win_size//2, node_nums-self.win_size//2):
            temp = 0.0
            for j in range(1, self.win_size//2+1):
                temp += adj[i][i-j] + adj[i][i+j]
            agg_loss += 1.0 - temp/(self.win_size-1)
            count += 1.0
        agg_loss /= count
        return agg_loss

    def _calSymmetry(self, adj):
        node_nums = adj.shape[0]
        sym_loss = 0.0
        count = 0.0
        for i in range(self.win_size//2, node_nums-self.win_size//2):
            temp = adj[i][node_nums - 1 - i]
            for j in range(1, self.win_size // 2 + 1):
                temp += adj[i][node_nums - 1 - i - j] + adj[i][node_nums - 1 - i + j]
            sym_loss += 1.0 - temp/self.win_size
            count += 1.0
        sym_loss /= count
        return sym_loss

    # def _calAggregation(self, adj):
    #     node_nums = adj.shape[0]
    #     agg_loss = 0.0
    #     for i in range(self.win_size//2, node_nums-self.win_size//2):
    #         agg_loss += 1.0 - (adj[i][i-1] + adj[i][i+1]) / 2.0
    #     agg_loss /= node_nums - self.win_size + self.win_size % 2
    #     return agg_loss
    #
    # def _calSymmetry(self, adj):
    #     node_nums = adj.shape[0]
    #     sym_loss = 0.0
    #     for i in range(self.win_size//2, node_nums-self.win_size//2):
    #         sym_loss += 1.0 - (adj[i][node_nums-2-i] + adj[i][node_nums-1-i] + adj[i][node_nums-i]) / 3.0
    #     sym_loss /= node_nums - self.win_size + self.win_size % 2
    #     return sym_loss


