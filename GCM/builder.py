import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class GraphBuilder(nn.Module):
    def __init__(self, feat_dim=128, queue_size=65536, topk=5, temperature=0.2):
        super().__init__()
        self.queue_size = queue_size
        self.topk = topk
        self.temperature = temperature
        self.register_buffer("features", torch.randn(queue_size, feat_dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        
        # 修改初始化方式
        nn.init.orthogonal_(self.features)
        self.features = F.normalize(self.features, p=2, dim=1)

    @torch.no_grad()
    def update(self, new_features):
        batch_size = new_features.size(0)
        
        # 确保不超过队列容量
        if batch_size > self.queue_size:
            new_features = new_features[:self.queue_size]
            batch_size = self.queue_size
        
        # 循环队列实现
        start = self.ptr % self.queue_size
        end = (self.ptr + batch_size) % self.queue_size
        
        if start < end:
            self.features[start:end] = F.normalize(new_features, p=2, dim=1)
        else:
            # 处理回绕情况
            part1_size = self.queue_size - start
            self.features[start:] = F.normalize(new_features[:part1_size], p=2, dim=1)
            self.features[:end] = F.normalize(new_features[part1_size:], p=2, dim=1)
        
        self.ptr = (self.ptr + batch_size) % self.queue_size
        
    def build_subgraph(self, queries):
        """构建动态子图并返回边索引和负样本"""
        assert queries.dim() == 2, f"Input queries must be 2D, got {queries.shape}"
        
        # 获取当前队列中的有效特征数量
        valid_size = min(self.ptr.item(), self.queue_size)
        if valid_size == 0:
            return torch.zeros(1, device=queries.device), torch.empty((2,0), device=queries.device)

        
        # 分块计算
        sim_matrix = []
        chunk_size = min(1024, valid_size)
        for i in range(0, valid_size, chunk_size):
            chunk = self.features[i:i+chunk_size]
            sim_chunk = torch.matmul(queries, chunk.t())  # 更高效的计算方式
            sim_matrix.append(sim_chunk)
        
        sim_matrix = torch.cat(sim_matrix, dim=1) / self.temperature
            
        # 动态调整topk值，不超过可用样本数
        current_topk = min(self.topk, valid_size)
        _, indices = torch.topk(sim_matrix, current_topk, dim=1)  # [N, current_topk]
        
        # 构建子图连接关系
        batch_size = queries.size(0)
        edge_index = []
        for i in range(batch_size):
            edge_index.extend([(i, batch_size + j) for j in range(current_topk)])
        
        # 合并query和负样本特征
        selected_features = self.features[indices.view(-1)]  # [N*current_topk, dim]
        subgraph_feats = torch.cat([
            queries,  # [N, dim]
            selected_features  # [N*current_topk, dim]
        ], dim=0)  # [N + N*current_topk, dim]
        
        return subgraph_feats, torch.tensor(edge_index).t().contiguous().to(queries.device)
    
    def contrastive_loss(self, subgraph_feats, edge_index):
        if subgraph_feats.dim() == 1 or edge_index.shape[1] == 0:
            return torch.tensor(0.0, device=subgraph_feats.device)

        # 归一化特征
        feats = F.normalize(subgraph_feats, p=2, dim=1)
        sim_matrix = torch.mm(feats, feats.t())  # [N, N]

        pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
        pos_mask[edge_index[0], edge_index[1]] = True

        # 提取正负样本相似度
        pos_sim = sim_matrix[pos_mask]      # [P]
        neg_sim = sim_matrix[~pos_mask]     # [Q]

        if pos_sim.numel() == 0 or neg_sim.numel() == 0:
            return torch.tensor(0.0, device=subgraph_feats.device)

        # 数值稳定版 InfoNCE
        # log(exp(p/t) / (exp(p/t) + sum(exp(n_i/t))) = p/t - log(exp(p/t) + sum(exp(n_i/t)))
        # = p/t - logsumexp([p/t, n_1/t, n_2/t, ...])

        pos_term = pos_sim.mean() / self.temperature
        neg_term = neg_sim.mean() / self.temperature

        # 稳定计算 log(exp(a) / (exp(a) + exp(b))) = a - log(exp(a) + exp(b))
        # = -log(1 + exp(b - a))  if a > b
        max_val = torch.max(pos_term, neg_term)
        log_sum_exp = max_val + torch.log(
            torch.exp(pos_term - max_val) + torch.exp(neg_term - max_val)
        )
        loss = -(pos_term - log_sum_exp)

        return loss