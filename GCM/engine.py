import torch
import torch.nn as nn
from .builder import GraphBuilder
import copy
import torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.99, T=0.2, encoder_output_dim=768):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        
        # 初始化编码器
        self.encoder_q = base_encoder()
        self.encoder_k = copy.deepcopy(self.encoder_q)
        
        # 冻结动量编码器参数
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False
            
        # 投影头
        self.proj_q = nn.Sequential(
            nn.Linear(encoder_output_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.proj_k = copy.deepcopy(self.proj_q)
        
        # 图结构构建器
        self.graph_builder = GraphBuilder(feat_dim=dim, queue_size=K)
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新key编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
    def forward(self, im_q, im_k):
        # 计算query特征
        q = self.proj_q(self.encoder_q(im_q))
        q = F.normalize(q, dim=1)
        
        # 计算key特征
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.proj_k(self.encoder_k(im_k))
            k = F.normalize(k, dim=1)
        
        # 更新图结构（添加异常处理）
        try:
            self.graph_builder.update(k)
            subgraph_feats, edge_index = self.graph_builder.build_subgraph(q)
            
            # 安全计算图损失
            if subgraph_feats.dim() == 2 and edge_index.shape[1] > 0:
                graph_loss = self.graph_builder.contrastive_loss(subgraph_feats, edge_index)
            else:
                graph_loss = torch.tensor(0.0, device=q.device)
        except Exception as e:
            print(f"图构建失败: {e}")
            graph_loss = torch.tensor(0.0, device=q.device)
        
        # 原始MoCo损失
        logits = torch.mm(q, k.t()) / self.T
        labels = torch.arange(logits.size(0), device=logits.device)
        moco_loss = F.cross_entropy(logits, labels)
        
        return moco_loss + 0.3 * graph_loss