# utils/linear_eval.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class LinearClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

def evaluate_linear(encoder, train_loader, test_loader, device, num_classes=10, lr=0.01, epochs=100):
    """线性评估协议核心实现"""
    # 冻结编码器
    encoder.eval()
    feat_dim = encoder.fc.weight.shape[1] if hasattr(encoder, 'fc') else 768  # 根据你的编码器调整
    
    # 初始化线性分类器
    classifier = LinearClassifier(feat_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # 训练线性层
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                features = encoder(x)  # 获取表征
            outputs = classifier(features)
            loss = criterion(outputs, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # 在测试集上评估
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            features = encoder(x)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return correct / total * 100