import argparse
import torch
from torch.utils.data import DataLoader
from moco.engine import MoCo
from dataset import JadeMoCoDataset
from utils.early_stop import EarlyStopper
from moco.DMS import DMSEncoder
from tqdm import tqdm
import torch.cuda as cuda
import timm
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=r"G:\CODE\data\data\train", help='玉石数据集路径')
    parser.add_argument('--bg_dir', default=r"G:\CODE\data\data\bac", help='背景图片目录')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--resume', default='G:\CODE\moco\output\\best_checkpoint.pth', help='检查点路径')
    parser.add_argument('--output_dir', default='output', help='输出目录')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 penalty)')
    return parser.parse_args()

class SwinEncoder(nn.Module):
    def __init__(self, base_model='swin_tiny_patch4_window7_224'):
        super(SwinEncoder, self).__init__()
        self.encoder = timm.create_model(base_model, pretrained=True, num_classes=0) 
        
    def forward(self, x):
        x = self.encoder(x)
        return x
    
def moco_collate_fn(batch):
    view1 = torch.stack([item[0] for item in batch])
    view2 = torch.stack([item[1] for item in batch])
    return [view1, view2]

def print_grad_norm(model, epoch):
    """
    遍历所有可训练参数，计算全局梯度 L2 范数并打印。
    如果想看分层，可把 param 分组或按 name 打印。
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f'[Epoch {epoch}] global grad norm = {total_norm:.6f}')


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化组件
    early_stopper = EarlyStopper(patience=10) 
    
    # 数据集和加载器
    dataset = JadeMoCoDataset(args.data, args.bg_dir)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True, 
                            collate_fn=moco_collate_fn)
    
    # 模型和优化器
    model = MoCo(base_encoder=DMSEncoder).to(device)
    # model = MoCo(base_encoder=SwinEncoder).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=args.weight_decay)

    for param_group in optimizer.param_groups:
        initial_lr = param_group['lr']
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # 保存最佳checkpoint
    best_loss = float('inf')
            
    # 训练循环
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch}/{args.epochs}] Learning Rate: {current_lr:.8f}')
 
        
        train_bar = tqdm(train_loader, desc=f'Epoch [{epoch}/{args.epochs}]')
        for batch_idx, images in enumerate(train_bar):
            im_q, im_k = images[0].to(device), images[1].to(device)

            loss = model(im_q, im_k) + 0.1 * model.encoder_q.get_total_regularization_loss()
            # loss = model(im_q, im_k)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            train_bar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch}/{args.epochs}] Avg Loss: {avg_loss:.4f}')
        print_grad_norm(model, epoch)
        
        if cuda.is_available():
            cuda.empty_cache()

        # 早停检查
        if early_stopper(avg_loss):
            print(f'Early stopping triggered at epoch {epoch}')
            break
        
        # 保存最佳检查点
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'{args.output_dir}/best_checkpoint.pth')
            print(f'Saved best checkpoint at epoch {epoch}')
        
        if epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'{args.output_dir}/checkpoint_{epoch}.pth')
            print(f'Saved checkpoint at epoch {epoch}')

if __name__ == '__main__':
    main()