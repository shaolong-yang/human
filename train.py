import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from unet import Model
from datasetsss import MyDataset

def parse_args():
    parser = argparse.ArgumentParser(description="训练UNet模型（支持ttsasr模式）")
    parser.add_argument("--dataset_dir", required=True, help="数据集目录（包含ttsasr特征）")
    parser.add_argument("--save_dir", required=True, help="模型保存目录")
    parser.add_argument("--asr_mode", choices=["ttsasr"], required=True, help="ASR模式")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batchsize", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--use_syncnet", action="store_true", help="是否使用SyncNet权重")
    parser.add_argument("--syncnet_checkpoint", type=str, default="", help="SyncNet模型路径（需支持ttsasr）")
    return parser.parse_args()

def train(net, args, dataset_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = MyDataset(
        img_dir=dataset_dir, 
        mode=args.asr_mode
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batchsize, 
        shuffle=True, 
        num_workers=4,
        pin_memory=False # 启用pin_memory加速GPU传输（通常更有效）
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_epochs = args.epochs
    total_batches = len(dataloader)  # 总batch数
    
    # 训练循环
    for epoch in range(total_epochs):
        net.train()
        running_loss = 0.0
        # 打印当前epoch开始信息
        print(f"===== 开始 Epoch {epoch+1}/{total_epochs} =====", flush=True)
        
        for i, (imgs, targets, audio_feat) in enumerate(dataloader):
            # 数据设备分配
            imgs = imgs.to(device)
            targets = targets.to(device)
            audio_feat = audio_feat.to(device)
            
            # 模型计算
            optimizer.zero_grad()
            preds = net(imgs, audio_feat)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            # 损失计算
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)
            
            # 打印当前进度（每个batch都显示，或每N个batch显示一次）
            # 这里选择每个batch都显示，确保进度可见
            print(
                f"Epoch {epoch+1}/{total_epochs} | "
                f"Batch {i+1}/{total_batches} | "
                f"当前平均损失: {avg_loss:.6f}",
                flush=True
            )
        
        # Epoch结束总结
        print(
            f"===== Epoch {epoch+1}/{total_epochs} 完成 | "
            f"最终平均损失: {avg_loss:.6f} =====",
            flush=True
        )
        
        # 学习率调度
        scheduler.step(avg_loss)
        
        # 模型保存
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"unet_{args.asr_mode}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"模型已保存至: {checkpoint_path}", flush=True)
    
    print(f"{args.asr_mode}模式训练完成！", flush=True)

if __name__ == "__main__":
    args = parse_args()
    
    net = Model(
        n_channels=3,
        mode=args.asr_mode
    )
    
    if args.use_syncnet and os.path.exists(args.syncnet_checkpoint):
        try:
            syncnet_weights = torch.load(args.syncnet_checkpoint, map_location="cpu")
            if 'model_state_dict' in syncnet_weights:
                syncnet_weights = syncnet_weights['model_state_dict']
            net.load_state_dict(syncnet_weights, strict=False)
            print(f"已加载SyncNet预训练权重（{args.asr_mode}模式）: {args.syncnet_checkpoint}", flush=True)
        except Exception as e:
            print(f"SyncNet权重加载警告: {str(e)}，将继续训练", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print(f"使用{device}进行训练（{args.asr_mode}模式）", flush=True)
    
    train(
        net=net,
        args=args,
        dataset_dir=args.dataset_dir,
        save_dir=args.save_dir
    )