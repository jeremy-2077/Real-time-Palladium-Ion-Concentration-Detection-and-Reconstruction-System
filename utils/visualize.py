import matplotlib.pyplot as plt
import os
import numpy as np

def plot_loss(pre_losses, recon_losses, total_losses, save_path='loss_curve.png'):
    """
    绘制损失曲线
    
    Args:
        losses: 每个epoch的损失值列表
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pre_losses) + 1), pre_losses, 'b-', label='Pre Loss')
    plt.plot(range(1, len(recon_losses) + 1), recon_losses, 'g-', label='Recon Loss')
    plt.plot(range(1, len(total_losses) + 1), total_losses, 'r-', label='Total Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    

def plot_train_val_loss(train_pre_losses, train_recon_losses, train_total_losses, 
                        val_pre_losses, val_recon_losses, val_total_losses, 
                        save_path='loss_curve.png'):
    """
    绘制训练集和验证集的损失曲线对比
    
    Args:
        train_*_losses: 训练集损失列表
        val_*_losses: 验证集损失列表
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(15, 10))
    
    # 创建三个子图
    plt.subplot(3, 1, 1)
    plt.plot(range(1, len(train_pre_losses) + 1), train_pre_losses, 'b-', label='训练集预测损失')
    plt.plot(range(1, len(val_pre_losses) + 1), val_pre_losses, 'r-', label='验证集预测损失')
    plt.title('预测损失曲线')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(range(1, len(train_recon_losses) + 1), train_recon_losses, 'g-', label='训练集重建损失')
    plt.plot(range(1, len(val_recon_losses) + 1), val_recon_losses, 'm-', label='验证集重建损失')
    plt.title('重建损失曲线')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(range(1, len(train_total_losses) + 1), train_total_losses, 'c-', label='训练集总损失')
    plt.plot(range(1, len(val_total_losses) + 1), val_total_losses, 'y-', label='验证集总损失')
    plt.title('总损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
