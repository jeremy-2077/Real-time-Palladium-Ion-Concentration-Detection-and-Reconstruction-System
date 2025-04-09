import os
import torch


def save_model(args, model, optimizer, current_epoch, prefix=""):
    """
    保存模型检查点
    
    Args:
        args: 参数
        model: 模型
        optimizer: 优化器
        current_epoch: 当前轮次
        prefix: 文件名前缀，用于区分不同类型的检查点
    """
    if prefix:
        out = os.path.join(args.model_path, f"{prefix}_checkpoint_{current_epoch}.tar")
    else:
        out = os.path.join(args.model_path, f"checkpoint_{current_epoch}.tar")
        
    state = {
        'net': model.state_dict(), 
        'optimizer': optimizer.state_dict(), 
        'epoch': current_epoch
    }
    torch.save(state, out)
