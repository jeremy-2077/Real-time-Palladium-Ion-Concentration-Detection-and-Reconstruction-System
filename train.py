import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from torchvision import models
import re
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import random
import torch
from utils.custom_image import CustomDataset
from utils.save_model import save_model
from utils.visualize import plot_loss, plot_train_val_loss
import argparse
from utils.yaml_config_hook import yaml_config_hook
from modules.resnet import get_resnet
from modules.network import Network
import time
from datetime import timedelta

def train(model, data_loader, optimizer, pre_criterion, recon_criterion, device, epoch, total_epochs, training_phase="combined"):
    if training_phase == "prediction":
        # 确保在预测阶段，只有predict_projector被训练
        model.eval()  # 整个模型先设为eval模式
        model.predict_projector.train()  # 只有预测投影器设为训练模式
    elif training_phase == "reconstruction":
        # 重建阶段，只有自编码器部分(ResNet+decoder)被训练
        model.train()  # 整个模型先设为训练模式
        model.predict_projector.eval()  # 预测投影器设为评估模式
    else:
        # 联合训练时，整个模型都训练
        model.train()
        
    loss_epoch = 0
    pre_loss_epoch = 0
    recon_loss_epoch = 0
    total_samples = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    
    # Initialize progress tracking
    start_time = time.time()
    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch}/{total_epochs} ({training_phase})")
    
    for step, (x, y) in enumerate(progress_bar):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        
        # 需要根据阶段设置梯度计算
        if training_phase == "prediction":
            # 预测阶段：ResNet部分不需要梯度计算
            with torch.no_grad():
                features = model.resnet(x)
            # 只有预测部分计算梯度
            y_pre = model.predict_projector(features)
            y_recon = model.decoder(features.detach())  # 解码器在此阶段也不更新
        elif training_phase == "reconstruction":
            # 重建阶段：预测部分不需要梯度计算
            features = model.resnet(x)
            with torch.no_grad():
                y_pre = model.predict_projector(features.detach())
            y_recon = model.decoder(features)
        else:
            # 正常的前向传播
            y_pre, y_recon = model(x)
        
        # 根据训练阶段计算不同的损失
        if training_phase == "reconstruction":
            # 第一阶段：只计算重建损失
            recon_loss = recon_criterion(x, y_recon)
            loss = recon_loss
            pre_loss = torch.tensor(0.0, device=device)  # 占位，不参与训练
        elif training_phase == "prediction":
            # 第二阶段：只计算预测损失
            pre_loss = pre_criterion(y, y_pre.squeeze())
            loss = pre_loss
            recon_loss = torch.tensor(0.0, device=device)  # 占位，不参与训练
        else:  # "combined"
            # 组合训练：同时计算两种损失
            pre_loss = pre_criterion(y, y_pre.squeeze())
            recon_loss = recon_criterion(x, y_recon)
            loss = pre_loss + recon_loss
            
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        pre_loss_epoch += pre_loss.item()
        recon_loss_epoch += recon_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'batch': f"{step+1}/{len(data_loader)}",
            'loss': f"{loss.item():.4f}",
            'pre_loss': f"{pre_loss.item():.4f}",
            'recon_loss': f"{recon_loss.item():.4f}"
        })
    
    elapsed_time = time.time() - start_time
    
    # Calculate average losses
    avg_loss = loss_epoch / len(data_loader)
    avg_pre_loss = pre_loss_epoch / len(data_loader)
    avg_recon_loss = recon_loss_epoch / len(data_loader)
    
    print(f"Training completed in {timedelta(seconds=elapsed_time)}")
    print(f"Processed {total_samples} samples in {len(data_loader)} batches")
    
    return avg_loss, avg_pre_loss, avg_recon_loss

def validate(model, data_loader, pre_criterion, recon_criterion, device, epoch, total_epochs, training_phase="combined"):
    model.eval()
    loss_epoch = 0
    pre_loss_epoch = 0
    recon_loss_epoch = 0
    
    # Initialize progress tracking
    start_time = time.time()
    progress_bar = tqdm(data_loader, desc=f"Validating Epoch {epoch}/{total_epochs} ({training_phase})")
    
    with torch.no_grad():
        for step, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y = y.to(device)
            
            # 预测阶段时的特殊处理，保持一致性
            if training_phase == "prediction":
                features = model.resnet(x)
                y_pre = model.predict_projector(features)
                y_recon = model.decoder(features)
            elif training_phase == "reconstruction":
                features = model.resnet(x)
                y_pre = model.predict_projector(features)
                y_recon = model.decoder(features)
            else:
                y_pre, y_recon = model(x)
            
            # 根据训练阶段计算不同的损失
            if training_phase == "reconstruction":
                # 第一阶段：只计算重建损失
                recon_loss = recon_criterion(x, y_recon)
                loss = recon_loss
                pre_loss = torch.tensor(0.0, device=device)
            elif training_phase == "prediction":
                # 第二阶段：只计算预测损失
                pre_loss = pre_criterion(y, y_pre.squeeze())
                loss = pre_loss
                recon_loss = torch.tensor(0.0, device=device)
            else:  # "combined"
                # 组合训练：同时计算两种损失
                pre_loss = pre_criterion(y, y_pre.squeeze())
                recon_loss = recon_criterion(x, y_recon)
                loss = pre_loss + recon_loss

            loss_epoch += loss.item()
            pre_loss_epoch += pre_loss.item()
            recon_loss_epoch += recon_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'batch': f"{step+1}/{len(data_loader)}",
                'loss': f"{loss.item():.4f}",
                'pre_loss': f"{pre_loss.item():.4f}",
                'recon_loss': f"{recon_loss.item():.4f}"
            })
    
    elapsed_time = time.time() - start_time
    print(f"Validation completed in {timedelta(seconds=elapsed_time)}")
    
    return loss_epoch / len(data_loader), pre_loss_epoch / len(data_loader), recon_loss_epoch / len(data_loader)

def test(model, data_loader, pre_criterion, recon_criterion, device):
    model.eval()
    predict_vector = []
    label_vector = []
    recon_vector = []
    
    # Initialize progress tracking
    start_time = time.time()
    progress_bar = tqdm(data_loader, desc="Testing")
    
    with torch.no_grad():
        for step, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y_pre, y_recon = model(x)
            y_pre = y_pre.cpu().detach().numpy()
            y_recon = y_recon.cpu().detach().numpy()
            predict_vector.extend(y_pre)
            recon_vector.extend(y_recon)
            label_vector.extend(y.numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'batch': f"{step+1}/{len(data_loader)}"
            })
    
    predict_vector = np.array(predict_vector).squeeze()
    label_vector = np.array(label_vector)
    
    # Calculate evaluation metrics
    mse = np.mean((predict_vector - label_vector) ** 2)
    mae = np.mean(np.abs(predict_vector - label_vector))
    # Calculate R^2
    r2 = 1 - np.sum((predict_vector - label_vector) ** 2) / np.sum((label_vector - np.mean(label_vector)) ** 2)
    
    elapsed_time = time.time() - start_time
    print(f"Testing completed in {timedelta(seconds=elapsed_time)}")
    
    return mse, mae, r2

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_gpu_memory_usage():
    """Print GPU memory usage if CUDA is available"""
    if torch.cuda.is_available():
        print(f"GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        print(f"  Device:    {torch.cuda.get_device_name(0)}")

def seed_worker(worker_id):
    """为DataLoader设置随机种子，确保多进程加载数据的可重复性"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Prepare data
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    dataset_path = args.dataset_dir
    
    # Load training set
    train_dataset = CustomDataset(
        data_path=dataset_path+"/train",  # Training set path
        transform=train_transform,
        augment=True,  # Enable data augmentation
        aug_ratio=args.aug_ratio
    )
    
    # Load validation set
    val_dataset = CustomDataset(
        data_path=dataset_path+"/val",  # Validation set path
        transform=train_transform,
        augment=False  # Validation set doesn't use data augmentation
    )
    
    # Load test set
    test_dataset = CustomDataset(
        data_path=dataset_path+"/test",  # Test set path
        transform=train_transform,
        augment=False  # Test set doesn't use data augmentation
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=seed_worker
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers
    )
    
    # Print dataset information
    print("\n===== Dataset Information =====")
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples:       {len(test_dataset)}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches:       {len(test_loader)}")
    
    # Initialize model
    print("\n===== Model Configuration =====")
    print(f"Base ResNet:        {args.resnet}")
    print(f"Pretrained:         {args.pretrained}")
    print(f"Feature dimension:  {args.feature_dim}")
    
    res = get_resnet(args.resnet, args.pretrained, args.feature_dim)
    model = Network(res)
    model = model.to(device)
    
    # Print model information
    num_params = count_parameters(model)
    num_resnet_params = count_parameters(model.resnet)
    num_projector_params = count_parameters(model.predict_projector)
    num_decoder_params = count_parameters(model.decoder)
    num_ae_params = num_resnet_params + num_decoder_params
    
    print(f"Total parameters:     {num_params:,}")
    print(f"ResNet parameters:    {num_resnet_params:,}")
    print(f"Decoder parameters:   {num_decoder_params:,}")
    print(f"AE parameters:        {num_ae_params:,}")
    print(f"Projector parameters: {num_projector_params:,}")
    
    # Print hardware information
    print(f"\n===== Hardware Information =====")
    print(f"Device:             {device}")
    print_gpu_memory_usage()
    
    # 第一阶段的优化器：只优化自编码器部分(ResNet + decoder)
    ae_params = list(model.resnet.parameters()) + list(model.decoder.parameters())
    optimizer_ae = torch.optim.Adam(ae_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 损失函数
    pre_criterion = torch.nn.MSELoss()
    recon_criterion = torch.nn.MSELoss()
    
    # Learning rate scheduler for AE
    scheduler_ae = torch.optim.lr_scheduler.StepLR(optimizer_ae, step_size=30, gamma=0.1)
    
    # Print training configuration
    print("\n===== Training Configuration =====")
    print(f"Learning rate:      {args.learning_rate}")
    print(f"Weight decay:       {args.weight_decay}")
    print(f"First phase epochs: {args.recon_epochs} (reconstruction only, predict_projector frozen)")
    print(f"Second phase epochs: {args.pred_epochs} (prediction only, ResNet frozen)")
    print(f"Total epochs:       {args.recon_epochs + args.pred_epochs}")
    print(f"LR scheduler:       StepLR (step={30}, gamma={0.1})")
    
    # For recording losses per epoch
    train_pre_losses = []
    train_recon_losses = []
    train_total_losses = []
    
    val_pre_losses = []
    val_recon_losses = []
    val_total_losses = []
    
    # Record best validation loss
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Training timer
    training_start_time = time.time()
    
    print("\n===== Starting First Phase: Reconstruction Training =====")
    print("Freezing predict_projector, only training autoencoder (ResNet + decoder)")
    
    # 打印第一阶段可训练参数数量
    trainable_params_ae = sum(p.numel() for p in ae_params if p.requires_grad)
    print(f"Trainable parameters in reconstruction phase: {trainable_params_ae:,} (only autoencoder)")
    
    # 第一阶段：训练重建任务
    for epoch in range(args.recon_epochs):
        epoch_start_time = time.time()
        
        # Current learning rate
        current_lr = scheduler_ae.get_last_lr()[0]
        print(f"\nEpoch [{epoch}/{args.recon_epochs}] (Reconstruction) - Learning Rate: {current_lr:.6f}")
        
        # Training phase - 使用自编码器优化器
        train_loss, train_pre_loss, train_recon_loss = train(
            model, train_loader, optimizer_ae, pre_criterion, recon_criterion, 
            device, epoch, args.recon_epochs, training_phase="reconstruction"
        )
        scheduler_ae.step()
        
        # Validation phase
        val_loss, val_pre_loss, val_recon_loss = validate(
            model, val_loader, pre_criterion, recon_criterion, 
            device, epoch, args.recon_epochs, training_phase="reconstruction"
        )
        
        # Record losses
        train_pre_losses.append(train_pre_loss)
        train_recon_losses.append(train_recon_loss)
        train_total_losses.append(train_loss)
        
        val_pre_losses.append(val_pre_loss)
        val_recon_losses.append(val_recon_loss)
        val_total_losses.append(val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print current epoch training and validation losses
        print(f"\n===== Epoch {epoch}/{args.recon_epochs} (Reconstruction) Summary =====")
        print(f"Time:                 {timedelta(seconds=epoch_time)}")
        print(f"Train Loss:           {train_loss:.6f}")
        print(f"  - Reconstruction:   {train_recon_loss:.6f}")
        print(f"Val Loss:             {val_loss:.6f}")
        print(f"  - Reconstruction:   {val_recon_loss:.6f}")
        
        # Print memory usage
        print_gpu_memory_usage()
        
        
        # Reconstruction phase validation loss
        if val_loss < best_val_loss and epoch > args.recon_epochs // 4 * 3:
            best_val_loss = val_loss
            best_epoch = epoch
            save_model(args, model, optimizer_ae, epoch, prefix="recon_best")
            print(f"==> Saved best reconstruction model, validation loss: {best_val_loss:.6f}")
    
    print("\n===== First Phase Completed, Starting Second Phase: Prediction Training =====")
    print("Freezing ResNet backbone and decoder, only training prediction projector")
    
    # 加载重建阶段的最佳模型权重
    best_recon_model_path = os.path.join(args.model_path, f"recon_best_checkpoint_{best_epoch}.tar")
    print(f"Loading best reconstruction model from epoch {best_epoch}")
    state_dict = torch.load(best_recon_model_path, map_location=device)
    model.load_state_dict(state_dict['net'])
    
    # 第二阶段：重新初始化优化器，只对predict_projector部分进行优化
    optimizer_pred = torch.optim.Adam(model.predict_projector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler_pred = torch.optim.lr_scheduler.StepLR(optimizer_pred, step_size=30, gamma=0.1)
    
    # 重置最佳验证损失以开始新阶段
    best_val_loss_pred = float('inf')
    best_epoch_pred = 0
    
    # 打印可训练参数数量
    trainable_params = sum(p.numel() for p in model.predict_projector.parameters() if p.requires_grad)
    print(f"Trainable parameters in prediction phase: {trainable_params:,} (only predict_projector)")
    
    # 第二阶段：训练预测任务
    for epoch in range(args.pred_epochs):
        epoch_start_time = time.time()
        
        # Current learning rate
        current_lr = scheduler_pred.get_last_lr()[0]
        print(f"\nEpoch [{epoch}/{args.pred_epochs}] (Prediction) - Learning Rate: {current_lr:.6f}")
        
        # Training phase - 使用新的优化器
        train_loss, train_pre_loss, train_recon_loss = train(
            model, train_loader, optimizer_pred, pre_criterion, recon_criterion, 
            device, epoch, args.pred_epochs, training_phase="prediction"
        )
        scheduler_pred.step()
        
        # Validation phase
        val_loss, val_pre_loss, val_recon_loss = validate(
            model, val_loader, pre_criterion, recon_criterion, 
            device, epoch, args.pred_epochs, training_phase="prediction"
        )
        
        # Record losses
        train_pre_losses.append(train_pre_loss)
        train_recon_losses.append(train_recon_loss)
        train_total_losses.append(train_loss)
        
        val_pre_losses.append(val_pre_loss)
        val_recon_losses.append(val_recon_loss)
        val_total_losses.append(val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print current epoch training and validation losses
        print(f"\n===== Epoch {epoch}/{args.pred_epochs} (Prediction) Summary =====")
        print(f"Time:                 {timedelta(seconds=epoch_time)}")
        print(f"Train Loss:           {train_loss:.6f}")
        print(f"  - Prediction Loss:  {train_pre_loss:.6f}")
        print(f"Val Loss:             {val_loss:.6f}")
        print(f"  - Prediction Loss:  {val_pre_loss:.6f}")
        
        # Print memory usage
        print_gpu_memory_usage()
        
        
        # If current validation loss is the best, save best model
        if val_loss < best_val_loss_pred and epoch > args.pred_epochs // 4 * 3:
            best_val_loss_pred = val_loss
            best_epoch_pred = epoch
            save_model(args, model, optimizer_pred, epoch + args.recon_epochs, prefix="pred_best")
            print(f"==> Saved best prediction model, validation loss: {best_val_loss_pred:.6f}")
    
    total_training_time = time.time() - training_start_time
    print(f"\n===== Training Completed =====")
    print(f"Total training time: {timedelta(seconds=total_training_time)}")
    print(f"Best reconstruction epoch: {best_epoch}")
    print(f"Best reconstruction validation loss: {best_val_loss:.6f}")
    print(f"Best prediction epoch: {best_epoch_pred + args.recon_epochs}")
    print(f"Best prediction validation loss: {best_val_loss_pred:.6f}")
    
    # Save final model
    save_model(args, model, optimizer_pred, args.recon_epochs + args.pred_epochs, prefix="final")
    
    # Plot training and validation loss curves
    plot_train_val_loss(
        train_pre_losses, train_recon_losses, train_total_losses,
        val_pre_losses, val_recon_losses, val_total_losses,
        os.path.join(args.model_path, 'loss_curve.png')
    )
    
    # Save loss data to text file
    loss_txt_path = os.path.join(args.model_path, 'loss_curve.txt')
    with open(loss_txt_path, 'w') as f:
        f.write("Epoch\tTrain_Pre_Loss\tTrain_Recon_Loss\tTrain_Total_Loss\tVal_Pre_Loss\tVal_Recon_Loss\tVal_Total_Loss\tPhase\n")
        for i, (tr_pre, tr_recon, tr_total, v_pre, v_recon, v_total) in enumerate(
                zip(train_pre_losses, train_recon_losses, train_total_losses,
                    val_pre_losses, val_recon_losses, val_total_losses)):
            epoch = i
            phase = "Reconstruction" if epoch < args.recon_epochs else "Prediction"
            f.write(f"{epoch}\t{tr_pre:.6f}\t{tr_recon:.6f}\t{tr_total:.6f}\t{v_pre:.6f}\t{v_recon:.6f}\t{v_total:.6f}\t{phase}\n")
    
    print("\n===== Starting Model Evaluation =====")
    
    # 使用预测阶段的最佳模型进行测试
    best_model_path = os.path.join(args.model_path, f"pred_best_checkpoint_{best_epoch_pred + args.recon_epochs}.tar")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict['net'])
    print(f"Loaded best prediction model from epoch {best_epoch_pred + args.recon_epochs}")
    
    # Evaluate best model on test set
    test_mse, test_mae, test_r2 = test(model, test_loader, pre_criterion, recon_criterion, device)
    
    # Output test set performance metrics
    print("\n===== Test Set Performance =====")
    print(f"Best model (Epoch {best_epoch_pred + args.recon_epochs}):")
    print(f"Test MSE:             {test_mse:.6f}")
    print(f"Test MAE:             {test_mae:.6f}")
    print(f"Test R²:              {test_r2:.6f}")
    
    # Save test results to file
    test_result_path = os.path.join(args.model_path, 'test_results.txt')
    with open(test_result_path, 'w') as f:
        f.write(f"Best Reconstruction Epoch: {best_epoch}\n")
        f.write(f"Best Reconstruction Validation Loss: {best_val_loss:.6f}\n")
        f.write(f"Best Prediction Epoch: {best_epoch_pred + args.recon_epochs}\n")
        f.write(f"Best Prediction Validation Loss: {best_val_loss_pred:.6f}\n")
        f.write(f"Test MSE: {test_mse:.6f}\n")
        f.write(f"Test MAE: {test_mae:.6f}\n")
        f.write(f"Test R²: {test_r2:.6f}\n")

    print("\n===== All Results Saved =====")
    print(f"Model checkpoints:    {args.model_path}")
    print(f"Loss curves:          {os.path.join(args.model_path, 'loss_curve.png')}")
    print(f"Loss data:            {loss_txt_path}")
    print(f"Test results:         {test_result_path}")
