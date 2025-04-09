import torch
import argparse
from utils import yaml_config_hook
from utils.custom_image import CustomDataset
from modules import resnet, network
from torchvision import transforms
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.visualize import plot_loss

def predict(loader, model, device):
    model.eval()
    predict_vector = []
    recon_vector = []
    label_vector = []
    with torch.no_grad():
        for step, (x, y) in enumerate(loader):
            x = x.to(device)
            with torch.no_grad():
                y_pre, y_recon = model(x)
            y_pre = y_pre.cpu().detach().numpy()
            y_recon = y_recon.cpu().detach().numpy()
            print(y_pre, y.numpy(), "\n")
            predict_vector.extend(y_pre)
            recon_vector.extend(y_recon)
            label_vector.extend(y.numpy())

    predict_vector = np.array(predict_vector)
    label_vector = np.array(label_vector)
    recon_vector = np.array(recon_vector)
    print("Predict shape {}".format(predict_vector.shape))
    print("Recon shape {}".format(recon_vector.shape))
    return predict_vector, label_vector, recon_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    dataset_path = args.dataset_dir
    dataset_path = args.dataset_dir
    dataset = CustomDataset(
        data_path=dataset_path+"/test",  # 训练集路径
        transform=train_transform,
        augment=False  # 启用数据增强
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim)
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    X, Y, Z = predict(data_loader, model, device)

    # 计算MSELoss
    criterion = torch.nn.MSELoss()
    loss = criterion(torch.tensor(X), torch.tensor(Y))
    print("Predict_MSE:", loss)

    loss = criterion(torch.tensor(X), torch.tensor(Z))
    print("Reconstruct_MSE:", loss)

    # 计算MAE
    mae = np.mean(np.abs(X - Y))
    print("Predict_MAE:", mae)    

    # 计算R方
    r2 = 1 - np.sum((X - Y) ** 2) / np.sum((Y - np.mean(Y)) ** 2)
    print("Predict_R2:", r2)


    
    