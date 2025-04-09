import argparse
import torch
import yaml
import os
from PIL import Image
import torchvision.transforms as transforms
from utils.yaml_config_hook import yaml_config_hook
import modules.network as network
import modules.resnet as resnet

def load_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    res = resnet.get_resnet(args.resnet, pretrained=False, feature_dim=args.feature_dim)
    model = network.Network(res)
    # model_fp = os.path.join(args.model_path, "best_checkpoint_{}.tar".format(args.eval_epoch))
    model_fp = os.path.join("/Users/jeremiahncross/Downloads/pred_best_checkpoint_99-2.tar")
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)
    model.eval()
    return model, device

def process_image(image_path):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        return None

def inference(model, image_tensor, device):
    try:
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            prediction, reconstruction = model(image_tensor)
            
            # 反归一化重建图像
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            reconstruction = reconstruction * std + mean
            
            # 确保像素值在[0,1]范围内
            reconstruction = torch.clamp(reconstruction, 0, 1)
            
            return prediction.item(), reconstruction.cpu()
    except Exception as e:
        print(f"推理过程中出错: {str(e)}")
        return None, None

def test_inference(image_path, args):
    """
    用于测试推理功能的函数
    """
    model, device = load_model(args)
    image_tensor = process_image(image_path)
    if image_tensor is not None:
        prediction, reconstruction = inference(model, image_tensor, device)
        if prediction is not None:
            print(f"预测值: {prediction:.2f}")
            return True
    return False

