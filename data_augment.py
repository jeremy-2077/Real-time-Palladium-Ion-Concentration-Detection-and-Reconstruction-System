import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
from scipy.ndimage import gaussian_filter

def add_gaussian_noise(image, mean=0, sigma=25):
    """添加高斯噪声"""
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def motion_blur(image, size=15):
    """添加运动模糊"""
    # 生成运动模糊核
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    
    # 对每个通道进行模糊处理
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def add_shake(image, severity=5):
    """模拟相机抖动"""
    rows, cols, _ = image.shape
    # 生成随机位移
    dx = np.random.randint(-severity, severity)
    dy = np.random.randint(-severity, severity)
    
    # 创建变换矩阵
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(image, M, (cols, rows))
    return shifted

def overexposure(image, factor=1.5):
    """模拟过曝"""
    # 转换为PIL Image以使用ImageEnhance
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(image_pil)
    enhanced = enhancer.enhance(factor)
    # 转回OpenCV格式
    return cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)

def create_degraded_images(input_folder, output_folder):
    """创建各种退化的图像"""
    # 创建输出目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 创建子目录
    degradation_types = ['noise', 'blur', 'shake', 'exposure']
    subdirs = {}
    for dtype in degradation_types:
        subdir = os.path.join(output_folder, dtype)
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        subdirs[dtype] = subdir

    # 处理每张图片
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 读取图片
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"无法读取图片: {filename}")
                continue
                
            # 获取文件名（不含扩展名）
            name = os.path.splitext(filename)[0]
            
            # 1. 添加噪声
            noisy_image = add_gaussian_noise(image)
            cv2.imwrite(os.path.join(subdirs['noise'], f"{name}_noise.jpg"), noisy_image)
            
            # 2. 添加运动模糊
            blurred_image = motion_blur(image)
            cv2.imwrite(os.path.join(subdirs['blur'], f"{name}_blur.jpg"), blurred_image)
            
            # 3. 添加抖动
            shaken_image = add_shake(image)
            cv2.imwrite(os.path.join(subdirs['shake'], f"{name}_shake.jpg"), shaken_image)
            
            # 4. 添加过曝
            exposed_image = overexposure(image)
            cv2.imwrite(os.path.join(subdirs['exposure'], f"{name}_exposure.jpg"), exposed_image)
            
            print(f"处理完成: {filename}")

def combine_degradations(input_folder, output_folder, num_combinations=3):
    """随机组合多种退化效果"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                continue
                
            name = os.path.splitext(filename)[0]
            
            # 可用的退化函数列表
            degradations = [
                (add_gaussian_noise, {'mean': 0, 'sigma': np.random.randint(15, 35)}),
                (motion_blur, {'size': np.random.randint(10, 20)}),
                (add_shake, {'severity': np.random.randint(3, 8)}),
                (overexposure, {'factor': np.random.uniform(1.3, 1.8)})
            ]
            
            # 随机选择并应用多个退化
            selected_degradations = random.sample(degradations, num_combinations)
            degraded_image = image.copy()
            
            for deg_func, params in selected_degradations:
                degraded_image = deg_func(degraded_image, **params)
            
            # 保存组合退化后的图像
            output_path = os.path.join(output_folder, f"{name}_combined.jpg")
            cv2.imwrite(output_path, degraded_image)
            
            print(f"组合退化完成: {filename}")

if __name__ == "__main__":
    # 设置输入输出路径
    input_folder = "datasets/data_process/test"  # 原始训练数据目录
    output_base = "datasets/data_degraded"        # 退化图像的基础目录
    
    # 创建单一退化效果的图像
    create_degraded_images(input_folder, output_base)
    
    # 创建组合退化效果的图像
    combined_output = os.path.join(output_base, "combined")
    combine_degradations(input_folder, combined_output) 