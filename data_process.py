import os
from PIL import Image
import numpy as np
import shutil
from sklearn.model_selection import train_test_split

def process_images(input_folder, output_folder, target_size=(256, 256)):
    """
    将输入文件夹中的大图片切分成小块并保存
    
    Args:
        input_folder: 输入图片所在文件夹路径
        output_folder: 输出图片保存文件夹路径
        target_size: 目标图片大小，默认256x256
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    c = 0
    dict1 = {}
    count = []
    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 获取图片label（假设文件名格式为"label_xxxxx.jpg"）
            label = filename.split('-')[0]
            if label not in dict1:
                dict1[label] = c
                c = c+1
                count.append(0)
            # 打开图片
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)
            
            # 获取图片尺寸
            width, height = img.size
            
            # 计算可以切分出多少个小块
            num_cols = width // target_size[0]
            num_rows = height // target_size[1]
            
            # 切分图片
            
            for i in range(num_rows):
                for j in range(num_cols):
                    # 计算当前小块的位置
                    left = j * target_size[0]
                    top = i * target_size[1]
                    right = left + target_size[0]
                    bottom = top + target_size[1]
                    
                    # 裁剪图片
                    crop = img.crop((left, top, right, bottom))
                    
                    # 保存裁剪后的图片
                    output_filename = f"{label}-{count[dict1[label]]:04d}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    crop.save(output_path, quality=95)
                    count[dict1[label]] += 1
            
            print(f"处理完成 {filename}，共生成 {count[dict1[label]]} 张子图片")

def split_dataset(data_folder, output_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    将数据集分割为训练集、验证集和测试集
    
    Args:
        data_folder: 包含处理后图片的文件夹路径
        output_folder: 分割后数据集的输出根目录
        train_ratio: 训练集比例，默认0.7
        val_ratio: 验证集比例，默认0.15
        test_ratio: 测试集比例，默认0.15
        random_state: 随机种子，默认42
    """
    # 确保比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例之和必须为1"
    
    # 创建输出根目录
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 创建训练集、验证集和测试集文件夹
    train_folder = os.path.join(output_folder, 'train')
    val_folder = os.path.join(output_folder, 'val')
    test_folder = os.path.join(output_folder, 'test')
    
    for folder in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # 获取所有图片文件名
    all_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 按标签分组
    label_files = {}
    for file in all_files:
        label = file.split('-')[0]
        if label not in label_files:
            label_files[label] = []
        label_files[label].append(file)
    
    # 对每个标签进行分割
    for label, files in label_files.items():
        # 计算相对比例用于第二次分割
        remaining_ratio = val_ratio + test_ratio
        val_ratio_relative = val_ratio / remaining_ratio
        
        # 首先分割出训练集和其他(验证+测试)
        train_files, temp_files = train_test_split(
            files, 
            train_size=train_ratio, 
            random_state=random_state
        )
        
        # 然后将剩余部分分割为验证集和测试集
        val_files, test_files = train_test_split(
            temp_files,
            train_size=val_ratio_relative,
            random_state=random_state
        )
        
        # 复制文件到对应文件夹（使用复制而非移动，保留原始处理后的数据）
        for file_list, target_folder in [
            (train_files, train_folder),
            (val_files, val_folder),
            (test_files, test_folder)
        ]:
            for file in file_list:
                src = os.path.join(data_folder, file)
                dst = os.path.join(target_folder, file)
                shutil.copy(src, dst)  # 使用copy而不是move
        
        print(f"标签 {label}: 训练集 {len(train_files)} 张图片, 验证集 {len(val_files)} 张图片, 测试集 {len(test_files)} 张图片")

if __name__ == "__main__":
    # 设置输入输出路径
    input_folder = "datasets/data"
    output_folder = "datasets/data_process"
    split_output_folder = "datasets/split_data"  # 新增分割数据的输出目录
    
    # 处理图片
    process_images(input_folder, output_folder, target_size=(224, 224))
    
    # 分割数据集
    split_dataset(output_folder, split_output_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

 