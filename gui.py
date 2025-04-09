import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, 
                           QVBoxLayout, QHBoxLayout, QWidget, QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch
import numpy as np
from utils.yaml_config_hook import yaml_config_hook
from inference import load_model, process_image, inference
import torchvision.transforms as transforms
import argparse

class ImageLabel(QLabel):
    def __init__(self, title=""):
        super().__init__()
        self.title = QLabel(title)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("""
            QLabel {
                font-family: Arial;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(224, 224)
        self.image_label.setStyleSheet("""
            QLabel {
                font-family: Arial;
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: #f5f5f5;
            }
        """)
        self.image_label.setAlignment(Qt.AlignCenter)

    def setPixmap(self, pixmap):
        scaled_pixmap = pixmap.scaled(224, 224, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def clear(self):
        self.image_label.clear()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Prediction and Reconstruction System")
        self.setGeometry(100, 100, 800, 500)
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
            QLabel {
                font-family: Arial;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton {
                font-family: Arial;
                font-size: 14px;
                padding: 8px 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        # 加载模型
        config = yaml_config_hook("./config/config.yaml")
        parser = argparse.ArgumentParser()
        for k, v in config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))
        self.args = parser.parse_args()
        self.model, self.device = load_model(self.args)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建布局
        layout = QHBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 左侧布局 - 原始图像
        left_layout = QVBoxLayout()
        self.original_image = ImageLabel("Original Image")
        self.upload_btn = QPushButton("Upload")
        
        left_layout.addWidget(self.original_image.title)
        left_layout.addWidget(self.original_image.image_label)
        left_layout.addWidget(self.upload_btn)
        left_layout.setAlignment(Qt.AlignCenter)
        
        # 中间布局 - 预测值
        middle_layout = QVBoxLayout()
        self.prediction_label = QLabel("Prediction")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_value = QLabel("--")
        self.prediction_value.setAlignment(Qt.AlignCenter)
        self.prediction_value.setStyleSheet("""
            QLabel {
                font-family: Arial;
                font-size: 24px;
                color: #2196F3;
                padding: 10px;
                background-color: #E3F2FD;
                border-radius: 5px;
            }
        """)
        self.detect_btn = QPushButton("Detect")
        self.detect_btn.setEnabled(False)
        
        middle_layout.addWidget(self.prediction_label)
        middle_layout.addWidget(self.prediction_value)
        middle_layout.addWidget(self.detect_btn)
        middle_layout.setAlignment(Qt.AlignCenter)
        
        # 右侧布局 - 重建图像
        right_layout = QVBoxLayout()
        self.recon_image = ImageLabel("Reconstructed Image")
        self.repair_btn = QPushButton("Repair")
        self.repair_btn.setEnabled(False)
        
        right_layout.addWidget(self.recon_image.title)
        right_layout.addWidget(self.recon_image.image_label)
        right_layout.addWidget(self.repair_btn)
        right_layout.setAlignment(Qt.AlignCenter)
        
        # 连接按钮信号
        self.upload_btn.clicked.connect(self.upload_image)
        self.detect_btn.clicked.connect(self.detect_image)
        self.repair_btn.clicked.connect(self.repair_image)
        
        # 将三个布局添加到主布局
        layout.addLayout(left_layout)
        layout.addLayout(middle_layout)
        layout.addLayout(right_layout)
        
        main_widget.setLayout(layout)
        
        # 类变量用于存储临时数据
        self.current_image_tensor = None
        self.current_prediction = None
        self.current_reconstruction = None
    
    def convert_cv_qt(self, cv_img):
        """将OpenCV图像转换为QImage"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)
    
    def upload_image(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", 
                                                   "Images (*.png *.xpm *.jpg *.bmp)")
            if file_name:
                # 使用OpenCV读取并显示原始图片
                original_img = cv2.imread(file_name)
                if original_img is None:
                    print("无法读取图像")
                    return
                    
                # 调整图像大小
                original_img = cv2.resize(original_img, (224, 224))
                # 转换并显示原始图像
                self.original_image.setPixmap(self.convert_cv_qt(original_img))
                
                # 处理图片
                self.current_image_tensor = process_image(file_name)
                if self.current_image_tensor is None:
                    print("图像处理失败")
                    return
                
                # 启用检测按钮
                self.detect_btn.setEnabled(True)
                # 重置其他显示
                self.prediction_value.setText("--")
                self.recon_image.clear()
                self.repair_btn.setEnabled(False)
                
        except Exception as e:
            print(f"上传图片时出错: {str(e)}")
    
    def detect_image(self):
        try:
            if self.current_image_tensor is None:
                return
                
            # 进行预测
            prediction, reconstruction = inference(self.model, self.current_image_tensor, self.device)
            if prediction is None:
                self.prediction_value.setText("预测失败")
                return
            
            # 保存结果
            self.current_prediction = prediction
            self.current_reconstruction = reconstruction
            
            # 显示预测值
            self.prediction_value.setText(f"{prediction:.2f}")
            
            # 启用修复按钮
            self.repair_btn.setEnabled(True)
            
        except Exception as e:
            print(f"检测图像时出错: {str(e)}")
            self.prediction_value.setText("检测失败")
    
    def repair_image(self):
        try:
            if self.current_reconstruction is None:
                return
                
            # 显示重建图像
            recon_np = self.current_reconstruction.squeeze().permute(1, 2, 0).numpy()
            recon_np = np.clip(recon_np * 255, 0, 255).astype(np.uint8)
            print(recon_np)
            recon_bgr = cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR)
            self.recon_image.setPixmap(self.convert_cv_qt(recon_bgr))
            
        except Exception as e:
            print(f"修复图像时出错: {str(e)}")
            self.recon_image.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 