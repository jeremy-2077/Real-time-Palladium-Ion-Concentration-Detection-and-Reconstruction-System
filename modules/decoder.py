import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, in_features=128, out_channels=3):
        super(Decoder, self).__init__()
        
        # 特征向量转换为空间特征图，使用多尺度特征
        self.feature_to_spatial = nn.Sequential(
            nn.Linear(in_features, 1024 * 7 * 7),
            nn.BatchNorm1d(1024 * 7 * 7),
            nn.ReLU(inplace=True)
        )
        
        # 初始特征处理
        self.init_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 解码器主体结构
        self.decoder_blocks = nn.ModuleList([
            ResDecoderBlock(512, 256),    # 7x7 -> 14x14
            ResDecoderBlock(256, 128),    # 14x14 -> 28x28
            ResDecoderBlock(128, 64),     # 28x28 -> 56x56
            ResDecoderBlock(64, 32),      # 56x56 -> 112x112
        ])
        
        # 最终输出层
        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # x shape: [batch_size, in_features]
        batch_size = x.size(0)
        
        # 特征转换
        x = self.feature_to_spatial(x)
        x = x.view(batch_size, 1024, 7, 7)
        x = self.init_conv(x)
        
        # 逐步解码
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)
            
        # 最终输出
        x = self.final_block(x)
        return x

class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDecoderBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            # 上采样路径
            nn.ConvTranspose2d(in_channels, out_channels, 
                             kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # 特征提取
            nn.Conv2d(out_channels, out_channels, 
                     kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差路径
        self.skip_connection = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                             kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.conv_block(x)
        out = out + identity
        out = self.relu(out)
        return out
