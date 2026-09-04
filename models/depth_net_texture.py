import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from models.md2.layers import disp_to_depth
from models.md2.resnet_encoder import ResnetEncoder
from models.md2.depth_decoder import DepthDecoder
from data.transforms import NormalizeDynamic

class TextureFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        mid_channels = out_channels // 2
        
        # 多尺度纹理特征提取
        self.conv_layers = nn.ModuleList([
            # 小尺度纹理 (3x3)
            nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            ),
            # 中尺度纹理 (5x5)
            nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 方向感知层
        self.direction_layers = nn.ModuleList([
            # 水平方向
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 5), padding=(0, 2)),
            # 垂直方向
            nn.Conv2d(in_channels, mid_channels, kernel_size=(5, 1), padding=(2, 0))
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 最终的纹理增强
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 可选：添加SE模块进行通道重标定
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        
        scale_features = [conv(x) for conv in self.conv_layers]
        # print(f"Scale features shapes: {[f.shape for f in scale_features]}")
        
        direction_features = [conv(x) for conv in self.direction_layers]
        # print(f"Direction features shapes: {[f.shape for f in direction_features]}")
        
        combined = torch.cat(scale_features + direction_features, dim=1)
        # print(f"Combined shape: {combined.shape}")
        
        features = self.fusion(combined)
        # print(f"After fusion shape: {features.shape}")
        
        attention = self.se(features)
        # print(f"Attention shape: {attention.shape}")
        
        features = features * attention
        # print(f"Final output shape: {features.shape}")
        
        return features

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class FeatureInjectionModule(nn.Module):
    def __init__(self, texture_channels, depth_channels):
        super().__init__()
        # 特征融合
        self.conv_reduce = nn.Conv2d(texture_channels + depth_channels, depth_channels, 1)
        
        # CBAM风格的注意力机制
        self.channel_attention = ChannelAttention(depth_channels)
        self.spatial_attention = SpatialAttention()
        
        # 最终融合
        self.fusion = nn.Sequential(
            nn.Conv2d(depth_channels, depth_channels, 3, padding=1),
            nn.BatchNorm2d(depth_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, texture_feat, depth_feat):
        # 调整纹理特征到当前尺度
        if texture_feat.shape[2:] != depth_feat.shape[2:]:
            texture_feat = F.interpolate(
                texture_feat, 
                size=depth_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # 初步融合特征
        combined = self.conv_reduce(torch.cat([texture_feat, depth_feat], dim=1))
        
        # 应用通道注意力
        channel_weight = self.channel_attention(combined)
        combined = combined * channel_weight
        
        # 应用空间注意力
        spatial_weight = self.spatial_attention(combined)
        combined = combined * spatial_weight
        
        # 最终融合
        return self.fusion(combined) + depth_feat  # 添加残差连接

class TextureAwareDepthNet(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        # 基础组件
        encoder_meta = cfg.MODEL.DEPTH.ENCODER.VERSION.split('-')
        assert encoder_meta[0].lower() in ['resnet']
        
        self.encoder = ResnetEncoder(
            num_layers=int(encoder_meta[1]), 
            pretrained=cfg.MODEL.DEPTH.ENCODER.PRETRAINED
        )
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        self.normalize = NormalizeDynamic(cfg)
        
        # 纹理感知组件
        self.texture_extractor = TextureFeatureExtractor(in_channels=3, out_channels=32)
        
        # 只在关键层级注入纹理特征，减少计算开销
        self.injection_modules = nn.ModuleList([
            FeatureInjectionModule(32, self.encoder.num_ch_enc[0]),  # 浅层
            FeatureInjectionModule(32, self.encoder.num_ch_enc[2])   # 中层
        ])

    def forward(self, img, daytime):
        # 标准化
        x = self.normalize(img, daytime)
        
        # 提取纹理特征
        texture_features = self.texture_extractor(x)
        
        # 编码器特征
        encoder_features = self.encoder(x)
        
        # 注入纹理特征
        enhanced_features = list(encoder_features)
        injection_layers = [0, 2]
        
        for idx, layer_idx in enumerate(injection_layers):
            enhanced_features[layer_idx] = self.injection_modules[idx](
                texture_features, 
                encoder_features[layer_idx]
            )
        
        # 解码得到深度图
        outputs = self.decoder(enhanced_features)
        
        # 添加纹理特征到输出
        outputs[('texture', 0)] = texture_features
        
        return outputs 