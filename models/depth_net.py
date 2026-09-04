import pytorch_lightning as pl
import torch

from functools import partial
from models.md2.layers import disp_to_depth
from models.md2.resnet_encoder import ResnetEncoder
from models.md2.depth_decoder import DepthDecoder
from data.transforms import NormalizeDynamic


class DepthNet(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        encoder_meta = cfg.MODEL.DEPTH.ENCODER.VERSION.split('-')
        assert encoder_meta[0].lower() in ['resnet']

        self.encoder = ResnetEncoder(num_layers=int(encoder_meta[1]), pretrained=cfg.MODEL.DEPTH.ENCODER.PRETRAINED)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)
        # self.scale_disp = partial(disp_to_depth, min_depth=cfg.MODEL.DEPTH.MIN_DEPTH, max_depth=cfg.MODEL.DEPTH.MAX_DEPTH)

        self.normalize = NormalizeDynamic(cfg)

    def forward(self, img, daytime):
        # 检查输入
        if torch.isnan(img).any():
            print("NaN detected in input image")
            print(f"Image stats - min: {img.min():.3f}, max: {img.max():.3f}")
        
        x = self.normalize(img, daytime)
        
        # 检查归一化后的输入
        if torch.isnan(x).any():
            print("NaN detected after normalization")
            print(f"Normalized stats - min: {x.min():.3f}, max: {x.max():.3f}")
        
        x = self.encoder(x)
        
        # 检查encoder输出
        if isinstance(x, list):
            for idx, feat in enumerate(x):
                if torch.isnan(feat).any():
                    print(f"NaN detected in encoder output {idx}")
                    print(f"Feature stats - min: {feat.min():.3f}, max: {feat.max():.3f}")
        
        x = self.decoder(x)
        return x # {scale: self.scale_disp(disp)[0] for scale, disp in x.items()}
