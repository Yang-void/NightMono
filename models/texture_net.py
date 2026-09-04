import pytorch_lightning as pl

from functools import partial
from models.md2.layers import disp_to_depth
#from models.md2.resnet_enhancer import ResnetEnhancer
#from models.md2.resnet_decoder import ResnetDecoder
from models.md2.myconv import DecomNet
# from models.md2.unet import UNet
#from models.md2.net import Restormer_Encoder, Restormer_Decoder
#from models.md2.median import median_filter
from data.transforms import NormalizeDynamic
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import time


class TextureNet(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()

        encoder_meta = cfg.MODEL.DEPTH.ENCODER.VERSION.split('-')
        assert encoder_meta[0].lower() in ['resnet']

        # self.unet = UNet(in_channels = 3, num_classes = 3)

        self.gradNet = DecomNet(n_feats=64)

        #DIDF_Encoder = Restormer_Encoder()
        #DIDF_Decoder = Restormer_Decoder()

        # self.normalize = NormalizeDynamic(cfg)

    def forward(self, img, daytime):
        
        # print(img)
        # print(img.min())
        # print(img.max())
        

        # texture = self.unet(img)
        # x = self.normalize(img, daytime)
        # print(img.shape)
        # start = time.perf_counter()
        x, grad = self.gradNet(img)
        # total_time_ms = 1000 * (time.perf_counter() - start)
        # print("Per-image time:", round(total_time_ms, 1), "ms")
        # x = self.apply_texture_strength_to_rgb(img*255, texture)


        # enhanced_detail = img* 5* texture
        # blurred_image = img
        # blurred_image[:, 0:1, :, :] = gaussian_blur(img[:, 0:1, :, :], kernel_size=5, sigma=1.5)
        # blurred_image[:, 1:2, :, :] = gaussian_blur(img[:, 1:2, :, :], kernel_size=5, sigma=1.5)
        # blurred_image[:, 2:3, :, :] = gaussian_blur(img[:, 2:3, :, :], kernel_size=5, sigma=1.5)
        # x = img + texture
        # x = (x - x.min()) / (x.max() - x.min() + 1e-8)


        #x = self.normalize(x, daytime)
        # print(x.shape)
        # print(type(x))
        return x, grad

