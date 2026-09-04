import torch.nn as nn
import torch
from torchvision import transforms

# from models.modules import InvertibleConv1x1
# from models.md2.network_dncnn import DnCNN
import torch.nn.init as init
import torch.nn.functional as F
# import torch.fft as fft

class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):  #__init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img, mean_c):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        # mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_map

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class GetGradientNoPaddingRGB(nn.Module):
    def __init__(self, in_channels=3, device='cuda'):
        super(GetGradientNoPaddingRGB, self).__init__()
        self.device = device
        self.weight_v = nn.Parameter(torch.tensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device), requires_grad=False)
        self.weight_h = nn.Parameter(torch.tensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device), requires_grad=False)
        self.conv_v = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False).to(device)
        self.conv_h = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False).to(device)
        with torch.no_grad():
            self.conv_v.weight.copy_(self.weight_v.repeat(1, in_channels, 1, 1))
            self.conv_h.weight.copy_(self.weight_h.repeat(1, in_channels, 1, 1))
            self.conv_v.weight.requires_grad = False
            self.conv_h.weight.requires_grad = False

        # 冻结整个卷积层的参数
        for param in self.conv_v.parameters():
            param.requires_grad = False

        for param in self.conv_h.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_v = self.conv_v(x)
        x_h = self.conv_h(x)
        x = torch.sqrt(x_v.pow(2) + x_h.pow(2) + 1e-6)
        return x

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
 

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class Denois_net(nn.Module):
    def __init__(self):
        super(Denois_net, self).__init__()
        self.relu = nn.PReLU()
        number_f = 64
        self.conv1 = nn.Conv2d(1, number_f, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv1_1 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv1_2 = nn.Conv2d(number_f, 3, 1, 1, 0, bias=True)
        self.norm64 = nn.BatchNorm2d(number_f, affine=True)
        self.norm = nn.BatchNorm2d(3, affine=True)
        
    def forward(self,x):
        x1 = self.norm64(self.relu(self.conv1(x)))
        # residual
        x1_1 = self.relu(self.conv1_1(x1))
        x1_2 = self.relu(self.conv1_1(x1_1))
        x1_3 = self.relu(self.conv1_2(x1 + x1_2))

        x2 = self.norm64(self.relu(self.conv2(x1)))
        x2_1 =  self.relu(self.conv1_1(x2))
        x2_2 =  self.relu(self.conv1_1(x2_1))
        x2_3 =  self.relu(self.conv1_2(x2_2 + x2))

        x3 = self.norm64(self.relu(self.conv2(x2)))
        x3_1 =  self.relu(self.conv1_1(x3))
        x3_2 =  self.relu(self.conv1_1(x3_1))
        x3_3 =  self.relu(self.conv1_2(x3_2 + x3))

        x4 = self.norm64(self.relu(self.conv2(x3)))
        x4_1 = self.relu(self.conv1_1(x4))
        x4_2 = self.relu(self.conv1_1(x4_1))
        x4_3 = self.relu(self.conv1_2(x4_2 + x4))

        x5 = self.norm64(self.relu(self.conv2(x4)))
        x5_1 = self.relu(self.conv1_1(x5))
        x5_2 = self.relu(self.conv1_1(x5_1))
        x5_3 = self.relu(self.conv1_2(x5_2 + x5))

        x_out = x1_3 + x3_3 + x2_3+ x4_3 + x5_3

        return x_out

class DecomNet(nn.Module):
    def __init__(self,n_feats):
        super(DecomNet, self).__init__()

        self.estimator = Illumination_Estimator(n_feats)
        self.grad_rgb = GetGradientNoPaddingRGB()
        self.denoiser = Denois_net()
        # self.dncnn = DnCNN(in_nc=3, out_nc=3)
        # self.fuse_process = nn.Sequential(
        #     nn.Conv2d(2 * 3, n_feats, 1, 1, 0),
        #     SELayer(n_feats),  # 使用SE模块替换残差块
        #     SELayer(n_feats)
        # )
        self.fuse_process = nn.Sequential(nn.Conv2d(2*3, n_feats, 1, 1, 0),
                                          ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=2),
                                          ResidualGroup(default_conv, n_feats, 3, reduction=8, n_resblocks=2))
        self.conv1_2 = nn.Conv2d(n_feats, 3, 1, 1, 0, bias=True)
        
    def calculate_grayscale_image(self, image_tensor):
        # 将彩色图像转换为灰度图像
        grayscale_image = 0.2989 * image_tensor[:, 0, :, :] + 0.5870 * image_tensor[:, 1, :, :] + 0.1140 * image_tensor[:, 2, :, :]
        gray_expanded = grayscale_image.unsqueeze(1)
        return gray_expanded

    def replace_blue_channel_with_gray(self, rgb_image):
        # 计算灰度图像
        gray = self.calculate_grayscale_image(rgb_image)
        # 替换蓝色通道为灰度图像
        rgb_image[:, 2:3, :, :] = gray
        return rgb_image

    def dynamic_rate_from_gray(self, gray_image, min_brightness = 0, max_brightness=1, min_rate=0.3, max_rate=1.0):
        # 计算灰度图像的平均亮度
        brightness = torch.mean(gray_image)
        
        # 映射亮度到比率范围内
        rate = (brightness - min_brightness) / (max_brightness - min_brightness)
        rate = torch.clamp(rate, min_rate, max_rate)  # 确保比率在[min_rate, max_rate]范围内
        return rate
    
    def fft1(self, x, rate):
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        fft = fft * mask
        fr = fft.real
        fi = fft.imag
        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real
        inv = torch.abs(inv)
        return inv

    def fft2(self, x, rate):
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        fft = fft * (1 - mask)
        fr = fft.real
        fi = fft.imag
        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real
        inv = torch.abs(inv)
        return inv
    
    def forward(self, rgb):

        #depth = self.upBlock(depth)
        # print(rgb.shape)
        intensity = self.calculate_grayscale_image(rgb)
        # rgb_new = self.replace_blue_channel_with_gray(rgb)
        rate = self.dynamic_rate_from_gray(intensity)

        rgb_low = self.fft1(rgb, rate)
        # rgb_high = self.fft2(rgb_new, rate)
        rgb_high = self.grad_rgb(rgb)
        # illu_map = self.estimator(rgb_low, intensity)
        
        # enhanced_low = rgb_low * illu_map + rgb_low
        denoised_high = self.denoiser(rgb_high)
        concated_tensor = torch.cat((rgb_low, denoised_high), dim = 1)
        fused_feature = self.fuse_process(concated_tensor)
        rgb_out = self.conv1_2(fused_feature)
        # rgb_out = rgb_low + denoised_high

        return rgb_out, denoised_high