import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image import interpolate_scales, match_scales
from models.md2.myconv import GetGradientNoPaddingRGB

# def inv_depths_normalize(inv_depths):
#     """
#     Inverse depth normalization

#     Parameters
#     ----------
#     inv_depths : list of torch.Tensor [B,1,H,W]
#         Inverse depth maps

#     Returns
#     -------
#     norm_inv_depths : list of torch.Tensor [B,1,H,W]
#         Normalized inverse depth maps
#     """
#     mean_inv_depths = [inv_depth.mean(2, True).mean(3, True) for inv_depth in inv_depths]
#     return [inv_depth / mean_inv_depth.clamp(min=1e-6)
#             for inv_depth, mean_inv_depth in zip(inv_depths, mean_inv_depths)]


def gradient_x(image):
    """
    Calculates the gradient of an image in the x dimension
    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Input image

    Returns
    -------
    gradient_x : torch.Tensor [B,3,H,W-1]
        Gradient of image with respect to x
    """
    padded_image = F.pad(image, (0, 1, 0, 0), mode='constant', value=0)
    return padded_image[:, :, :, :-1] - padded_image[:, :, :, 1:]


def gradient_y(image):
    """
    Calculates the gradient of an image in the y dimension
    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Input image

    Returns
    -------
    gradient_y : torch.Tensor [B,3,H-1,W]
        Gradient of image with respect to y
    """
    
    padded_image = F.pad(image, (0, 0, 0, 1), mode='constant', value=0)
    return padded_image[:, :, :-1, :] - padded_image[:, :, 1:, :]

def calculate_grayscale_image(image_tensor):
    # 将彩色图像转换为灰度图像
    grayscale_image = 0.2989 * image_tensor[:, 0, :, :] + 0.5870 * image_tensor[:, 1, :, :] + 0.1140 * image_tensor[:, 2, :, :]
    gray_expanded = grayscale_image.unsqueeze(1)
    return gray_expanded

def dynamic_rate_from_gray(gray_image, min_brightness = 0, max_brightness=1, min_rate=0.05, max_rate=1.0):
    # 计算灰度图像的平均亮度
    brightness = torch.mean(gray_image)
    
    # 映射亮度到比率范围内
    rate = 0.7*(brightness - min_brightness) / (max_brightness - min_brightness)
    rate = torch.clamp(rate, min_rate, max_rate)  # 确保比率在[min_rate, max_rate]范围内
    return rate

def fft2(x, rate):
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

class TextureLoss(nn.Module):
    def __init__(self, cfg):
        super(TextureLoss, self).__init__()

        self.grad_rgb = GetGradientNoPaddingRGB()

    @staticmethod
    def calc_gradient(self, images):
        """
        Calculate smoothness values for inverse depths

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Inverse depth maps
        images : list of torch.Tensor [B,3,H,W]
            Inverse depth maps
        scales : list
            Scales considered

        Returns
        -------
        smoothness_x : list of torch.Tensor [B,1,H,W]
            Smoothness values in direction x
        smoothness_y : list of torch.Tensor [B,1,H,W]
            Smoothness values in direction y
        """

        # image_gradients_x = gradient_x(images)
        # image_gradients_y = gradient_y(images)
        
        # gradient = image_gradients_x + image_gradients_y
        # # print(gradient.shape)
        # # median_scalar = torch.median(gradient)
        # min_val = torch.min(gradient.view(gradient.size(0), gradient.size(1), -1), dim=2, keepdim=True)[0].view(gradient.size(0), gradient.size(1), 1, 1)
        # max_val = torch.max(gradient.view(gradient.size(0), gradient.size(1), -1), dim=2, keepdim=True)[0].view(gradient.size(0), gradient.size(1), 1, 1)
        # gradient_normalized = (gradient - min_val) / (max_val - min_val + 1e-8)
        # # print(gradient_normalized.shape)
        # gradient = gradient/median_scalar
        # weights_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_x]
        # weights_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in image_gradients_y]
        gradient = self.grad_rgb(images)
        # # Note: Fix gradient addition
        # smoothness_x = [inv_depth_gradients_x[i] * weights_x[i] for i in scales]
        # smoothness_y = [inv_depth_gradients_y[i] * weights_y[i] for i in scales]
        return gradient

    def calc_texture_loss(self, texture, images):
        """
        Calculates the smoothness loss for inverse depth maps.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted inverse depth maps for all scales
        images : list of torch.Tensor [B,3,H,W]
            Original images for all scales

        Returns
        -------
        smoothness_loss : torch.Tensor [1]
            Smoothness loss
        """
        # Calculate smoothness gradients
        gradient = self.grad_rgb(images)
        # gray = calculate_grayscale_image(images)
        # rate = dynamic_rate_from_gray(gray)
        # gradient = fft2(images, rate)
        
        # print(gradient.shape)
        # Calculate smoothness loss
        # smoothness_loss = sum([(smoothness_x[i].abs().mean() +
        #                         smoothness_y[i].abs().mean()) / 2 ** i
        #                        for i in self.scales]) / len(self.scales)
        # Return smoothness loss
        texture_loss = torch.abs(gradient - texture).mean()
        # print(texture.shape)
        return texture_loss

    def forward(self, inputs):
        texture = inputs[('texture', 0)]
        images = inputs[("color", 0)]

        texture_loss = self.calc_texture_loss(texture, images)
        # print(type(texture_loss))
        return texture_loss
