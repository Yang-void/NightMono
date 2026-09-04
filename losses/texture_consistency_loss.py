import torch
import torch.nn as nn
import torch.nn.functional as F

class TextureConsistencyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
        # 确保是float类型
        self.high_percentile = float(cfg.high_percentile if hasattr(cfg, 'high_percentile') else 0.75)
        self.suppress_percentile = float(cfg.suppress_percentile if hasattr(cfg, 'suppress_percentile') else 0.3)
        
        # 初始化sobel算子
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=torch.float32).view(1, 1, 3, 3)
    
    def compute_gradient(self, x):
        """计算图像梯度"""
        # 检查输入
        if torch.isnan(x).any():
            print("Warning: Input contains nan values")
            x = torch.nan_to_num(x, 0.0)
        
        # 确保sobel算子在正确的设备上
        if self.sobel_x.device != x.device:
            self.sobel_x = self.sobel_x.to(x.device)
            self.sobel_y = self.sobel_y.to(x.device)
        
        # 处理多通道输入
        if x.shape[1] > 1:
            sobel_x = self.sobel_x.repeat(x.shape[1], 1, 1, 1)
            sobel_y = self.sobel_y.repeat(x.shape[1], 1, 1, 1)
        else:
            sobel_x = self.sobel_x
            sobel_y = self.sobel_y
        
        # 添加梯度值的范围限制
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])
        
        # 检查卷积结果
        if torch.isnan(grad_x).any() or torch.isnan(grad_y).any():
            print("Warning: Convolution produced nan values")
            grad_x = torch.nan_to_num(grad_x, 0.0)
            grad_y = torch.nan_to_num(grad_y, 0.0)
        
        # 安全的梯度计算
        grad_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        
        # 最后的安全检查
        if torch.isnan(grad_magnitude).any():
            print("Warning: Gradient magnitude contains nan values")
            grad_magnitude = torch.nan_to_num(grad_magnitude, 0.0)
        
        return grad_magnitude
    
    def normalize_gradient(self, gradient, eps=1e-6):
        """归一化梯度，保持每个batch内的相对关系"""
        B = gradient.shape[0]
        gradient_flat = gradient.view(B, -1)
        
        # 检查是否有无效值
        if torch.isnan(gradient_flat).any() or torch.isinf(gradient_flat).any():
            print("Warning: Found nan or inf in gradient before normalization")
        
        gradient_max = gradient_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        gradient_min = gradient_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        
        # 检查是否有相同的最大最小值
        diff = gradient_max - gradient_min
        mask = diff < eps
        if mask.any():
            print(f"Warning: {mask.sum().item()} batches have identical max and min values")
            diff[mask] = eps
        
        # 安全的归一化
        normalized = (gradient - gradient_min) / (diff + eps)
        
        # 最后的安全检查
        if torch.isnan(normalized).any() or torch.isinf(normalized).any():
            print("Warning: Found nan or inf after normalization")
            normalized = torch.clamp(normalized, 0, 1)
        
        return normalized
    
    def forward(self, inputs, outputs):
        texture_features = outputs[('texture', 0)]  # [B, 32, H, W]
        target_size = texture_features.shape[2:]
        device = texture_features.device
        
        # 确保sobel算子在正确的设备上
        if self.sobel_x.device != device:
            self.sobel_x = self.sobel_x.to(device)
            self.sobel_y = self.sobel_y.to(device)
        
        teacher_depth = F.interpolate(
            inputs['pso_depth'].unsqueeze(1),
            size=target_size,
            mode='bilinear',
            align_corners=True
        )
        
        # color_img = inputs[("color", 0)]
        # brightness = 0.299 * color_img[:,0] + 0.587 * color_img[:,1] + 0.114 * color_img[:,2]
        # brightness = F.interpolate(
        #     brightness.unsqueeze(1),
        #     size=target_size,
        #     mode='bilinear',
        #     align_corners=True
        # )
        
        # 计算亮度权重mask
        # with torch.no_grad():
        #     q98 = torch.quantile(brightness, 0.98)
        #     q5 = torch.quantile(brightness, 0.05)
        #     q15 = torch.quantile(brightness, 0.15)
            
        #     dark_thresh = max(0.02, q5.item())
        #     dark_end = max(0.04, q15.item())
            
        #     valid_region = torch.ones((brightness.shape[0], 1, *target_size), 
        #                             dtype=torch.bool, 
        #                             device=device)
            
        #     upper_region = torch.zeros_like(brightness, dtype=torch.bool)
        #     upper_region[:, :, :int(0.4*target_size[0]), :] = True
        #     bright_mask = (brightness >= q98) & upper_region
            
        #     # 使用已经移到正确设备的sobel算子
        #     bright_boundary = (F.conv2d(bright_mask.float(), 
        #                               self.sobel_x.abs() + self.sobel_y.abs(), 
        #                               padding=1) > 0).bool()
        #     valid_region = valid_region & (~bright_boundary)
            
        #     dark_mask = brightness <= dark_end
        #     dark_boundary = (F.conv2d(dark_mask.float(), 
        #                             self.sobel_x.abs() + self.sobel_y.abs(), 
        #                             padding=1) > 0).bool()
        #     valid_region = valid_region & (~dark_boundary)
            
        #     valid_region = (valid_region > 0).bool()
        
        # 在计算梯度之前先获取原始的梯度
        original_depth_grad = self.compute_gradient(teacher_depth)
        original_depth_grad_norm = self.normalize_gradient(original_depth_grad)
        
        # 处理天空区域和亮度
        # sky_depth = 80.0 * torch.ones_like(teacher_depth)
        # teacher_depth = torch.where(valid_region > 0.5, sky_depth, teacher_depth)
        
        # 计算纹理特征的梯度
        texture_grads = []
        for c in range(texture_features.shape[1]):
            channel_feature = texture_features[:, c:c+1]
            channel_grad = self.compute_gradient(channel_feature)
            texture_grads.append(channel_grad)
        
        texture_grad = torch.stack(texture_grads, dim=1).mean(dim=1)
        texture_grad_norm = self.normalize_gradient(texture_grad)
        
        # 在计算loss时使用valid_region来排除处理后的边界
        depth_grad_x = F.conv2d(teacher_depth, self.sobel_x, padding=1)
        depth_grad_y = F.conv2d(teacher_depth, self.sobel_y, padding=1)
        depth_grad_angle = torch.atan2(depth_grad_y, depth_grad_x)
        
        # 确保所有mask都是严格的0/1二值
        horizontal_mask = (torch.abs(depth_grad_angle) < torch.pi / 6).bool()
        depth_thresh = torch.quantile(original_depth_grad_norm.view(-1), self.high_percentile)
        depth_significant = (original_depth_grad_norm > depth_thresh).bool()
        
        # 使用布尔运算并确保结果是严格的0/1二值
        valid_depth_grad = (depth_significant & (~horizontal_mask)).bool()
        
        # 添加这两行用于可视化
        outputs[("valid_region", 0)] = valid_depth_grad.float()
        outputs[("depth_grad_masked", 0)] = original_depth_grad_norm * valid_depth_grad.float()
        
        # 后续计算保持不变
        enhance_loss = self.l1_loss(
            torch.clamp(texture_grad_norm * valid_depth_grad, -1e6, 1e6),
            torch.clamp(original_depth_grad_norm * valid_depth_grad, -1e6, 1e6)
        )
        
        # low_thresh = torch.quantile(original_depth_grad_norm.view(-1), self.suppress_percentile)
        # noise_region = (original_depth_grad_norm < low_thresh) & (~valid_depth_grad)
        # suppress_loss = torch.mean(texture_grad_norm * noise_region)
        
        total_loss = torch.mean(enhance_loss) # + 0.1 * suppress_loss
        
        stats = {
            'valid_ratio': torch.mean(valid_depth_grad.float()),
            # 'noise_ratio': torch.mean(noise_region.float()),
            'mean_enhance_loss': torch.mean(enhance_loss),
            # 'mean_suppress_loss': suppress_loss
        }
        
        return total_loss, stats 