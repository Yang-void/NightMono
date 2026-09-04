import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

class BerHuLoss(nn.Module):
    """Class implementing the BerHu loss."""
    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, gt):
        huber_c = torch.max(pred - gt)
        huber_c = self.threshold * huber_c
        diff = (pred - gt).abs()

        huber_mask = (diff > huber_c).detach()
        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2
        return torch.cat((diff, diff2)).mean()


# class SilogLoss(nn.Module):
#     def __init__(self, ratio=10, ratio2=0.85):
#         super().__init__()
#         self.ratio = ratio
#         self.ratio2 = ratio2

#     def forward(self, pred, gt):
#         log_diff = torch.log(pred * self.ratio) - \
#                    torch.log(gt * self.ratio)
#         silog1 = torch.mean(log_diff ** 2)
#         silog2 = self.ratio2 * (log_diff.mean() ** 2)
#         silog_loss = torch.sqrt(silog1 - silog2) * self.ratio
#         return silog_loss
class SILogLoss(nn.Module):
    """SILogloss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None):
        if input.shape[-1] != target.shape[-1]:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode='bilinear', align_corners=True)

        if target.ndim == 3:
            target = target.unsqueeze(1)

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            input = input[mask]
            target = target[mask]

        with autocast(enabled=False):  # amp causes NaNs in this loss function
            alpha = 1e-7
            g = torch.log(input + alpha) - torch.log(target + alpha)
            Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)
            loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        return loss

def get_loss_func(supervised_method):
    if supervised_method.endswith('l1'):
        return nn.L1Loss()
    elif supervised_method.endswith('mse'):
        return nn.MSELoss()
    elif supervised_method.endswith('berhu'):
        return BerHuLoss()
    elif supervised_method.endswith('silog'):
        return SILogLoss()
    elif supervised_method.endswith('abs_rel'):
        return lambda x, y: torch.mean(torch.abs(x - y) / x)
    elif supervised_method.endswith('sasi'):
        return ScaleAndShiftInvariantLoss()
    else:
        raise ValueError('Unknown supervised loss {}'.format(supervised_method))

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        prediction = prediction.squeeze(1)
        target = target.squeeze(1)
        mask = mask.squeeze(1)
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)

class PsoSupervisedLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.supervised_method = cfg.LOSS.PSOSUPERVISED.METHOD
        self.loss_func = get_loss_func(self.supervised_method)
        self.scales = cfg.DATASET.SCALES
        # 为不同尺度设置权重
        self.weights = {0: 1.0, 1: 1, 2: 1, 3: 1}

    def get_diff_weight(self, diff, non_sky_mask):
        """
        根据预测值与真实值的差异生成权重，非天空区域中差异最大的15%区域权重置0
        """
        with torch.no_grad():
            # 初始化权重全1
            weight = torch.ones_like(diff)
            
            # 如果没有非天空区域直接返回
            if non_sky_mask.sum() == 0:
                return weight
            
            # 提取非天空区域的差异值
            non_sky_diff = diff[non_sky_mask]
            
            # 计算差异的85%分位点（保留差异最大的15%）
            q85 = torch.quantile(non_sky_diff, 0.85)
            
            # 创建mask（非天空区域且差异大于阈值）
            invalid_mask = non_sky_mask & (diff > q85)
            
            # 将差异最大的15%区域置0
            weight[invalid_mask] = 0.0
            
            # 打印统计信息
            zero_ratio = invalid_mask.float().mean().item()
            print(f"Invalid region ratio: {zero_ratio:.3f} ({zero_ratio*100:.1f}%)")
            print(f"Diff threshold (q85): {q85.item():.3f}")
        
        return weight

    def calculate_loss(self, inv_depths, pso_gt_depths, pseudo_mask):
        """
        计算多尺度损失，同时保持原有的天空和置信度处理，添加异常处理
        """
        losses = []
        
        for scale in self.scales:
            try:
                # 上采样当前尺度的预测到ground truth尺度
                curr_pred = F.interpolate(inv_depths[scale], 
                                        size=pso_gt_depths.shape[2:], 
                                        mode='bilinear', 
                                        align_corners=False)
                
                basic_mask = (pso_gt_depths > 0) & pseudo_mask
                sky_mask = basic_mask & (self.sky_mask > 0.5)
                non_sky_mask = basic_mask & (self.sky_mask <= 0.5)
                valid_weight_mask = non_sky_mask & (self.confidence_weight > 0)
                
                # 检查是否有有效像素
                if not (valid_weight_mask.sum() > 0 or sky_mask.sum() > 0):
                    print(f"Warning: No valid pixels at scale {scale}, skipping")
                    losses.append(torch.tensor(0.0, device=curr_pred.device))
                    continue
                
                scale_loss = 0
                
                # 非天空区域损失计算
                if valid_weight_mask.sum() > 0:
                    non_sky_loss = self.loss_func(curr_pred, pso_gt_depths, valid_weight_mask)
                    if torch.isnan(non_sky_loss) or torch.isinf(non_sky_loss):
                        print(f"Warning: Invalid non-sky loss at scale {scale}, skipping this part")
                    else:
                        scale_loss += non_sky_loss
                        if scale == 0:
                            print(f"Scale {scale} non-sky loss: {non_sky_loss.item():.4f}")
                
                # 天空区域损失计算
                if sky_mask.sum() > 0:
                    try:
                        sky_diff = torch.abs(curr_pred[sky_mask] - pso_gt_depths[sky_mask])
                        sky_loss = (sky_diff * self.confidence_weight[sky_mask]).mean()
                        if torch.isnan(sky_loss) or torch.isinf(sky_loss):
                            print(f"Warning: Invalid sky loss at scale {scale}, skipping this part")
                        else:
                            scale_loss += sky_loss
                            if scale == 0:
                                print(f"Scale {scale} sky loss: {sky_loss.item():.4f}")
                    except Exception as e:
                        print(f"Error in sky loss calculation at scale {scale}: {e}")
                
                # 应用尺度权重并检查最终损失
                weighted_loss = self.weights[scale] * scale_loss
                if not torch.isnan(weighted_loss) and not torch.isinf(weighted_loss):
                    losses.append(weighted_loss)
                else:
                    print(f"Warning: Invalid weighted loss at scale {scale}, skipping")
                    losses.append(torch.tensor(0.0, device=curr_pred.device))
                
            except Exception as e:
                print(f"Error at scale {scale}: {e}")
                losses.append(torch.tensor(0.0, device=pso_gt_depths.device))
                continue
        
        # 检查是否有有效的损失
        valid_losses = [loss for loss in losses if not torch.isnan(loss) and not torch.isinf(loss)]
        if not valid_losses:
            print("Warning: No valid losses found, returning zero loss")
            return torch.tensor(0.0, device=pso_gt_depths.device, requires_grad=True)
        
        # 计算总损失
        total_loss = sum(valid_losses) / sum(self.weights.values())
        print(f"Total multi-scale loss: {total_loss.item():.4f}")
        
        return total_loss

    def forward(self, inputs, outputs):
        inv_depths = {}
        for scale in self.scales:
            if ("disp", 0, scale) in outputs:
                inv_depths[scale] = outputs[("disp", 0, scale)]
        
        # 获取预测深度和真实深度
        pso_gt_depth = inputs["pso_depth"]
        self.sky_mask = inputs["sky_mask"].clone().detach().to(dtype=torch.float32, device=inv_depths[0].device).unsqueeze(1)
        pso_gt_depth = pso_gt_depth.clone().detach().to(dtype=torch.float32, device=inv_depths[0].device).unsqueeze(1)
        
        # 计算最高分辨率尺度的差异
        pred_depth = inv_depths[0]
        pred_depth = F.interpolate(pred_depth, 
                                 size=pso_gt_depth.shape[2:], 
                                 mode='bilinear', 
                                 align_corners=False)
        diff = torch.abs(pred_depth - pso_gt_depth)
        
        # 生成非天空区域mask
        non_sky_mask = (self.sky_mask <= 0.5)

        sky_depth = 80*torch.ones_like(pso_gt_depth)
        pso_gt_depth = torch.where(self.sky_mask > 0.5, sky_depth, pso_gt_depth)
        
        # 生成差异权重
        diff_weight = self.get_diff_weight(diff, non_sky_mask)
        
        # 设置天空区域权重为0.02
        sky_weight = 0.02 * torch.ones_like(pso_gt_depth)
        self.confidence_weight = torch.where(self.sky_mask > 0.5, sky_weight, diff_weight)
        
        outputs[("weight_mask", 0)] = self.confidence_weight
        outputs[("disp_teacher", 0)] = pso_gt_depth
        
        pseudo_mask = torch.ones_like(pso_gt_depth)
        pseudo_mask = pseudo_mask.bool()
        loss = self.calculate_loss(inv_depths, pso_gt_depth, pseudo_mask)
        return loss
