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
        self.percentile = 0.90  # 设置90%分位数阈值
        self.weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}  # 这些权重值可以根据需要调整

    # def calculate_loss(self, depths, pso_gt_depths, lidar_mask):
    #     pseudo_mask = ~lidar_mask  # 伪标签区域（无雷达点的区域）
        
    #     # 1. 基础的有效性检查
    #     valid_mask = (pso_gt_depths > 0) & pseudo_mask
        
    #     if not valid_mask.any():
    #         return torch.tensor(0.0, device=depths.device, requires_grad=True)
            
    #     # 2. 使用尺度与偏置无关损失，不需要考虑具体的深度范围
    #     loss = self.loss_func(depths[valid_mask], pso_gt_depths[valid_mask])
        
    #     return loss
    
    def calculate_loss(self, inv_depths, pso_gt_depths, mask):
        losses = []
        
        # 对每个尺度的预测进行上采样到ground truth的尺度
        for scale in self.scales:
            try:
                # 添加预测值检查
                if torch.isnan(inv_depths[scale]).any():
                    print(f"Scale {scale} - NaN in predictions before interpolation")
                    print(f"Pred stats - min: {inv_depths[scale].min():.3f}, max: {inv_depths[scale].max():.3f}")
                    
                curr_pred = F.interpolate(inv_depths[scale], 
                                        size=pso_gt_depths.shape[2:], 
                                        mode='bilinear', 
                                        align_corners=False)
                
                # 添加插值后的检查
                if torch.isnan(curr_pred).any():
                    print(f"Scale {scale} - NaN in predictions after interpolation")
                    print(f"Interpolated pred stats - min: {curr_pred.min():.3f}, max: {curr_pred.max():.3f}")
                
                depth_mask = (pso_gt_depths > 0).bool()
                pseudo_mask = mask.bool()
                basic_mask = depth_mask & pseudo_mask
                
                if torch.sum(basic_mask) == 0:
                    losses.append(torch.tensor(0.1).to(curr_pred.device))
                    continue
                
                # 添加掩码后的值检查
                masked_pred = curr_pred[basic_mask]
                masked_gt = pso_gt_depths[basic_mask]
                if not torch.isnan(masked_pred).any() and not torch.isnan(masked_gt).any():
                    print(f"Scale {scale} - Masked values OK")
                    print(f"Masked pred - min: {masked_pred.min():.3f}, max: {masked_pred.max():.3f}")
                    print(f"Masked GT - min: {masked_gt.min():.3f}, max: {masked_gt.max():.3f}")
                
                # 使用权重
                loss = self.weights[scale] * self.loss_func(curr_pred, pso_gt_depths, basic_mask)
                
                # 检查loss是否有效
                if not torch.isnan(loss) and not torch.isinf(loss):
                    losses.append(loss)
                else:
                    print(f"Warning: Invalid loss at scale {scale}, skipping")
                    print(f"Loss value: {loss.item() if not torch.isnan(loss) else 'NaN'}")
                    
            except Exception as e:
                print(f"Error at scale {scale}: {e}")
                continue
        
        # depth_mask = (pso_gt_depths > 0).bool()
        # pseudo_mask = mask.bool()
        
        # # 组合掩码
        # basic_mask = depth_mask & pseudo_mask
        
        # # 保持4D张量结构 [batch, channel, height, width]
        # valid_gt = pso_gt_depths
        # valid_depths = inv_depths
        
        # if torch.sum(basic_mask) == 0:
        #     return torch.tensor(0.0).to(inv_depths.device)
        
        # # 传递掩码到loss_func进行计算
        # loss = self.loss_func(valid_depths, valid_gt, basic_mask)
        
        # return loss
        
        # 如果没有有效的损失，返回零损失
        if not losses:
            return torch.tensor(0.1).to(pso_gt_depths.device)
            
        return sum(losses) / sum(self.weights.values())

    def forward(self, inputs, outputs):
        # inv_depths = outputs[("disp", 0, 0)]
        # inv_depths = [outputs[("disp", 0, scale)] for scale in self.scales]
        inv_depths = {}
        for scale in self.scales:
            if ("disp", 0, scale) in outputs:
                inv_depths[scale] = outputs[("disp", 0, scale)]
        
        # 获取pso深度图并处理
        pso_gt_depth = inputs["pso_depth"]
        pso_gt_depth = pso_gt_depth.clone().detach().to(dtype=torch.float32, device=inv_depths[0].device).unsqueeze(1)
        
        # 调整预测深度图的大小以匹配目标
        # inv_depths1 = F.interpolate(inv_depths, size=pso_gt_depth.shape[2:], mode='bilinear', align_corners=False)
        
        outputs[("disp_teacher", 0)] = pso_gt_depth
        mask = torch.ones_like(pso_gt_depth) 
        
        # print(f"inv_depths0 shape: {inv_depths[0].shape}")
        # print(f"inv_depths1 shape: {inv_depths[1].shape}")
        # print(f"inv_depths2 shape: {inv_depths[2].shape}")
        # print(f"inv_depths3 shape: {inv_depths[3].shape}")
        # print(f"pso_gt_depth shape: {pso_gt_depth.shape}")

        loss = self.calculate_loss(inv_depths, pso_gt_depth, mask)
        return loss
