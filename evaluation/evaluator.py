# Adapted from https://github.com/TRI-ML/packnet-sfm/blob/c03e4bf929f202ff67819340135c53778d36047f/packnet_sfm/models/model_wrapper.py

import copy

from collections import OrderedDict
from utils.depth import compute_depth_metrics, post_process_inv_depth, inv2depth
from utils.image import flip_lr

import torch

class Evaluator:
    def __init__(self, model, cfg):
        assert cfg.DATASET.LOAD.GT.DEPTH

        self.model = model
        # self.texture_model = texture_model
        self.min_depth = cfg.EVALUATION.DEPTH.MIN_DEPTH
        self.max_depth = cfg.EVALUATION.DEPTH.MAX_DEPTH
        self.temp_context = cfg.DATASET.TEMP_CONTEXT

        # Task metrics
        self.metrics_name = 'depth'
        self.metrics_keys = ('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')
        self.metrics_modes = ('', '_pp', '_gt', '_pp_gt')

        self.metric_conditions = ('all-conditions', 'day', 'night', 'clear', 'rain', 'day-clear', 'day-rain',
                                  'night-clear', 'night-rain') if cfg.EVALUATION.CONDITION_WISE else ('all-conditions',)

        # Dictionary for metrics in different conditions
        self.metrics = OrderedDict({condition: {mode: {metric: 0.0 for metric in self.metrics_keys} for mode in self.metrics_modes} for condition in self.metric_conditions})
        for condition in self.metrics.keys():
            self.metrics[condition]['count'] = 0

    @property
    def metrics_dicts(self):
        return self.metrics

    def reset_metrics(self):
        for condition in self.metric_conditions:
            for mode in self.metrics_modes:
                for metric in self.metrics_keys:
                    self.metrics[condition][mode][metric] = 0.0
            self.metrics[condition]['count'] = 0

    def compute_average_metrics(self):
        intermediate_res = copy.deepcopy(self.metrics)
        for condition in self.metric_conditions:
            for mode in self.metrics_modes:
                for metric in self.metrics_keys:
                    intermediate_res[condition][mode][metric] = self.metrics[condition][mode][metric] / self.metrics[condition]['count'] if self.metrics[condition]['count'] != 0 else 0.0

        return intermediate_res

    def compute_average_metrics_and_export(self):
        results = self.compute_average_metrics()
        export_dict = OrderedDict({condition: {'everything': {}} for condition in self.metric_conditions})
        for condition in self.metric_conditions:
            for mode in self.metrics_modes:
                for metric in self.metrics_keys:
                    export_dict[condition]['everything'][metric + mode] = results[condition][mode][metric]
            export_dict[condition]['everything']['count'] = float(results[condition]['count'])

        return export_dict

    def evaluate_depth(self, batch):
        """
        评估批次数据的深度预测，包含完整的数值检查和错误处理
        """
        def clip_and_check(tensor, name, min_val=1e-7, max_val=1e7):
            """辅助函数：检查并修正异常数值"""
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"警告: {name} 包含 NaN 或 Inf")
                tensor = torch.nan_to_num(tensor, min_val, max_val)
            
            if tensor.max() > max_val or tensor.min() < min_val:
                print(f"警告: {name} 超出正常范围 [{tensor.min().item():.3e}, {tensor.max().item():.3e}]")
                tensor = torch.clamp(tensor, min_val, max_val)
            return tensor

        try:
            # 1. 输入检查
            # if "color" not in batch or "weather" not in batch or "depth_gt" not in batch:
            #     print("错误: 批次数据缺少必要字段")
            #     return None

            # 2. Texture Model 处理
            # images, texture = self.texture_model(batch["color", 0], batch["weather"])
            # print(f"Texture model 输出范围: min={images.min().item():.3e}, max={images.max().item():.3e}")
            # images = clip_and_check(images, "texture_model_output")
            
            # 3. 主模型前向传播
            batch[('color_ad', 0)] = batch[('color', 0)]
            outputs = self.model(batch["color_ad", 0], batch["weather"])
            
            if ("disp", 0, 0) not in outputs:
                print("错误: 模型输出缺少视差数据")
                return None
            
            # 4. 初始深度预测
            depth = outputs[("disp", 0, 0)]
            print(f"初始深度预测范围: min={depth.min().item():.3e}, max={depth.max().item():.3e}")
            depth = clip_and_check(depth, "initial_depth")
            
            # 5. 翻转图像处理
            batch[("color_ad", 0)] = flip_lr(batch[("color_ad", 0)])
            inv_depths_flipped = self.model(batch["color_ad", 0], batch["weather"])[("disp", 0, 0)]
            print(f"翻转后深度预测范围: min={inv_depths_flipped.min().item():.3e}, max={inv_depths_flipped.max().item():.3e}")
            inv_depths_flipped = clip_and_check(inv_depths_flipped, "flipped_depth")
            
            # 6. 后处理
            inv_depth_pp = post_process_inv_depth(depth, inv_depths_flipped, method='mean')
            print(f"后处理深度范围: min={inv_depth_pp.min().item():.3e}, max={inv_depth_pp.max().item():.3e}")
            depth_pp = clip_and_check(inv_depth_pp, "post_processed_depth")
            
            # 恢复原始图像
            batch[("color_ad", 0)] = flip_lr(batch[("color_ad", 0)])
            
            # 7. 地面真值检查
            gt = batch["depth_gt"]
            valid_gt = gt[gt > 0]
            if len(valid_gt) > 0:
                print(f"地面真值深度范围: min={valid_gt.min().item():.3e}, max={gt.max().item():.3e}")
            
            # 8. 指标计算
            metrics_updated = False
            for i, weather in enumerate(batch["weather"]):
                gt_i = batch["depth_gt"][i]
                valid = (gt_i > self.min_depth) & (gt_i < self.max_depth)
                valid_count = valid.sum().item()
                
                if valid_count == 0:
                    print(f"警告: 样本 {i} 没有有效的地面真值深度")
                    continue
                    
                metrics_updated = True
                for metric_condition in self.metric_conditions:
                    if metric_condition in weather:
                        self.metrics[metric_condition]['count'] += 1
                self.metrics['all-conditions']['count'] += 1

            # 9. 计算评估指标
            if metrics_updated:
                for mode in self.metrics_modes:
                    pred = depth_pp if 'pp' in mode else depth
                    compute_depth_metrics(
                        gt=batch["depth_gt"],
                        pred=pred,
                        weather=batch["weather"],
                        metrics=self.metrics,
                        mode=mode,
                        min_depth=self.min_depth,
                        max_depth=self.max_depth,
                        use_gt_scale='gt' in mode
                    )
            
            return outputs

        except Exception as e:
            print(f"评估过程发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_batch(self, batch):
        return self.evaluate_depth(batch)
