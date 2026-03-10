"""
概念引导剪枝实验类
继承自 Exp_Prune，使用概念重要性替代Taylor重要性
"""

import math
import pickle
import torch
from pathlib import Path

from exp.exp_prune import Exp_Prune
from layers.prune_mask import MaskedLayer


class Exp_ConceptPrune(Exp_Prune):
    """使用概念引导重要性的剪枝实验"""
    
    def __init__(self, args):
        super().__init__(args)
        
        # 根据模型和数据集动态构建重要性文件路径
        dataset_name = args.data_path.replace('.csv', '').lower()
        
        # 构建模型目录名 (与 src/pruning/*/compute_*_importance.py 的保存路径一致)
        if args.model == 'TimeMoE':
            model_dir = 'TimeMoe-50M'
            importance_file = f'timemoe_{dataset_name}_importance.pkl'
        elif args.model == 'Chronos':
            model_dir = 'Chronos_bolt_small'
            importance_file = f'chronos_{dataset_name}_importance.pkl'
        elif args.model == 'TTM':
            model_dir = 'ttm'
            importance_file = f'ttm_{dataset_name}_importance.pkl'
        elif args.model == 'TimerXL':
            model_dir = 'timerxl'
            importance_file = f'timerxl_{dataset_name}_importance.pkl'
        elif args.model == 'moirai':
            model_dir = 'moirai'
            importance_file = f'moirai_{dataset_name}_importance.pkl'
        else:
            # 默认路径
            model_dir = args.model.lower()
            importance_file = f'{args.model.lower()}_{dataset_name}_importance.pkl'
        
        importance_path = Path(f'./pruning_importance/{model_dir}/{importance_file}')
        
        print("\n" + "="*60)
        print("概念引导剪枝初始化")
        print("="*60)
        print(f"  模型: {args.model} ({args.model_size if hasattr(args, 'model_size') else 'default'})")
        print(f"  数据集: {args.data_path}")
        print(f"  重要性文件: {importance_path}")
        
        if not importance_path.exists():
            raise FileNotFoundError(
                f"\n概念重要性文件不存在: {importance_path}\n"
                f"请先运行:\n"
                f"  bash scripts/{model_dir.lower().replace('-', '_')}/04_compute_{args.model.lower()}_concept_importance.sh"
            )
        
        with open(importance_path, 'rb') as f:
            self.concept_importance = pickle.load(f)
        
        # Apply minimum floor to prevent 0-importance modules from being
        # permanently targeted every single prune step (causes catastrophic damage)
        min_floor = 1e-3
        self.concept_importance = {k: max(float(v), min_floor) for k, v in self.concept_importance.items()}
        
        print(f"\n✓ 成功加载概念引导重要性")
        print(f"  模块数量: {len(self.concept_importance)}")
        
        # 显示重要性范围
        scores = list(self.concept_importance.values())
        print(f"  分数范围: [{min(scores):.6f}, {max(scores):.6f}]")
        
        # Top-5 重要模块
        sorted_modules = sorted(
            self.concept_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        print(f"\n  Top-5 重要模块:")
        for name, score in sorted_modules[:5]:
            print(f"    {name:50s}: {score:.6f}")
        
        # Bottom-5 不重要模块
        print(f"\n  Bottom-5 不重要模块（优先剪枝）:")
        for name, score in sorted_modules[-5:]:
            print(f"    {name:50s}: {score:.6f}")
        print("="*60)
        # Accumulator for fractional prune counts (prevents ceil() from over-pruning)
        self._prune_accumulated = 0.0
    
    def _map_module_name(self, target_name):
        """
        映射模块名: 从剪枝代码的命名到概念映射的命名
        
        Args:
            target_name: 例如 'backbone.encoder.block.0.layer.0.SelfAttention.q.mask_in'
        
        Returns:
            mapped_name: 例如 'encoder.block.0'
        """
        # 移除 .mask_in 和 .mask_out 后缀
        base_name = target_name.replace('.mask_in', '').replace('.mask_out', '')
        
        # 移除 'backbone.' 前缀（如果存在）
        if base_name.startswith('backbone.'):
            base_name = base_name[len('backbone.'):]
        
        # 提取核心层名 (encoder.block.X 或 decoder.block.X)
        parts = base_name.split('.')
        
        if len(parts) >= 3:
            # 格式: encoder.block.0.layer.0...
            # 提取: encoder.block.0
            if parts[0] in ['encoder', 'decoder'] and parts[1] == 'block':
                return f'{parts[0]}.{parts[1]}.{parts[2]}'
        
        # 如果无法映射，返回原始名称
        return base_name
    
    def _find_concept_importance(self, module_name):
        """
        查找模块的概念重要性
        
        Args:
            module_name: 模块名
        
        Returns:
            importance: 概念重要性分数，找不到返回0.5（中等）
        """
        # 精确匹配
        if module_name in self.concept_importance:
            return self.concept_importance[module_name]
        
        # 模糊匹配：包含关系
        for concept_key, importance in self.concept_importance.items():
            if concept_key in module_name or module_name in concept_key:
                return importance
        
        # 未找到，返回中等重要性
        print(f"⚠️ 未找到匹配: {module_name}，使用默认重要性0.5")
        return 0.5
    
    @torch.no_grad()
    def _prune(self, imps, unused_names):
        """
        使用概念引导重要性进行剪枝
        
        Args:
            imps: Taylor重要性字典（在本类中不使用）
            unused_names: 未使用的模块名集合
        """
        # 收集所有模块的概念重要性
        module_scores = {}
        
        for target_name in self.target_module_names:
            # 跳过未使用的模块
            mask_in_name = target_name + '.mask_in'
            mask_out_name = target_name + '.mask_out'
            
            if mask_in_name in unused_names and mask_out_name in unused_names:
                continue
            
            # 映射模块名
            mapped_name = self._map_module_name(target_name)
            
            # 获取概念重要性
            importance = self._find_concept_importance(mapped_name)
            
            # 为 mask_in 和 mask_out 都记录重要性（跳过已全部剪除的掩码）
            module = self._model.get_submodule(target_name)
            if getattr(self.args, 'mask_in', True) and mask_in_name not in unused_names:
                if module.mask_in.mask.sum() > 0:
                    module_scores[mask_in_name] = importance
            if getattr(self.args, 'mask_out', True) and mask_out_name not in unused_names:
                if module.mask_out.mask.sum() > 0:
                    module_scores[mask_out_name] = importance
        
        if not module_scores:
            print("⚠️ 没有可剪枝的模块")
            return
        
        # Use accumulative counting instead of ceil()
        if not hasattr(self, '_prune_accumulated'):
            self._prune_accumulated = 0.0
        
        self._prune_accumulated += len(module_scores) * self.prune_ratio
        prune_num = int(self._prune_accumulated)
        self._prune_accumulated -= prune_num
        
        if prune_num == 0:
            return
        
        # 严格按 score 升序排列，只取最低的 prune_num 个
        # 避免 threshold-based 方式在 ties 时一次性剪掉多于 prune_num 的模块
        # (例如: 多个模块分数相同时全被选中，导致灾难性破坏)
        sorted_items = sorted(module_scores.items(), key=lambda x: x[1])
        modules_to_prune = sorted_items[:prune_num]
        actual_threshold = modules_to_prune[-1][1] if modules_to_prune else 0.0
        
        pruned_count = 0
        for module_name, score in modules_to_prune:
            # 获取模块
            base_name = '.'.join(module_name.split('.')[:-1])
            module = self._model.get_submodule(base_name)
            
            # 剪枝
            if module_name.endswith('.mask_in'):
                module.mask_in.mask.data.zero_()
                pruned_count += module.mask_in.mask.numel()
            elif module_name.endswith('.mask_out'):
                module.mask_out.mask.data.zero_()
                pruned_count += module.mask_out.mask.numel()
        
        # 打印剪枝信息
        print(f"\n[概念引导剪枝] 阈值={actual_threshold:.6f}, "
              f"剪枝模块数={prune_num}/{len(module_scores)}, "
              f"剪枝参数数={pruned_count}")


if __name__ == '__main__':
    # 测试代码
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Chronos')
    parser.add_argument('--model_size', type=str, default='small')
    parser.add_argument('--prune_ratio', type=float, default=0.02)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--mask_in', type=bool, default=True)
    parser.add_argument('--mask_out', type=bool, default=True)
    
    args = parser.parse_args([])
    
    print("测试概念引导剪枝类初始化...")
    try:
        exp = Exp_ConceptPrune(args)
        print("\n✓ 初始化成功!")
    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")


# """
# 概念引导剪枝实验类
# 继承自 Exp_Prune，使用概念重要性替代Taylor重要性
# """

# import math
# import pickle
# import torch
# from pathlib import Path

# from exp.exp_prune import Exp_Prune
# from layers.prune_mask import MaskedLayer


# class Exp_ConceptPrune(Exp_Prune):
#     """使用概念引导重要性的剪枝实验"""
    
#     def __init__(self, args):
#         super().__init__(args)
        
#         # 根据模型和数据集动态构建重要性文件路径
#         dataset_name = args.data_path.replace('.csv', '').lower()
        
#         # 构建模型目录名 (与 src/pruning/*/compute_*_importance.py 的保存路径一致)
#         if args.model == 'TimeMoE':
#             model_dir = 'TimeMoe-50M'
#             importance_file = f'timemoe_{dataset_name}_importance.pkl'
#         elif args.model == 'Chronos':
#             model_dir = 'Chronos_bolt_small'
#             importance_file = f'chronos_{dataset_name}_importance.pkl'
#         elif args.model == 'TTM':
#             model_dir = 'ttm'
#             importance_file = f'ttm_{dataset_name}_importance.pkl'
#         elif args.model == 'TimerXL':
#             model_dir = 'timerxl'
#             importance_file = f'timerxl_{dataset_name}_importance.pkl'
#         elif args.model == 'moirai':
#             model_dir = 'moirai'
#             importance_file = f'moirai_{dataset_name}_importance.pkl'
#         else:
#             # 默认路径
#             model_dir = args.model.lower()
#             importance_file = f'{args.model.lower()}_{dataset_name}_importance.pkl'
        
#         importance_path = Path(f'./pruning_importance/{model_dir}/{importance_file}')
        
#         print("\n" + "="*60)
#         print("概念引导剪枝初始化")
#         print("="*60)
#         print(f"  模型: {args.model} ({args.model_size if hasattr(args, 'model_size') else 'default'})")
#         print(f"  数据集: {args.data_path}")
#         print(f"  重要性文件: {importance_path}")
        
#         if not importance_path.exists():
#             raise FileNotFoundError(
#                 f"\n概念重要性文件不存在: {importance_path}\n"
#                 f"请先运行:\n"
#                 f"  bash scripts/{model_dir.lower().replace('-', '_')}/04_compute_{args.model.lower()}_concept_importance.sh"
#             )
        
#         with open(importance_path, 'rb') as f:
#             self.concept_importance = pickle.load(f)
        
#         print(f"\n✓ 成功加载概念引导重要性")
#         print(f"  模块数量: {len(self.concept_importance)}")
        
#         # 显示重要性范围
#         scores = list(self.concept_importance.values())
#         print(f"  分数范围: [{min(scores):.6f}, {max(scores):.6f}]")
        
#         # Top-5 重要模块
#         sorted_modules = sorted(
#             self.concept_importance.items(), 
#             key=lambda x: x[1], 
#             reverse=True
#         )
#         print(f"\n  Top-5 重要模块:")
#         for name, score in sorted_modules[:5]:
#             print(f"    {name:50s}: {score:.6f}")
        
#         # Bottom-5 不重要模块
#         print(f"\n  Bottom-5 不重要模块（优先剪枝）:")
#         for name, score in sorted_modules[-5:]:
#             print(f"    {name:50s}: {score:.6f}")
#         print("="*60)
    
#     def _map_module_name(self, target_name):
#         """
#         映射模块名: 从剪枝代码的命名到概念映射的命名
        
#         Args:
#             target_name: 例如 'backbone.encoder.block.0.layer.0.SelfAttention.q.mask_in'
        
#         Returns:
#             mapped_name: 例如 'encoder.block.0'
#         """
#         # 移除 .mask_in 和 .mask_out 后缀
#         base_name = target_name.replace('.mask_in', '').replace('.mask_out', '')
        
#         # 移除 'backbone.' 前缀（如果存在）
#         if base_name.startswith('backbone.'):
#             base_name = base_name[len('backbone.'):]
        
#         # 提取核心层名 (encoder.block.X 或 decoder.block.X)
#         parts = base_name.split('.')
        
#         if len(parts) >= 3:
#             # 格式: encoder.block.0.layer.0...
#             # 提取: encoder.block.0
#             if parts[0] in ['encoder', 'decoder'] and parts[1] == 'block':
#                 return f'{parts[0]}.{parts[1]}.{parts[2]}'
        
#         # 如果无法映射，返回原始名称
#         return base_name
    
#     def _find_concept_importance(self, module_name):
#         """
#         查找模块的概念重要性
        
#         Args:
#             module_name: 模块名
        
#         Returns:
#             importance: 概念重要性分数，找不到返回0.5（中等）
#         """
#         # 精确匹配
#         if module_name in self.concept_importance:
#             return self.concept_importance[module_name]
        
#         # 模糊匹配：包含关系
#         for concept_key, importance in self.concept_importance.items():
#             if concept_key in module_name or module_name in concept_key:
#                 return importance
        
#         # 未找到，返回中等重要性
#         print(f"⚠️ 未找到匹配: {module_name}，使用默认重要性0.5")
#         return 0.5
    
#     @torch.no_grad()
#     def _prune(self, imps, unused_names):
#         """
#         使用概念引导重要性进行剪枝
        
#         Args:
#             imps: Taylor重要性字典（在本类中不使用）
#             unused_names: 未使用的模块名集合
#         """
#         # 收集所有模块的概念重要性
#         module_scores = {}
        
#         for target_name in self.target_module_names:
#             # 跳过未使用的模块
#             mask_in_name = target_name + '.mask_in'
#             mask_out_name = target_name + '.mask_out'
            
#             if mask_in_name in unused_names and mask_out_name in unused_names:
#                 continue
            
#             # 映射模块名
#             mapped_name = self._map_module_name(target_name)
            
#             # 获取概念重要性
#             importance = self._find_concept_importance(mapped_name)
            
#             # 为 mask_in 和 mask_out 都记录重要性
#             if getattr(self.args, 'mask_in', True) and mask_in_name not in unused_names:
#                 module_scores[mask_in_name] = importance
#             if getattr(self.args, 'mask_out', True) and mask_out_name not in unused_names:
#                 module_scores[mask_out_name] = importance
        
#         if not module_scores:
#             print("⚠️ 没有可剪枝的模块")
#             return
        
#         # 转换为tensor
#         scores = torch.tensor(list(module_scores.values()))
        
#         # 使用父类在 prune() 中已计算好的 per-batch 剪枝率
#         # self.prune_ratio = prune_ratio_per_epoch / len(train_loader) * grad_accumulation
#         # 这样一个 epoch 累计剪掉的比例恰好等于 prune_ratio_per_epoch
#         target_ratio = self.prune_ratio
#         prune_num = math.ceil(len(scores) * target_ratio)
        
#         if prune_num == 0:
#             return
        
#         # 找到剪枝阈值（低于阈值的会被剪掉）
#         if len(scores) > prune_num:
#             threshold = torch.topk(scores, prune_num, largest=False)[0][-1]
#         else:
#             threshold = scores.max()
        
#         # 多GPU同步阈值
#         if self.args.use_multi_gpu:
#             import torch.distributed as dist
#             dist.barrier()
#             dist.all_reduce(threshold, op=dist.ReduceOp.MIN)
        
#         # 执行剪枝
#         pruned_count = 0
#         for module_name, score in module_scores.items():
#             if score <= threshold:
#                 # 获取模块
#                 base_name = '.'.join(module_name.split('.')[:-1])
#                 module = self._model.get_submodule(base_name)
                
#                 # 剪枝
#                 if module_name.endswith('.mask_in'):
#                     module.mask_in.mask.data.zero_()
#                     pruned_count += module.mask_in.mask.numel()
#                 elif module_name.endswith('.mask_out'):
#                     module.mask_out.mask.data.zero_()
#                     pruned_count += module.mask_out.mask.numel()
        
#         # 打印剪枝信息
#         print(f"\n[概念引导剪枝] 阈值={threshold:.6f}, "
#               f"剪枝模块数={prune_num}/{len(module_scores)}, "
#               f"剪枝参数数={pruned_count}")


# if __name__ == '__main__':
#     # 测试代码
#     import argparse
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=str, default='Chronos')
#     parser.add_argument('--model_size', type=str, default='small')
#     parser.add_argument('--prune_ratio', type=float, default=0.02)
#     parser.add_argument('--use_multi_gpu', type=bool, default=False)
#     parser.add_argument('--mask_in', type=bool, default=True)
#     parser.add_argument('--mask_out', type=bool, default=True)
    
#     args = parser.parse_args([])
    
#     print("测试概念引导剪枝类初始化...")
#     try:
#         exp = Exp_ConceptPrune(args)
#         print("\n✓ 初始化成功!")
#     except Exception as e:
#         print(f"\n✗ 初始化失败: {e}")


# # """
# # 概念引导剪枝实验类
# # 继承自 Exp_Prune，使用概念重要性替代Taylor重要性
# # """

# # import math
# # import pickle
# # import torch
# # from pathlib import Path

# # from exp.exp_prune import Exp_Prune
# # from layers.prune_mask import MaskedLayer


# # class Exp_ConceptPrune(Exp_Prune):
# #     """使用概念引导重要性的剪枝实验"""
    
# #     def __init__(self, args):
# #         super().__init__(args)
        
# #         # 加载概念引导重要性
# #         importance_path = Path('./pruning_importance/concept_guided_importance.pkl')
        
# #         if not importance_path.exists():
# #             raise FileNotFoundError(
# #                 f"概念重要性文件不存在: {importance_path}\n"
# #                 f"请先运行: bash scripts/04_compute_concept_importance.sh"
# #             )
        
# #         with open(importance_path, 'rb') as f:
# #             self.concept_importance = pickle.load(f)
        
# #         print("="*60)
# #         print("✓ 加载概念引导重要性")
# #         print("="*60)
# #         print(f"  模块数量: {len(self.concept_importance)}")
        
# #         # 显示重要性范围
# #         scores = list(self.concept_importance.values())
# #         print(f"  分数范围: [{min(scores):.6f}, {max(scores):.6f}]")
        
# #         # Top-5 重要模块
# #         sorted_modules = sorted(
# #             self.concept_importance.items(), 
# #             key=lambda x: x[1], 
# #             reverse=True
# #         )
# #         print(f"\n  Top-5 重要模块:")
# #         for name, score in sorted_modules[:5]:
# #             print(f"    {name:40s}: {score:.6f}")
        
# #         # Bottom-5 不重要模块
# #         print(f"\n  Bottom-5 不重要模块（优先剪枝）:")
# #         for name, score in sorted_modules[-5:]:
# #             print(f"    {name:40s}: {score:.6f}")
# #         print("="*60)
    
# #     def _map_module_name(self, target_name):
# #         """
# #         映射模块名: 从剪枝代码的命名到概念映射的命名
        
# #         Args:
# #             target_name: 例如 'backbone.encoder.block.0.layer.0.SelfAttention.q.mask_in'
        
# #         Returns:
# #             mapped_name: 例如 'encoder.block.0'
# #         """
# #         # 移除 .mask_in 和 .mask_out 后缀
# #         base_name = target_name.replace('.mask_in', '').replace('.mask_out', '')
        
# #         # 移除 'backbone.' 前缀（如果存在）
# #         if base_name.startswith('backbone.'):
# #             base_name = base_name[len('backbone.'):]
        
# #         # 提取核心层名 (encoder.block.X 或 decoder.block.X)
# #         parts = base_name.split('.')
        
# #         if len(parts) >= 3:
# #             # 格式: encoder.block.0.layer.0...
# #             # 提取: encoder.block.0
# #             if parts[0] in ['encoder', 'decoder'] and parts[1] == 'block':
# #                 return f'{parts[0]}.{parts[1]}.{parts[2]}'
        
# #         # 如果无法映射，返回原始名称
# #         return base_name
    
# #     def _find_concept_importance(self, module_name):
# #         """
# #         查找模块的概念重要性
        
# #         Args:
# #             module_name: 模块名
        
# #         Returns:
# #             importance: 概念重要性分数，找不到返回0.5（中等）
# #         """
# #         # 精确匹配
# #         if module_name in self.concept_importance:
# #             return self.concept_importance[module_name]
        
# #         # 模糊匹配：包含关系
# #         for concept_key, importance in self.concept_importance.items():
# #             if concept_key in module_name or module_name in concept_key:
# #                 return importance
        
# #         # 未找到，返回中等重要性
# #         print(f"⚠️ 未找到匹配: {module_name}，使用默认重要性0.5")
# #         return 0.5
    
# #     @torch.no_grad()
# #     def _prune(self, imps, unused_names):
# #         """
# #         使用概念引导重要性进行剪枝
        
# #         Args:
# #             imps: Taylor重要性字典（在本类中不使用）
# #             unused_names: 未使用的模块名集合
# #         """
# #         # 收集所有模块的概念重要性
# #         module_scores = {}
        
# #         for target_name in self.target_module_names:
# #             # 跳过未使用的模块
# #             mask_in_name = target_name + '.mask_in'
# #             mask_out_name = target_name + '.mask_out'
            
# #             if mask_in_name in unused_names and mask_out_name in unused_names:
# #                 continue
            
# #             # 映射模块名
# #             mapped_name = self._map_module_name(target_name)
            
# #             # 获取概念重要性
# #             importance = self._find_concept_importance(mapped_name)
            
# #             # 为 mask_in 和 mask_out 都记录重要性
# #             if getattr(self.args, 'mask_in', True) and mask_in_name not in unused_names:
# #                 module_scores[mask_in_name] = importance
# #             if getattr(self.args, 'mask_out', True) and mask_out_name not in unused_names:
# #                 module_scores[mask_out_name] = importance
        
# #         if not module_scores:
# #             print("⚠️ 没有可剪枝的模块")
# #             return
        
# #         # 转换为tensor
# #         scores = torch.tensor(list(module_scores.values()))
        
# #         # 计算剪枝数量
# #         prune_num = math.ceil(len(scores) * self.prune_ratio)
        
# #         if prune_num == 0:
# #             return
        
# #         # 找到剪枝阈值（低于阈值的会被剪掉）
# #         if len(scores) > prune_num:
# #             threshold = torch.topk(scores, prune_num, largest=False)[0][-1]
# #         else:
# #             threshold = scores.max()
        
# #         # 多GPU同步阈值
# #         if self.args.use_multi_gpu:
# #             import torch.distributed as dist
# #             dist.barrier()
# #             dist.all_reduce(threshold, op=dist.ReduceOp.MIN)
        
# #         # 执行剪枝
# #         pruned_count = 0
# #         for module_name, score in module_scores.items():
# #             if score <= threshold:
# #                 # 获取模块
# #                 base_name = '.'.join(module_name.split('.')[:-1])
# #                 module = self._model.get_submodule(base_name)
                
# #                 # 剪枝
# #                 if module_name.endswith('.mask_in'):
# #                     module.mask_in.mask.data.zero_()
# #                     pruned_count += module.mask_in.mask.numel()
# #                 elif module_name.endswith('.mask_out'):
# #                     module.mask_out.mask.data.zero_()
# #                     pruned_count += module.mask_out.mask.numel()
        
# #         # 打印剪枝信息
# #         print(f"\n[概念引导剪枝] 阈值={threshold:.6f}, "
# #               f"剪枝模块数={prune_num}/{len(module_scores)}, "
# #               f"剪枝参数数={pruned_count}")


# # if __name__ == '__main__':
# #     # 测试代码
# #     import argparse
    
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('--model', type=str, default='Chronos')
# #     parser.add_argument('--model_size', type=str, default='small')
# #     parser.add_argument('--prune_ratio', type=float, default=0.02)
# #     parser.add_argument('--use_multi_gpu', type=bool, default=False)
# #     parser.add_argument('--mask_in', type=bool, default=True)
# #     parser.add_argument('--mask_out', type=bool, default=True)
    
# #     args = parser.parse_args([])
    
# #     print("测试概念引导剪枝类初始化...")
# #     try:
# #         exp = Exp_ConceptPrune(args)
# #         print("\n✓ 初始化成功!")
# #     except Exception as e:
# #         print(f"\n✗ 初始化失败: {e}")
