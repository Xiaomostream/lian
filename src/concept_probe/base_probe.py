"""
概念探针基类和通用工具函数
提供跨模型共享的功能，减少代码重复
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def compute_fisher_score(
    act_pos: np.ndarray,
    act_neg: np.ndarray
) -> float:
    """
    计算Fisher线性判别比
    
    Args:
        act_pos: 正样本激活 (num_samples, dim)
        act_neg: 负样本激活 (num_samples, dim)
    
    Returns:
        Fisher分数
    """
    num_samples = len(act_pos)
    activations = np.concatenate((act_pos, act_neg), axis=0)
    labels = np.concatenate((np.ones(num_samples), np.zeros(num_samples)))
    
    # LDA投影
    lda = LinearDiscriminantAnalysis()
    lda.fit(activations, labels)
    projects = lda.transform(activations)
    
    proj_pos = projects[:num_samples]
    proj_neg = projects[num_samples:]
    
    # Fisher判别比
    mean_diff_sq = (proj_pos.mean() - proj_neg.mean()) ** 2
    var_sum = proj_pos.var() + proj_neg.var()
    
    return mean_diff_sq / (var_sum + 1e-8)


def load_concept_data(concept_dir: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    加载概念数据集
    
    Args:
        concept_dir: 概念数据目录路径 (例如: './concept_data/chronos-bolt-small')
    
    Returns:
        concept_datasets: {
            'periodicity': (positive_data, negative_data),
            'trend': (positive_data, negative_data),
            ...
        }
    """
    print(f"从 {concept_dir} 加载概念数据...")
    concept_datasets = {}
    
    for concept in ['periodicity', 'trend', 'seasonality', 'volatility', 'stationarity']:
        pos_path = f'{concept_dir}/{concept}_positive.npy'
        neg_path = f'{concept_dir}/{concept}_negative.npy'
        
        pos_data = np.load(pos_path)
        neg_data = np.load(neg_path)
        
        concept_datasets[concept] = (pos_data, neg_data)
        print(f"  ✓ {concept}: {pos_data.shape}")
    
    return concept_datasets


class BaseConceptProbe:
    """
    概念探针基类
    
    子类需要实现:
        - get_target_layers(): 返回目标层名称列表
        - _forward_pass(batch_tensor): 执行前向传播
    """
    
    def __init__(self, model, device='cuda'):
        """
        初始化基类
        
        Args:
            model: 已加载的模型对象
            device: 'cuda' 或 'cpu'
        """
        self.model = model
        self.device = device
        self.activations = {}
        self.model.eval()
    
    def get_target_layers(self) -> List[str]:
        """
        获取目标层名称列表
        
        子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现 get_target_layers() 方法")
    
    def register_hooks(self, target_modules: List[str]):
        """
        注册前向钩子到目标模块
        
        Args:
            target_modules: 目标模块名称列表
        """
        # 初始化激活值字典
        self.activations = {name: None for name in target_modules}
        
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                # 转换BFloat16 -> Float32 -> CPU -> Numpy
                if hidden.dtype == torch.bfloat16:
                    hidden = hidden.float()
                self.activations[name] = hidden.detach().cpu().numpy()
            return hook
        
        for name in target_modules:
            module = dict(self.model.named_modules())[name]
            module.register_forward_hook(get_activation(name))
    
    def _forward_pass(self, batch_tensor: torch.Tensor):
        """
        执行前向传播
        
        子类可以重写此方法以适配不同的模型接口
        
        Args:
            batch_tensor: 输入tensor
        """
        # 默认实现：直接调用模型
        _ = self.model(batch_tensor)
    
    @torch.no_grad()
    def extract_activations(
        self,
        data: np.ndarray,
        batch_size: int = 16
    ) -> Dict[str, np.ndarray]:
        """
        提取激活值
        
        Args:
            data: 输入数据 (num_samples, seq_len)
            batch_size: 批次大小
        
        Returns:
            layer_acts: {layer_name: activations (num_samples, dim)}
        """
        num_samples = len(data)
        layer_acts = {name: [] for name in self.activations.keys()}
        
        for i in tqdm(range(0, num_samples, batch_size), desc="提取激活"):
            batch = data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(self.device)
            
            # 前向传播（子类可重写）
            self._forward_pass(batch_tensor)
            
            # 收集激活(平均池化到样本级)
            for name, act in self.activations.items():
                # act: (batch, seq_len, dim) -> (batch, dim)
                sample_repr = act.mean(axis=1)
                layer_acts[name].append(sample_repr)
        
        # 拼接
        for name in layer_acts:
            layer_acts[name] = np.concatenate(layer_acts[name], axis=0)
        
        return layer_acts
    
    def compute_concept_map(
        self,
        concept_datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
        batch_size: int = 16
    ) -> Dict[str, Dict[str, float]]:
        """
        计算概念映射
        
        Args:
            concept_datasets: {'concept_name': (positive_data, negative_data)}
            batch_size: 批次大小
        
        Returns:
            concept_map: {
                'periodicity': {'layer1': score1, 'layer2': score2, ...},
                'trend': {...},
                ...
            }
        """
        # 获取目标层
        target_layers = self.get_target_layers()
        print(f"\n目标层数: {len(target_layers)}")
        
        # 注册钩子
        self.register_hooks(target_layers)
        
        concept_map = {}
        
        for concept_name, (pos_data, neg_data) in concept_datasets.items():
            print(f"\n处理概念: {concept_name}")
            
            # 提取激活
            print("  正样本...")
            pos_acts = self.extract_activations(pos_data, batch_size)
            print("  负样本...")
            neg_acts = self.extract_activations(neg_data, batch_size)
            
            # 计算Fisher分数
            sensitivity = {}
            for layer_name in pos_acts.keys():
                score = compute_fisher_score(
                    pos_acts[layer_name],
                    neg_acts[layer_name]
                )
                sensitivity[layer_name] = score
            
            concept_map[concept_name] = sensitivity
            
            # 显示Top-3
            sorted_layers = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
            print(f"  Top-3敏感层:")
            for name, score in sorted_layers[:3]:
                print(f"    {name}: {score:.4f}")
        
        return concept_map
    
    def save_concept_map(self, concept_map: Dict, save_path: str):
        """
        保存概念映射到文件
        
        Args:
            concept_map: 概念映射字典
            save_path: 保存路径
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(concept_map, f)
        
        print(f"\n✓ 保存到: {save_path}")
