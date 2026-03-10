"""
TTM模型概念探针
基于BaseConceptProbe，适配TinyTimeMixer架构（MLP-Mixer）
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

# 使用相对导入（用于 python -m 运行）
from .base_probe import BaseConceptProbe, load_concept_data


class TTMConceptProbe(BaseConceptProbe):
    """TTM模型专用概念探针"""
    
    def __init__(self, device='cuda'):
        """
        初始化TTM探针
        
        Args:
            device: 'cuda' 或 'cpu'
        """
        print("加载TTM模型...")
        
        # 本地路径（服务器实际路径）
        local_path = Path('/home/ncut/Xiaomo/checkpoints/hf_models/models--ibm-granite--granite-timeseries-ttm-r2')
        
        if local_path.exists():
            model_path = str(local_path)
            print(f"  使用本地模型: {model_path}")
        else:
            model_path = "ibm-granite/granite-timeseries-ttm-r2"
            print(f"  从HuggingFace下载: {model_path}")
        
        # 加载TTM模型
        from tsfm_public.toolkit.get_model import get_model
        model = get_model(
            model_path,
            context_length=512,        # TTM-R2 实际选择的context length
            prediction_length=96,
            freq_prefix_tuning=False,
            prefer_l1_loss=False,
        )
        
        # 打印模型结构以便调试
        print("\n[调试] TTM模型顶层结构:")
        for name, child in model.named_children():
            print(f"  - {name}: {type(child).__name__}")
        
        print("\n[调试] 所有模块名称 (前30个):")
        all_module_names = [name for name, _ in model.named_modules() if name]
        for name in all_module_names[:30]:
            print(f"  - {name}")
        
        # 获取context_length
        self.context_length = model.config.context_length
        print(f"\n  TTM context_length = {self.context_length}")
        
        # 调用基类构造函数
        super().__init__(model=model, device=device)
        
        # 将模型移动到指定设备
        self.model = self.model.to(device)
        
        # 动态获取层数
        self._find_encoder_layers()
        print(f"✓ 模型加载完成 (层数: {self.num_layers})")
    
    def _find_encoder_layers(self):
        """动态查找encoder层结构"""
        all_modules = dict(self.model.named_modules())
        
        # TTM的实际结构是:
        # backbone.encoder.mlp_mixer_encoder.mixers.{i}.mixer_layers.{j}
        # 其中每个mixer_layers包含 patch_mixer 和 feature_mixer
        self.target_layer_names = []
        
        # 查找所有 mixer_layers.X 级别的模块（这是TTM的基本计算单元）
        for name in sorted(all_modules.keys()):
            # 匹配 backbone.encoder.mlp_mixer_encoder.mixers.X.mixer_layers.Y
            if 'mlp_mixer_encoder.mixers.' in name and '.mixer_layers.' in name:
                parts = name.split('.')
                # 确保是 mixer_layers 的直接子项（即 mixers.X.mixer_layers.Y，不是更深的子模块）
                try:
                    mixer_idx = parts.index('mixer_layers')
                    # mixer_layers.Y 后面不应该还有更多层级
                    if mixer_idx == len(parts) - 2 and parts[-1].isdigit():
                        self.target_layer_names.append(name)
                except ValueError:
                    continue
        
        # 也加入decoder的mixer层
        for name in sorted(all_modules.keys()):
            if 'decoder.decoder_block.mixers.' in name:
                parts = name.split('.')
                # decoder.decoder_block.mixers.X 级别
                if len(parts) == 4 and parts[-1].isdigit():
                    self.target_layer_names.append(name)
        
        self.num_layers = len(self.target_layer_names)
        print(f"  找到 {self.num_layers} 个目标层:")
        for name in self.target_layer_names:
            print(f"    - {name}")
    
    def get_target_layers(self) -> List[str]:
        """获取TTM模型的目标层"""
        print(f"\n实际注册的目标层 (共{len(self.target_layer_names)}个):")
        for layer in self.target_layer_names:
            print(f"  - {layer}")
        return self.target_layer_names
    
    @torch.no_grad()
    def extract_activations(
        self,
        data: np.ndarray,
        batch_size: int = 16
    ) -> Dict[str, np.ndarray]:
        """
        重写激活提取方法，处理TTM的多维输出
        TTM mixer_layer输出可能是3D+(batch, patches, channels, ...)
        需要展平为2D (num_samples, features) 供LDA使用
        """
        from tqdm import tqdm
        
        num_samples = len(data)
        layer_acts = {name: [] for name in self.activations.keys()}
        
        for i in tqdm(range(0, num_samples, batch_size), desc="提取激活"):
            batch = data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(self.device)
            
            # 前向传播
            self._forward_pass(batch_tensor)
            
            # 收集激活并展平
            for name, act in self.activations.items():
                if act is None:
                    continue
                # act可能是: (batch, patches, dim) 或 (batch, patches, channels, dim)
                # 全部展平为: (batch, features)
                batch_act = act
                if batch_act.ndim > 2:
                    batch_act = batch_act.reshape(batch_act.shape[0], -1)
                layer_acts[name].append(batch_act)
        
        # 拼接
        for name in layer_acts:
            if layer_acts[name]:
                layer_acts[name] = np.concatenate(layer_acts[name], axis=0)
            else:
                # 如果某层没有激活，用零填充
                layer_acts[name] = np.zeros((num_samples, 1))
        
        return layer_acts
    
    def _forward_pass(self, batch_tensor: torch.Tensor):
        """
        TTM特定的前向传播
        
        TTM输入格式: (batch, context_length, num_channels)
        概念数据格式: (batch, seq_len) → 需要截断/填充至context_length → (batch, context_length, 1)
        """
        with torch.no_grad():
            # 将 (batch, seq_len) 截断或填充到 context_length
            if batch_tensor.dim() == 2:
                seq_len = batch_tensor.shape[1]
                if seq_len > self.context_length:
                    batch_tensor = batch_tensor[:, -self.context_length:]
                elif seq_len < self.context_length:
                    padding = torch.zeros(batch_tensor.shape[0], self.context_length - seq_len, device=batch_tensor.device)
                    batch_tensor = torch.cat([padding, batch_tensor], dim=1)
                # 转为 (batch, context_length, 1) - 单变量
                batch_tensor = batch_tensor.unsqueeze(-1)
            
            # TTM forward
            _ = self.model(past_values=batch_tensor)


if __name__ == '__main__':
    # 加载TTM专用概念数据
    concept_datasets = load_concept_data('./concept_data/ttm_data')
    
    # 创建TTM探针
    probe = TTMConceptProbe(device='cuda')
    
    # 如果概念数据长度与模型不匹配，自动截断
    adjusted_datasets = {}
    for concept, (pos, neg) in concept_datasets.items():
        ctx = probe.context_length
        if pos.shape[1] != ctx:
            print(f"  调整 {concept}: {pos.shape[1]} -> {ctx}")
            pos = pos[:, -ctx:] if pos.shape[1] > ctx else np.pad(pos, ((0,0),(ctx-pos.shape[1],0)))
            neg = neg[:, -ctx:] if neg.shape[1] > ctx else np.pad(neg, ((0,0),(ctx-neg.shape[1],0)))
        adjusted_datasets[concept] = (pos, neg)
    
    # 计算概念映射
    concept_map = probe.compute_concept_map(adjusted_datasets, batch_size=32)
    
    # 保存
    save_path = './concept_maps/ttm_concept_map.pkl'
    probe.save_concept_map(concept_map, save_path)
