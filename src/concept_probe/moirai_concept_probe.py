"""
Moirai (moirai-1.0-R-base) 模型概念探针
基于BaseConceptProbe，适配Moirai Transformer Encoder架构
"""

import sys
import os
import torch
from pathlib import Path
from typing import List

# 将 src/tsfm 加入搜索路径（layers模块在src/tsfm/layers/下）
_tsfm_path = os.path.join(os.path.dirname(__file__), '..', 'tsfm')
if os.path.isdir(_tsfm_path) and _tsfm_path not in sys.path:
    sys.path.insert(0, os.path.abspath(_tsfm_path))

# 使用相对导入（用于 python -m 运行）
from .base_probe import BaseConceptProbe, load_concept_data


class MoiraiConceptProbe(BaseConceptProbe):
    """Moirai模型专用概念探针"""
    
    def __init__(self, model_size='base', device='cuda'):
        """
        初始化Moirai探针
        
        Args:
            model_size: 模型规模 ('small', 'base', 'large')
            device: 'cuda' 或 'cpu'
        """
        print(f"加载Moirai-{model_size}模型...")
        
        # 本地路径（服务器实际路径）
        local_paths = {
            'small': '/home/ncut/Xiaomo/checkpoints/hf_models/models--Salesforce--moirai-1.0-R-small',
            'base': '/home/ncut/Xiaomo/checkpoints/hf_models/models--Salesforce--moirai-1.0-R-base',
            'large': '/home/ncut/Xiaomo/checkpoints/hf_models/models--Salesforce--moirai-1.0-R-large',
        }
        
        local_path = Path(local_paths.get(model_size, ''))
        if local_path.exists():
            model_path = str(local_path)
            print(f"  使用本地模型: {model_path}")
        else:
            model_path = f"Salesforce/moirai-1.0-R-{model_size}"
            print(f"  从HuggingFace下载: {model_path}")
        
        # 加载Moirai模型
        from layers.moirai_module import MoiraiModule
        model = MoiraiModule.from_pretrained(model_path)
        model = model.to(device)
        
        # 调用基类构造函数
        super().__init__(model=model, device=device)
        
        self.model_size = model_size
        self.num_layers = len(model.encoder.layers)
        
        # 打印模型结构
        print(f"\n[调试] Moirai模型顶层结构:")
        for name, child in model.named_children():
            print(f"  - {name}: {type(child).__name__}")
        
        print(f"✓ 模型加载完成 (层数: {self.num_layers})")
    
    def get_target_layers(self) -> List[str]:
        """
        获取Moirai模型的目标层
        每层包含: self_attn + ffn
        """
        layers = []
        
        all_modules = dict(self.model.named_modules())
        
        # 打印部分模块名供调试
        layer_modules = [name for name in all_modules.keys() 
                        if 'encoder.layers.' in name and len(name.split('.')) <= 4]
        print(f"\n可用的层模块示例 (前10个):")
        for name in sorted(layer_modules)[:10]:
            print(f"  - {name}")
        
        for i in range(self.num_layers):
            # 注意力层
            attn_name = f'encoder.layers.{i}.self_attn'
            if attn_name in all_modules:
                layers.append(attn_name)
            
            # FFN层 (Moirai使用 ffn)
            ffn_name = f'encoder.layers.{i}.ffn'
            if ffn_name in all_modules:
                layers.append(ffn_name)
        
        print(f"\n实际注册的层 (共{len(layers)}个):")
        for layer in layers[:6]:
            print(f"  - {layer}")
        if len(layers) > 6:
            print(f"  ... 还有 {len(layers)-6} 个层")
        
        return layers
    
    def _forward_pass(self, batch_tensor: torch.Tensor):
        """
        Moirai特定的前向传播
        
        Args:
            batch_tensor: 输入数据 (batch, seq_len)
        """
        with torch.no_grad():
            # 确保设备和dtype匹配
            batch_tensor = batch_tensor.to(self.device)
            
            # 从 norm1.weight 可靠获取 d_model (base=768, small=384, large=1024)
            embed_dim = self.model.encoder.layers[0].norm1.weight.shape[0]
            
            if batch_tensor.shape[-1] != embed_dim:
                batch_tensor = batch_tensor.unsqueeze(-1).expand(-1, -1, embed_dim)
            
            # 转为模型参数的dtype
            model_dtype = next(self.model.parameters()).dtype
            if batch_tensor.dtype != model_dtype:
                batch_tensor = batch_tensor.to(model_dtype)
            
            # 使用encoder直接前向传播
            _ = self.model.encoder(batch_tensor)


if __name__ == '__main__':
    # 加载Moirai专用概念数据
    concept_datasets = load_concept_data('./concept_data/moirai_data')
    
    # 创建Moirai探针
    probe = MoiraiConceptProbe(model_size='base', device='cuda')
    
    # 计算概念映射
    concept_map = probe.compute_concept_map(concept_datasets, batch_size=16)
    
    # 保存
    save_path = './concept_maps/moirai_base_concept_map.pkl'
    probe.save_concept_map(concept_map, save_path)
