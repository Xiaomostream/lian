"""
Chronos模型概念探针
基于BaseConceptProbe，适配Chronos-Bolt架构
"""

import torch
from pathlib import Path
from typing import List
from chronos import ChronosBoltPipeline

# 使用相对导入（用于 python -m 运行）
from .base_probe import BaseConceptProbe, load_concept_data


class ChronosConceptProbe(BaseConceptProbe):
    """Chronos模型专用概念探针"""
    
    def __init__(self, model_size='small', device='cuda'):
        """
        初始化Chronos探针
        
        Args:
            model_size: 模型规模 ('small', 'base', 'large')
            device: 'cuda' 或 'cpu'
        """
        print(f"加载Chronos-{model_size}模型...")
        
        # 直接使用用户提供的模型路径
        local_path = Path(f'/home/ncut/Xiaomo/checkpoints/hf_models/models--amazon--chronos-bolt-{model_size}')
        
        if local_path.exists():
            model_path = str(local_path)
            print(f"  使用本地模型: {model_path}")
        else:
            model_path = f"amazon/chronos-bolt-{model_size}"
            print(f"  从HuggingFace下载: {model_path}")
        
        # 加载模型
        pipeline = ChronosBoltPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        # 调用基类构造函数
        super().__init__(model=pipeline.model, device=device)
        
        self.model_size = model_size
        print("✓ 模型加载完成")
    
    def get_target_layers(self) -> List[str]:
        """
        获取Chronos模型的目标层
        
        返回Encoder和Decoder的所有block层
        """
        layers = []
        
        # Encoder层
        for i in range(len(self.model.encoder.block)):
            layers.append(f'encoder.block.{i}')
        
        # Decoder层  
        for i in range(len(self.model.decoder.block)):
            layers.append(f'decoder.block.{i}')
        
        return layers
    
    def _forward_pass(self, batch_tensor: torch.Tensor):
        """
        Chronos特定的前向传播
        
        Args:
            batch_tensor: 输入数据 (batch, seq_len)
        """
        # Chronos使用context参数
        _ = self.model(context=batch_tensor)


if __name__ == '__main__':
    # 加载Chronos专用概念数据
    concept_datasets = load_concept_data('./concept_data/chronos-bolt-small')
    
    # 创建Chronos探针
    probe = ChronosConceptProbe(model_size='small', device='cuda')
    
    # 计算概念映射
    concept_map = probe.compute_concept_map(concept_datasets, batch_size=32)
    
    # 保存
    save_path = './concept_maps/chronos_small_concept_map.pkl'
    probe.save_concept_map(concept_map, save_path)
