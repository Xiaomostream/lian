"""
TimerXL (timer-base-84m) 模型概念探针
基于BaseConceptProbe，适配TimerXL Transformer Decoder架构
"""

import torch
from pathlib import Path
from typing import List
from transformers import AutoModelForCausalLM

# 使用相对导入（用于 python -m 运行）
from .base_probe import BaseConceptProbe, load_concept_data


class TimerXLConceptProbe(BaseConceptProbe):
    """TimerXL模型专用概念探针"""
    
    def __init__(self, device='cuda'):
        """
        初始化TimerXL探针
        
        Args:
            device: 'cuda' 或 'cpu'
        """
        print("加载TimerXL (timer-base-84m)模型...")
        
        # 本地路径（服务器实际路径）
        local_path = Path('/home/ncut/Xiaomo/checkpoints/hf_models/models--thuml--timer-base-84m')
        
        if local_path.exists():
            model_path = str(local_path)
            print(f"  使用本地模型: {model_path}")
        else:
            model_path = "thuml/timer-base-84m"
            print(f"  从HuggingFace下载: {model_path}")
        
        # 加载TimerXL模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device,
        )
        
        # 调用基类构造函数
        super().__init__(model=model, device=device)
        
        self.num_layers = len(model.model.layers)
        
        # 打印模型结构
        print(f"\n[调试] TimerXL模型顶层结构:")
        for name, child in model.named_children():
            print(f"  - {name}: {type(child).__name__}")
        
        print(f"✓ 模型加载完成 (层数: {self.num_layers})")
    
    def get_target_layers(self) -> List[str]:
        """
        获取TimerXL模型的目标层
        每层包含: self_attn + ffn_layer
        """
        layers = []
        
        all_modules = dict(self.model.named_modules())
        
        # 打印部分模块名供调试
        layer_modules = [name for name in all_modules.keys() if 'layers.' in name and len(name.split('.')) <= 4]
        print(f"\n可用的层模块示例 (前10个):")
        for name in sorted(layer_modules)[:10]:
            print(f"  - {name}")
        
        for i in range(self.num_layers):
            # 注意力层
            attn_name = f'model.layers.{i}.self_attn'
            if attn_name in all_modules:
                layers.append(attn_name)
            
            # FFN层 (TimerXL使用 ffn_layer)
            ffn_name = f'model.layers.{i}.ffn_layer'
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
        TimerXL特定的前向传播
        
        Args:
            batch_tensor: 输入数据 (batch, seq_len)
        """
        with torch.no_grad():
            # 确保设备和dtype匹配
            batch_tensor = batch_tensor.to(self.device)
            
            embed_dim = self.model.config.hidden_size
            
            if batch_tensor.shape[-1] != embed_dim:
                batch_tensor = batch_tensor.unsqueeze(-1).expand(-1, -1, embed_dim)
            
            # 转为模型参数的dtype
            model_dtype = next(self.model.parameters()).dtype
            if batch_tensor.dtype != model_dtype:
                batch_tensor = batch_tensor.to(model_dtype)
            
            # 使用 inputs_embeds 直接传入
            _ = self.model.model(inputs_embeds=batch_tensor)


if __name__ == '__main__':
    # 加载TimerXL专用概念数据
    concept_datasets = load_concept_data('./concept_data/timerxl_data')
    
    # 创建TimerXL探针
    probe = TimerXLConceptProbe(device='cuda')
    
    # 计算概念映射
    concept_map = probe.compute_concept_map(concept_datasets, batch_size=16)
    
    # 保存
    save_path = './concept_maps/timerxl_concept_map.pkl'
    probe.save_concept_map(concept_map, save_path)
