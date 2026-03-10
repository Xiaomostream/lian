"""
TimeMoE模型概念探针
基于BaseConceptProbe，适配TimeMoE-MoE架构
"""

import torch
from pathlib import Path
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

# 使用相对导入（用于 python -m 运行）
from .base_probe import BaseConceptProbe, load_concept_data


class TimeMoEConceptProbe(BaseConceptProbe):
    """TimeMoE模型专用概念探针"""
    
    def __init__(self, model_size='50m', device='cuda'):
        """
        初始化TimeMoE探针
        
        Args:
            model_size: 模型规模 ('50m', '200m')
            device: 'cuda' 或 'cpu'
        """
        print(f"加载TimeMoE-{model_size}模型...")
        
        # 本地路径映射（服务器实际路径）
        local_paths = {
            '50m': '/home/ncut/Xiaomo/checkpoints/hf_models/TimeMoE-50M',
            '200m': '/home/ncut/Xiaomo/checkpoints/hf_models/TimeMoE-200M',
        }
        
        model_path = local_paths.get(model_size, f"Maple728/TimeMoE-{model_size}")
        print(f"  模型路径: {model_path}")
        
        # 加载TimeMoE模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        
        # 调用基类构造函数
        super().__init__(model=model, device=device)
        
        self.model_size = model_size
        self.num_layers = len(model.model.layers)
        
        # 获取expert数量 (假设所有层的expert数相同)
        if hasattr(model.model.layers[0], 'mlp') and hasattr(model.model.layers[0].mlp, 'experts'):
            self.num_experts = len(model.model.layers[0].mlp.experts)
        else:
            self.num_experts = 1  # 如果没有MoE结构，默认为1
        
        print(f"✓ 模型加载完成 (层数: {self.num_layers}, Expert数: {self.num_experts})")
    
    def get_target_layers(self) -> List[str]:
        """
        获取TimeMoE模型的目标层
        
        返回所有层的self_attn和MLP层
        """
        layers = []
        
        # 调试：打印所有可用的模块名称
        all_modules = dict(self.model.named_modules())
        layer_modules = [name for name in all_modules.keys() if 'layers.' in name and len(name.split('.')) <= 4]
        print(f"\n可用的层模块示例 (前10个):")
        for name in sorted(layer_modules)[:10]:
            print(f"  - {name}")
        
        for i in range(self.num_layers):
            # 注意力层 - 尝试多种可能的命名
            attn_candidates = [
                f'model.layers.{i}.self_attn',
                f'model.layers.{i}.attention',
                f'layers.{i}.self_attn',
            ]
            for attn_name in attn_candidates:
                if attn_name in all_modules:
                    layers.append(attn_name)
                    break
            
            # MLP层 - 检查是否有MoE结构
            if self.num_experts > 1:
                # 有MoE，注册每个expert
                for j in range(self.num_experts):
                    expert_name = f'model.layers.{i}.mlp.experts.{j}'
                    if expert_name in all_modules:
                        layers.append(expert_name)
            else:
                # 没有MoE，尝试多种MLP命名（TimeMoE使用 ffn_layer）
                mlp_candidates = [
                    f'model.layers.{i}.ffn_layer',  # TimeMoE
                    f'model.layers.{i}.mlp',
                    f'model.layers.{i}.feed_forward',
                    f'model.layers.{i}.ffn',
                    f'layers.{i}.mlp',
                ]
                for mlp_name in mlp_candidates:
                    if mlp_name in all_modules:
                        layers.append(mlp_name)
                        break
        
        print(f"\n实际注册的层 (共{len(layers)}个):")
        for layer in layers[:5]:
            print(f"  - {layer}")
        if len(layers) > 5:
            print(f"  ... 还有 {len(layers)-5} 个层")
        
        return layers
    
    def _forward_pass(self, batch_tensor: torch.Tensor):
        """
        TimeMoE特定的前向传播
        
        Args:
            batch_tensor: 输入数据 (batch, seq_len)
        """
        # TimeMoE作为CausalLM，使用input_ids参数
        # 注意：这里假设输入是连续值，需要适配为token形式
        # 如果TimeMoE有专门的时序输入接口，需要调整
        
        # 简化处理：将连续值作为embeddings直接输入
        # 更准确的方法需要查看TimeMoE的具体接口
        with torch.no_grad():
            # 获取embedding层的输出维度
            embed_dim = self.model.config.hidden_size
            
            # 转换为bfloat16以匹配模型dtype
            if batch_tensor.dtype != torch.bfloat16:
                batch_tensor = batch_tensor.to(torch.bfloat16)
            
            # 如果输入维度不匹配，需要投影
            if batch_tensor.shape[-1] != embed_dim:
                # 简单的线性投影（实际使用中可能需要更复杂的处理）
                batch_tensor = batch_tensor.unsqueeze(-1).expand(-1, -1, embed_dim)
            
            # 直接传入model的核心层
            _ = self.model.model(inputs_embeds=batch_tensor)


if __name__ == '__main__':
    # 加载TimeMoE专用概念数据
    concept_datasets = load_concept_data('./concept_data/timemoe_data')
    
    # 创建TimeMoE探针
    probe = TimeMoEConceptProbe(model_size='50m', device='cuda')
    
    # 计算概念映射
    concept_map = probe.compute_concept_map(concept_datasets, batch_size=16)
    
    # 保存
    save_path = './concept_maps/timemoe_50m_concept_map.pkl'
    probe.save_concept_map(concept_map, save_path)
