# """
# 计算TimeMoE在ETTh1上的概念重要性（Module-level）
# 使用TimeMoE专用的ConceptGuidedPruning类
# """

# from .concept_guided_pruning import ConceptGuidedPruning


# def compute_block_importance(module_importance, num_layers=12):
#     """
#     将module-level重要性聚合为Block-level重要性
    
#     Args:
#         module_importance: {module_name: importance}
#         num_layers: Transformer层数
    
#     Returns:
#         block_importance: {block_idx: importance}
#     """
#     block_importance = {}
    
#     for layer_idx in range(num_layers):
#         # 收集该Block的所有子模块重要性
#         attn_name = f'model.layers.{layer_idx}.self_attn'
#         ffn_name = f'model.layers.{layer_idx}.ffn_layer'
        
#         attn_imp = module_importance.get(attn_name, 0.0)
#         ffn_imp = module_importance.get(ffn_name, 0.0)
        
#         # Block重要性 = (attn + ffn) / 2
#         block_importance[layer_idx] = (attn_imp + ffn_imp) / 2.0
    
#     return block_importance


# if __name__ == '__main__':
#     print("="*60)
#     print("TimeMoE + ETTh1  概念重要性计算（Block-level）")
#     print("="*60)
    
#     # 创建概念引导剪枝器
#     pruner = ConceptGuidedPruning(
#         concept_map_path='./concept_maps/timemoe_50m_concept_map.pkl',
#         task_vector_path='./task_vectors/etth1_task_vector.pkl'
#     )
    
#     # 计算模块重要性
#     module_importance = pruner.compute_module_importance()
    
#     # 聚合为Block重要性
#     print("\n" + "="*60)
#     print("聚合为Block重要性")
#     print("="*60)
    
#     block_importance = compute_block_importance(module_importance, num_layers=12)
    
#     # 显示Block重要性排名
#     sorted_blocks = sorted(block_importance.items(), key=lambda x: x[1], reverse=True)
    
#     print("\n✅ Block重要性排名（从高到低）:")
#     for rank, (block_idx, importance) in enumerate(sorted_blocks, 1):
#         bar = '█' * int(importance * 30) if importance > 0 else ''
#         print(f"  #{rank:2d}  Block {block_idx:2d}: {importance:.6f} {bar}")
    
#     # 保存Block重要性
#     import pickle
#     from pathlib import Path
    
#     output_dir = './pruning_importance/TimeMoe-50M'
#     output_path = f'{output_dir}/timemoe_etth1_importance.pkl'
    
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
    
#     with open(output_path, 'wb') as f:
#         pickle.dump(block_importance, f)
    
#     print(f"\n✓ Block重要性已保存到: {output_path}")
    
#     print("\n" + "="*60)
#     print("✓ TimeMoE + ETTh1 概念重要性计算完成!")
#     print("="*60)
"""
计算TimeMoE在ETTh1上的概念重要性
使用TimeMoE专用的ConceptGuidedPruning类
"""

from .concept_guided_pruning import ConceptGuidedPruning


if __name__ == '__main__':
    print("="*60)
    print("TimeMoE + ETTh1 概念重要性计算")
    print("="*60)
    
    # 创建概念引导剪枝器
    pruner = ConceptGuidedPruning(
        concept_map_path='./concept_maps/timemoe_50m_concept_map.pkl',
        task_vector_path='./task_vectors/etth1_task_vector.pkl'
    )
    
    # 计算模块重要性
    module_importance = pruner.compute_module_importance()
    
    # 保存重要性
    pruner.save_importance(
        module_importance,
        save_path='./pruning_importance/TimeMoe-50M/timemoe_etth1_importance.pkl'
    )
    
    # 分析分布
    pruner.analyze_importance_distribution(module_importance)
    
    print("\n" + "="*60)
    print("✓ TimeMoE + ETTh1 概念重要性计算完成!")
    print("="*60)

