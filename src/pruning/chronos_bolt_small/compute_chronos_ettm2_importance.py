"""
计算Chronos在ETTm2上的概念重要性
复用Chronos的概念映射，使用ETTm2的任务向量
"""

import sys
sys.path.append('src')

from pruning.concept_guided_pruning import ConceptGuidedPruning


if __name__ == '__main__':
    print("="*60)
    print("Chronos + ETTm2 概念重要性计算")
    print("="*60)
    
    # 创建概念引导剪枝器
    pruner = ConceptGuidedPruning(
        concept_map_path='./concept_maps/chronos_small_concept_map.pkl',
        task_vector_path='./task_vectors/ettm2_task_vector.pkl'
    )
    
    # 计算模块重要性
    module_importance = pruner.compute_module_importance()
    
    # 保存重要性
    pruner.save_importance(
        module_importance,
        save_path='./pruning_importance/Chronos_bolt_small/chronos_ettm2_importance.pkl'
    )
    
    # 分析分布
    pruner.analyze_importance_distribution(module_importance)
    
    print("\n" + "="*60)
    print("✓ Chronos + ETTm2 概念重要性计算完成!")
    print("="*60)
