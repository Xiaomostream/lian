"""
计算TimerXL在ETTh1上的概念重要性
"""

from .concept_guided_pruning import ConceptGuidedPruning


if __name__ == '__main__':
    print("="*60)
    print("TimerXL + ETTh1 概念重要性计算")
    print("="*60)
    
    pruner = ConceptGuidedPruning(
        concept_map_path='./concept_maps/timerxl_concept_map.pkl',
        task_vector_path='./task_vectors/etth1_task_vector.pkl'
    )
    
    module_importance = pruner.compute_module_importance()
    
    pruner.save_importance(
        module_importance,
        save_path='./pruning_importance/timerxl/timerxl_etth1_importance.pkl'
    )
    
    pruner.analyze_importance_distribution(module_importance)
    
    print("\n" + "="*60)
    print("✓ TimerXL + ETTh1 概念重要性计算完成!")
    print("="*60)
