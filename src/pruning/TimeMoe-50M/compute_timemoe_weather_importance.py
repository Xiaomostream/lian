"""
计算TimeMoE在Weather上的概念重要性
使用TimeMoE专用的ConceptGuidedPruning类
"""

from .concept_guided_pruning import ConceptGuidedPruning


if __name__ == '__main__':
    print("="*60)
    print("TimeMoE + Weather 概念重要性计算")
    print("="*60)

    pruner = ConceptGuidedPruning(
        concept_map_path='./concept_maps/timemoe_50m_concept_map.pkl',
        task_vector_path='./task_vectors/weather_task_vector.pkl'
    )

    module_importance = pruner.compute_module_importance()

    pruner.save_importance(
        module_importance,
        save_path='./pruning_importance/TimeMoe-50M/timemoe_weather_importance.pkl'
    )

    pruner.analyze_importance_distribution(module_importance)

    print("\n" + "="*60)
    print("✓ TimeMoE + Weather 概念重要性计算完成!")
    print("="*60)
