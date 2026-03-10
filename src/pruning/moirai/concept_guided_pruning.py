"""
概念引导剪枝实现（Moirai版本）
结合 Task[c] 和 Map[c, module] 计算模块重要性
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict


class ConceptGuidedPruning:
    """概念引导的剪枝重要性计算"""
    
    def __init__(
        self,
        concept_map_path: str = './concept_maps/moirai_base_concept_map.pkl',
        task_vector_path: str = './task_vectors/etth1_task_vector.pkl'
    ):
        print("="*60)
        print("概念引导剪枝初始化 (Moirai)")
        print("="*60)
        
        print(f"\n加载概念映射: {concept_map_path}")
        with open(concept_map_path, 'rb') as f:
            self.concept_map = pickle.load(f)
        
        print(f"  ✓ 概念数量: {len(self.concept_map)}")
        print(f"  ✓ 概念列表: {list(self.concept_map.keys())}")
        
        print(f"\n加载任务向量: {task_vector_path}")
        with open(task_vector_path, 'rb') as f:
            self.task_vector = pickle.load(f)
        
        print(f"  ✓ 任务概念权重:")
        for concept, weight in self.task_vector.items():
            bar = '█' * int(weight * 20)
            print(f"    {concept:15s}: {weight:.4f} {bar}")
        
        concept_map_concepts = set(self.concept_map.keys())
        task_vector_concepts = set(self.task_vector.keys())
        
        if concept_map_concepts != task_vector_concepts:
            print(f"\n⚠️ 警告: 概念不一致")
            print(f"  Map中有但Task中没有: {concept_map_concepts - task_vector_concepts}")
            print(f"  Task中有但Map中没有: {task_vector_concepts - concept_map_concepts}")
        else:
            print(f"\n✓ 概念一致性检查通过")
    
    def compute_module_importance(self) -> Dict[str, float]:
        """
        I(module) = Σ_c Task[c] × Map[c, module]
        """
        print("\n" + "="*60)
        print("计算模块概念重要性")
        print("="*60)
        
        first_concept = list(self.concept_map.keys())[0]
        all_modules = list(self.concept_map[first_concept].keys())
        
        print(f"\n模块总数: {len(all_modules)}")
        
        module_importance = {module: 0.0 for module in all_modules}
        
        for concept, task_weight in self.task_vector.items():
            concept_sensitivities = self.concept_map[concept]
            for module, sensitivity in concept_sensitivities.items():
                module_importance[module] += task_weight * sensitivity
        
        max_importance = max(module_importance.values()) if module_importance else 1.0
        min_importance = min(module_importance.values()) if module_importance else 0.0
        
        if max_importance > min_importance:
            for module in module_importance:
                module_importance[module] = (
                    (module_importance[module] - min_importance) / 
                    (max_importance - min_importance)
                )
        
        sorted_modules = sorted(module_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\n✅ Top-10 最重要模块:")
        for i, (module, score) in enumerate(sorted_modules[:10], 1):
            print(f"  {i:2d}. {module:40s}: {score:.6f}")
        
        print("\n❌ Bottom-10 最不重要模块:")
        for i, (module, score) in enumerate(sorted_modules[-10:], 1):
            print(f"  {i:2d}. {module:40s}: {score:.6f}")
        
        return module_importance
    
    def save_importance(self, module_importance, save_path):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(module_importance, f)
        print(f"\n✓ 模块重要性已保存到: {save_path}")
    
    @staticmethod
    def load_importance(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def analyze_importance_distribution(self, module_importance):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        scores = list(module_importance.values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.hist(scores, bins=30, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Importance Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Moirai Module Importance Distribution', fontsize=14)
        ax1.grid(alpha=0.3)
        
        sorted_scores = sorted(scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        
        ax2.plot(sorted_scores, cumulative, linewidth=2, color='darkblue')
        ax2.set_xlabel('Importance Score', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title('Cumulative Distribution', fontsize=14)
        ax2.grid(alpha=0.3)
        
        prune_ratio = 0.2
        threshold_idx = int(len(sorted_scores) * prune_ratio)
        threshold_score = sorted_scores[threshold_idx]
        
        ax2.axvline(threshold_score, color='red', linestyle='--', 
                    label=f'Prune {prune_ratio*100:.0f}% (threshold={threshold_score:.4f})')
        ax2.axhline(prune_ratio, color='red', linestyle='--', alpha=0.5)
        ax2.legend()
        
        plt.tight_layout()
        
        save_path = './visualizations/moirai_importance_distribution.pdf'
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 重要性分布图已保存到: {save_path}")
        plt.close()
