"""
TimeMoE概念映射可视化工具
生成TimeMoE-50M的概念敏感度热图和柱状图
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def visualize_concept_sensitivity_heatmap(
    concept_map: dict,
    save_path: str = './visualizations/probe_training_timemoe-50M/concept_sensitivity_heatmap.pdf'
):
    """
    绘制概念敏感度热图
    
    Args:
        concept_map: {
            'periodicity': {'model.layers.0.self_attn': 0.85, ...},
            'trend': {...},
            ...
        }
        save_path: 保存路径
    """
    # 准备数据
    concepts = list(concept_map.keys())
    modules = list(concept_map[concepts[0]].keys())
    
    # 构建矩阵: (num_modules, num_concepts)
    matrix = np.zeros((len(modules), len(concepts)))
    
    for j, concept in enumerate(concepts):
        for i, module in enumerate(modules):
            matrix[i, j] = concept_map[concept][module]
    
    # 归一化到0-1
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)
    
    # 绘图
    sns.set(font_scale=1.2, style="whitegrid")
    plt.rcParams['font.family'] = 'serif'
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # 热图
    sns.heatmap(
        matrix,
        cmap='viridis',
        ax=ax,
        cbar_kws={'label': 'Normalized Fisher Score'},
        xticklabels=[c.capitalize() for c in concepts],
        yticklabels=[m.replace('model.layers.', 'L') for m in modules],  # 简化层名
        annot=False
    )
    
    ax.set_xlabel('Concept', fontsize=14, labelpad=10)
    ax.set_ylabel('TimeMoE Module', fontsize=14, labelpad=10)
    ax.set_title('TimeMoE-50M Concept Sensitivity Heatmap', fontsize=16, pad=15)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 热图已保存到: {save_path}")
    plt.close()


def visualize_concept_bar_chart(
    concept_map: dict,
    concept_name: str,
    top_k: int = 10,
    save_path: str = None
):
    """
    绘制某个概念的top-k敏感模块柱状图
    
    Args:
        concept_map: 概念映射字典
        concept_name: 要可视化的概念名
        top_k: 显示top-k个模块
        save_path: 保存路径
    """
    sensitivity = concept_map[concept_name]
    
    # 排序并取top-k
    sorted_items = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    top_modules = [item[0].replace('model.layers.', 'L') for item in sorted_items[:top_k]]
    top_scores = [item[1] for item in sorted_items[:top_k]]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(range(len(top_modules)), top_scores, color='steelblue')
    ax.set_yticks(range(len(top_modules)))
    ax.set_yticklabels(top_modules)
    ax.invert_yaxis()
    ax.set_xlabel('Fisher Score', fontsize=12)
    ax.set_title(f'Top-{top_k} TimeMoE Modules for {concept_name.capitalize()} Concept', 
                 fontsize=14, pad=10)
    ax.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        ax.text(score, i, f' {score:.3f}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f'./visualizations/probe_training_timemoe-50M/{concept_name}_top_modules.pdf'
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ {concept_name} 柱状图已保存到: {save_path}")
    plt.close()


def analyze_concept_map(concept_map: dict):
    """
    分析概念映射的统计特性
    
    Args:
        concept_map: 概念映射字典
    """
    print("\n" + "="*60)
    print("TimeMoE-50M 概念映射分析")
    print("="*60)
    
    for concept, sensitivity in concept_map.items():
        scores = list(sensitivity.values())
        
        print(f"\n[{concept.upper()}]")
        print(f"  均值: {np.mean(scores):.4f}")
        print(f"  标准差: {np.std(scores):.4f}")
        print(f"  最大值: {np.max(scores):.4f}")
        print(f"  最小值: {np.min(scores):.4f}")
        
        # Top-3敏感模块
        sorted_items = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top-3 敏感模块:")
        for name, score in sorted_items[:3]:
            simplified_name = name.replace('model.layers.', 'L')
            print(f"    {simplified_name}: {score:.4f}")


if __name__ == '__main__':
    # 加载TimeMoE概念映射
    concept_map_path = './concept_maps/timemoe_50m_concept_map.pkl'
    
    if not Path(concept_map_path).exists():
        print(f"错误: 未找到概念映射文件 {concept_map_path}")
        print("请先运行 bash scripts/timemoe-50M/02_train_timemoe_concept_probe.sh")
        exit(1)
    
    print("加载TimeMoE-50M概念映射...")
    with open(concept_map_path, 'rb') as f:
        concept_map = pickle.load(f)
    
    # 统计分析
    analyze_concept_map(concept_map)
    
    # 生成热图
    print("\n生成可视化...")
    visualize_concept_sensitivity_heatmap(
        concept_map,
        save_path='./visualizations/probe_training_timemoe-50M/concept_sensitivity_heatmap.pdf'
    )
    
    # 为每个概念生成柱状图
    for concept in concept_map.keys():
        visualize_concept_bar_chart(
            concept_map, 
            concept, 
            top_k=15,  # TimeMoE有更多层，显示top-15
            save_path=f'./visualizations/probe_training_timemoe-50M/{concept}_top_modules.pdf'
        )
    
    print("\n✓ 可视化完成! 结果保存在 ./visualizations/probe_training_timemoe-50M/")
