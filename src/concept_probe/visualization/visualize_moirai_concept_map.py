"""
Moirai概念映射可视化工具
生成Moirai的概念敏感度热图和柱状图
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def simplify_layer_name(name):
    """简化Moirai层名，方便显示"""
    name = name.replace('encoder.layers.', 'L')
    name = name.replace('.self_attn', '.Attn')
    name = name.replace('.ffn', '.FFN')
    return name


def visualize_concept_sensitivity_heatmap(concept_map, save_path):
    """绘制概念敏感度热图"""
    concepts = list(concept_map.keys())
    modules = list(concept_map[concepts[0]].keys())
    
    matrix = np.zeros((len(modules), len(concepts)))
    for j, concept in enumerate(concepts):
        for i, module in enumerate(modules):
            matrix[i, j] = concept_map[concept][module]
    
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)
    
    sns.set(font_scale=1.2, style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, max(8, len(modules) * 0.4)))
    
    sns.heatmap(
        matrix, cmap='viridis', ax=ax,
        cbar_kws={'label': 'Normalized Fisher Score'},
        xticklabels=[c.capitalize() for c in concepts],
        yticklabels=[simplify_layer_name(m) for m in modules],
        annot=True, fmt='.2f'
    )
    
    ax.set_xlabel('Concept', fontsize=14, labelpad=10)
    ax.set_ylabel('Moirai Module', fontsize=14, labelpad=10)
    ax.set_title('Moirai Concept Sensitivity Heatmap', fontsize=16, pad=15)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 热图已保存到: {save_path}")
    plt.close()


def visualize_concept_bar_chart(concept_map, concept_name, top_k=8, save_path=None):
    """绘制某个概念的top-k敏感模块柱状图"""
    sensitivity = concept_map[concept_name]
    sorted_items = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    
    top_modules = [simplify_layer_name(item[0]) for item in sorted_items[:top_k]]
    top_scores = [item[1] for item in sorted_items[:top_k]]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(top_modules)), top_scores, color='steelblue')
    ax.set_yticks(range(len(top_modules)))
    ax.set_yticklabels(top_modules)
    ax.invert_yaxis()
    ax.set_xlabel('Fisher Score', fontsize=12)
    ax.set_title(f'Top-{top_k} Moirai Modules for {concept_name.capitalize()} Concept', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        ax.text(score, i, f' {score:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    if save_path is None:
        save_path = f'./visualizations/probe_training_moirai/{concept_name}_top_modules.pdf'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ {concept_name} 柱状图已保存到: {save_path}")
    plt.close()


def analyze_concept_map(concept_map):
    """分析概念映射统计特性"""
    print("\n" + "="*60)
    print("Moirai 概念映射分析")
    print("="*60)
    
    for concept, sensitivity in concept_map.items():
        scores = list(sensitivity.values())
        print(f"\n[{concept.upper()}]")
        print(f"  均值: {np.mean(scores):.4f}")
        print(f"  标准差: {np.std(scores):.4f}")
        print(f"  最大值: {np.max(scores):.4f}")
        print(f"  最小值: {np.min(scores):.4f}")
        
        sorted_items = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top-3 敏感模块:")
        for name, score in sorted_items[:3]:
            print(f"    {simplify_layer_name(name)}: {score:.4f}")


if __name__ == '__main__':
    concept_map_path = './concept_maps/moirai_base_concept_map.pkl'
    
    if not Path(concept_map_path).exists():
        print(f"错误: 未找到概念映射文件 {concept_map_path}")
        print("请先运行 bash scripts/moirai/02_train_moirai_concept_probe.sh")
        exit(1)
    
    print("加载Moirai概念映射...")
    with open(concept_map_path, 'rb') as f:
        concept_map = pickle.load(f)
    
    analyze_concept_map(concept_map)
    
    print("\n生成可视化...")
    visualize_concept_sensitivity_heatmap(
        concept_map,
        save_path='./visualizations/probe_training_moirai/concept_sensitivity_heatmap.pdf'
    )
    
    for concept in concept_map.keys():
        visualize_concept_bar_chart(
            concept_map, concept, top_k=8,
            save_path=f'./visualizations/probe_training_moirai/{concept}_top_modules.pdf'
        )
    
    print("\n✓ 可视化完成! 结果保存在 ./visualizations/probe_training_moirai/")
