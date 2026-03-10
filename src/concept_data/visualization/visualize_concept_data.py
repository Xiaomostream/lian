"""
概念数据可视化工具
用于检查生成的概念数据质量
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def visualize_concept_samples(concept_datasets, save_dir='./visualizations'):
    """
    可视化每类概念的样本数据
    
    Args:
        concept_datasets: dict, {'concept_name': (positive, negative)}
        save_dir: str, 保存图片的目录
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    sns.set_style("whitegrid")
    
    for concept_name, (pos_data, neg_data) in concept_datasets.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Concept: {concept_name.upper()}', fontsize=16, fontweight='bold')
        
        # 绘制3个正样本
        for i in range(3):
            axes[0, i].plot(pos_data[i], color='steelblue', linewidth=1.5)
            axes[0, i].set_title(f'Positive Sample {i+1}', fontsize=12)
            axes[0, i].set_xlabel('Time Steps')
            axes[0, i].set_ylabel('Value')
            axes[0, i].grid(True, alpha=0.3)
        
        # 绘制3个负样本
        for i in range(3):
            axes[1, i].plot(neg_data[i], color='coral', linewidth=1.5)
            axes[1, i].set_title(f'Negative Sample {i+1}', fontsize=12)
            axes[1, i].set_xlabel('Time Steps')
            axes[1, i].set_ylabel('Value')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = Path(save_dir) / f'{concept_name}_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 保存: {save_path}")
        plt.close()


def visualize_concept_statistics(concept_datasets, save_dir='./visualizations'):
    """
    可视化概念数据的统计特征对比
    
    Args:
        concept_datasets: dict
        save_dir: str
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Concept Data Statistics Comparison', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, (concept_name, (pos_data, neg_data)) in enumerate(concept_datasets.items()):
        ax = axes[idx]
        
        # 计算统计量
        pos_mean = pos_data.mean(axis=1)
        neg_mean = neg_data.mean(axis=1)
        pos_std = pos_data.std(axis=1)
        neg_std = neg_data.std(axis=1)
        
        # 绘制分布
        ax.hist(pos_mean, bins=30, alpha=0.6, label='Positive (mean)', color='steelblue')
        ax.hist(neg_mean, bins=30, alpha=0.6, label='Negative (mean)', color='coral')
        
        ax.set_title(f'{concept_name.capitalize()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Mean Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏第6个子图
    axes[5].axis('off')
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'concept_statistics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 保存: {save_path}")
    plt.close()


def test_concept_data_quality(concept_datasets):
    """
    测试概念数据质量
    
    检查:
    1. 数据形状是否正确
    2. 是否有NaN/Inf值
    3. 正负样本是否有明显差异
    """
    print("\n" + "="*60)
    print("概念数据质量检查")
    print("="*60)
    
    all_passed = True
    
    for concept_name, (pos_data, neg_data) in concept_datasets.items():
        print(f"\n[{concept_name.upper()}]")
        
        # 检查形状
        assert pos_data.shape == neg_data.shape, f"✗ 形状不匹配: {pos_data.shape} vs {neg_data.shape}"
        print(f"  ✓ 形状: {pos_data.shape}")
        
        # 检查NaN/Inf
        if np.isnan(pos_data).any() or np.isnan(neg_data).any():
            print(f"  ✗ 存在NaN值!")
            all_passed = False
        else:
            print(f"  ✓ 无NaN值")
        
        if np.isinf(pos_data).any() or np.isinf(neg_data).any():
            print(f"  ✗ 存在Inf值!")
            all_passed = False
        else:
            print(f"  ✓ 无Inf值")
        
        # 统计差异
        pos_mean = pos_data.mean()
        neg_mean = neg_data.mean()
        pos_std = pos_data.std()
        neg_std = neg_data.std()
        
        print(f"  正样本: mean={pos_mean:.4f}, std={pos_std:.4f}")
        print(f"  负样本: mean={neg_mean:.4f}, std={neg_std:.4f}")
        
        # 简单的可分性检查 (均值差异)
        mean_diff = abs(pos_mean - neg_mean)
        if mean_diff > 0.1:
            print(f"  ✓ 正负样本均值差异明显: {mean_diff:.4f}")
        else:
            print(f"  ⚠️ 正负样本均值差异较小: {mean_diff:.4f}")
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有检查通过!")
    else:
        print("⚠️ 部分检查未通过，请检查数据生成逻辑")
    print("="*60)
    
    return all_passed


if __name__ == '__main__':
    # 加载数据
    print("加载概念数据...")
    concept_datasets = {}
    data_dir = Path('./concept_data/chronos-bolt-small')
    
    for concept_name in ['periodicity', 'trend', 'seasonality', 'volatility', 'stationarity']:
        pos_path = data_dir / f'{concept_name}_positive.npy'
        neg_path = data_dir / f'{concept_name}_negative.npy'
        
        if pos_path.exists() and neg_path.exists():
            pos_data = np.load(pos_path)
            neg_data = np.load(neg_path)
            concept_datasets[concept_name] = (pos_data, neg_data)
            print(f"  ✓ {concept_name}: {pos_data.shape}")
        else:
            print(f"  ✗ {concept_name}: 数据文件不存在")
    
    if not concept_datasets:
        print("\n错误: 未找到概念数据，请先运行 01_generate_concept_data.bat/sh")
        exit(1)
    
    # 质量检查
    test_concept_data_quality(concept_datasets)
    
    # 可视化
    print("\n生成可视化...")
    visualize_concept_samples(concept_datasets)
    visualize_concept_statistics(concept_datasets)
    
    print("\n✓ 完成! 可视化结果保存在 ./visualizations/")
