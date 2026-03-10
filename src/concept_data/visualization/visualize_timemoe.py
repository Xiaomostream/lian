"""
TimeMoE概念数据可视化脚本
"""
import sys
import numpy as np
from pathlib import Path

# 使用相对导入（同目录下的文件）
from visualize_concept_data import (
    visualize_concept_samples,
    visualize_concept_statistics,
    test_concept_data_quality
)

# 加载TimeMoE概念数据
print("加载TimeMoE概念数据...")
concept_datasets = {}
data_dir = Path('./concept_data/timemoe_data')

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
    print("\n错误: 未找到概念数据，请先运行 generate_timemoe_concepts.bat")
    sys.exit(1)

# 质量检查
test_concept_data_quality(concept_datasets)

# 生成可视化
save_dir = './visualizations/concept_data_timemoe-50M_png'
print(f"\n生成可视化到 {save_dir}...")
visualize_concept_samples(concept_datasets, save_dir=save_dir)
visualize_concept_statistics(concept_datasets, save_dir=save_dir)

print(f"\n✓ 完成! 可视化结果保存在 {save_dir}/")
