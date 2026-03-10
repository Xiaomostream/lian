"""
电力负荷概念数据生成器
基于 representations-in-tsfms/steering/steertool/data_generator.py 改编
针对ETTh1数据集特征优化
"""

import numpy as np
import torch
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class ConceptConfig:
    """概念数据配置"""
    length: int = 512  # 序列长度 (与ETTh1的seq_len=512对齐)
    num_samples: int = 1000  # 每类概念生成的样本数
    

class PowerLoadConceptGenerator:
    """
    电力负荷概念数据生成器
    生成5类时序概念数据用于概念探针训练
    """
    
    def __init__(self, config: ConceptConfig = None):
        self.config = config or ConceptConfig()
    
    def generate_periodicity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        周期性概念: 模拟电力负荷的日周期(24h) + 周周期(168h)
        
        Returns:
            positive_samples: (num_samples, length) 有周期性的数据
            negative_samples: (num_samples, length) 无周期性的数据(纯噪声)
        """
        positive_samples = []
        negative_samples = []
        
        for _ in range(self.config.num_samples):
            t = np.arange(self.config.length)
            
            # 正样本: 日周期 + 周周期 + 噪声
            # 日周期 (24h): 模拟白天高、夜晚低的用电规律
            daily_period = 24
            daily_amp = np.random.uniform(0.5, 2.0)
            daily_component = daily_amp * np.sin(2 * np.pi * t / daily_period)
            
            # 周周期 (168h): 模拟工作日vs周末的差异
            weekly_period = 168
            weekly_amp = np.random.uniform(0.2, 1.0)
            weekly_component = weekly_amp * np.sin(2 * np.pi * t / weekly_period)
            
            # 添加噪声
            noise = np.random.normal(0, 0.1, self.config.length)
            
            positive_sample = daily_component + weekly_component + noise
            positive_samples.append(positive_sample)
            
            # 负样本: 纯噪声 (无周期性)
            negative_sample = np.random.normal(0, 1.0, self.config.length)
            negative_samples.append(negative_sample)
        
        return np.array(positive_samples), np.array(negative_samples)
    
    def generate_trend(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        趋势性概念: 模拟电力负荷的增长/下降趋势
        
        Returns:
            positive_samples: 有明显趋势的数据
            negative_samples: 无趋势的平稳数据
        """
        positive_samples = []
        negative_samples = []
        
        for _ in range(self.config.num_samples):
            t = np.arange(self.config.length)
            
            # 正样本: 线性/指数趋势
            trend_type = np.random.choice(['linear', 'exp', 'piecewise'])
            
            if trend_type == 'linear':
                slope = np.random.uniform(-0.01, 0.01)
                trend = slope * t
            elif trend_type == 'exp':
                rate = np.random.uniform(-0.002, 0.002)
                trend = np.exp(rate * t) - 1
            else:  # piecewise
                breakpoint = np.random.randint(self.config.length//4, 3*self.config.length//4)
                slope1 = np.random.uniform(-0.01, 0.01)
                slope2 = np.random.uniform(-0.01, 0.01)
                trend = np.concatenate([
                    slope1 * t[:breakpoint],
                    slope1 * breakpoint + slope2 * (t[breakpoint:] - breakpoint)
                ])
            
            noise = np.random.normal(0, 0.1, self.config.length)
            positive_sample = trend + noise
            positive_samples.append(positive_sample)
            
            # 负样本: 零均值噪声
            negative_sample = np.random.normal(0, 0.5, self.config.length)
            negative_samples.append(negative_sample)
        
        return np.array(positive_samples), np.array(negative_samples)
    
    def generate_seasonality(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        季节性概念: 模拟长周期模式 (季节变化)
        
        Returns:
            positive_samples: 有季节性的数据
            negative_samples: 无季节性的数据
        """
        positive_samples = []
        negative_samples = []
        
        for _ in range(self.config.num_samples):
            t = np.arange(self.config.length)
            
            # 正样本: 长周期 + 幅值随时间变化
            long_period = np.random.choice([168, 336, 672])  # 1周、2周、4周
            base_amp = np.random.uniform(0.5, 1.5)
            
            # 幅值调制 (模拟季节性强度变化)
            amp_modulation = 1 + 0.3 * np.sin(2 * np.pi * t / (2 * long_period))
            seasonal_component = base_amp * amp_modulation * np.sin(2 * np.pi * t / long_period)
            
            noise = np.random.normal(0, 0.1, self.config.length)
            positive_sample = seasonal_component + noise
            positive_samples.append(positive_sample)
            
            # 负样本: 无规律噪声
            negative_sample = np.random.normal(0, 0.8, self.config.length)
            negative_samples.append(negative_sample)
        
        return np.array(positive_samples), np.array(negative_samples)
    
    def generate_volatility(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        波动性概念: 模拟方差随时间变化 (GARCH式)
        
        Returns:
            positive_samples: 高波动性数据
            negative_samples: 低波动性数据
        """
        positive_samples = []
        negative_samples = []
        
        for _ in range(self.config.num_samples):
            # 正样本: GARCH式方差变化
            volatility = np.ones(self.config.length)
            for i in range(1, self.config.length):
                # 简化的GARCH(1,1): σ_t^2 = α + β*σ_{t-1}^2 + γ*ε_{t-1}^2
                alpha = 0.1
                beta = 0.85
                gamma = 0.05
                epsilon_prev = np.random.normal(0, np.sqrt(volatility[i-1]))
                volatility[i] = alpha + beta * volatility[i-1] + gamma * (epsilon_prev ** 2)
            
            positive_sample = np.random.normal(0, np.sqrt(volatility))
            positive_samples.append(positive_sample)
            
            # 负样本: 恒定方差
            negative_sample = np.random.normal(0, 0.5, self.config.length)
            negative_samples.append(negative_sample)
        
        return np.array(positive_samples), np.array(negative_samples)
    
    def generate_stationarity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        平稳性概念: 平稳 vs 非平稳
        
        Returns:
            positive_samples: 平稳数据
            negative_samples: 非平稳数据 (随机游走)
        """
        positive_samples = []
        negative_samples = []
        
        for _ in range(self.config.num_samples):
            # 正样本: AR(1)平稳过程
            phi = np.random.uniform(0.3, 0.7)  # AR系数 < 1 保证平稳
            white_noise = np.random.normal(0, 0.5, self.config.length)
            positive_sample = np.zeros(self.config.length)
            for i in range(1, self.config.length):
                positive_sample[i] = phi * positive_sample[i-1] + white_noise[i]
            positive_samples.append(positive_sample)
            
            # 负样本: 随机游走 (非平稳)
            drift = np.random.uniform(-0.01, 0.01)
            white_noise = np.random.normal(0, 0.3, self.config.length)
            negative_sample = np.cumsum(drift + white_noise)
            negative_samples.append(negative_sample)
        
        return np.array(positive_samples), np.array(negative_samples)
    
    def generate_all_concepts(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        生成所有5类概念数据
        
        Returns:
            concept_datasets: {
                'periodicity': (positive, negative),
                'trend': (positive, negative),
                ...
            }
        """
        print("生成概念数据...")
        
        concept_datasets = {
            'periodicity': self.generate_periodicity(),
            'trend': self.generate_trend(),
            'seasonality': self.generate_seasonality(),
            'volatility': self.generate_volatility(),
            'stationarity': self.generate_stationarity()
        }
        
        print(f"✓ 完成! 每类概念生成 {self.config.num_samples} 个正样本 + {self.config.num_samples} 个负样本")
        print(f"  数据形状: ({self.config.num_samples}, {self.config.length})")
        
        return concept_datasets
    
    def save_datasets(self, concept_datasets: Dict, save_dir: str = './concept_data/chronos-bolt-small'):
        """保存概念数据集"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for concept_name, (pos_data, neg_data) in concept_datasets.items():
            np.save(f'{save_dir}/{concept_name}_positive.npy', pos_data)
            np.save(f'{save_dir}/{concept_name}_negative.npy', neg_data)
        
        print(f"✓ 数据已保存到 {save_dir}/")


if __name__ == '__main__':
    # 测试代码
    config = ConceptConfig(length=512, num_samples=1000)
    generator = PowerLoadConceptGenerator(config)
    
    # 生成所有概念数据
    datasets = generator.generate_all_concepts()
    
    # 保存数据
    generator.save_datasets(datasets, save_dir='./concept_data/chronos-bolt-small')
    
    # 验证数据
    print("\n数据验证:")
    for name, (pos, neg) in datasets.items():
        print(f"  {name}: pos={pos.shape}, neg={neg.shape}, pos_mean={pos.mean():.3f}, neg_mean={neg.mean():.3f}")
