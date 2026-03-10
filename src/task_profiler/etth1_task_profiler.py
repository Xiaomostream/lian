"""
ETTh1任务画像器
计算ETTh1数据集对5类概念的依赖强度
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.seasonal import STL
from scipy.stats import kendalltau


class ETTh1TaskProfiler:
    """ETTh1数据集的任务概念向量计算"""
    
    def __init__(self, data_path: str = None):
        """
        Args:
            data_path: ETTh1数据路径，如 './datasets/ETTh1.csv'
        """
        if data_path and Path(data_path).exists():
            self.data = pd.read_csv(data_path)
            print(f"✓ 加载数据: {data_path}, 形状: {self.data.shape}")
        else:
            self.data = None
            print("⚠️ 未提供数据路径，将使用手动输入的数据")
    
    def compute_periodicity_score(self, series: np.ndarray, freq: str = 'h') -> float:
        """
        周期性强度: ACF在关键滞后位置的峰值
        
        Args:
            series: 时间序列数据
            freq: 频率 ('h'=小时, 'm'=分钟)
        
        Returns:
            周期性分数 [0, 1]
        """
        # ETTh1是小时数据，关键周期: 24h(日), 168h(周)
        if freq == 'h':
            key_lags = [24, 48, 168]  # 1天, 2天, 1周
        else:
            key_lags = [96, 192, 672]  # 对应15分钟数据
        
        # 计算ACF
        max_lag = min(max(key_lags) + 1, len(series) // 2)
        acf_values = acf(series, nlags=max_lag, fft=True)
        
        # 提取关键滞后位置的ACF值
        valid_lags = [lag for lag in key_lags if lag < len(acf_values)]
        if not valid_lags:
            return 0.0
        
        periodicity_score = np.mean([abs(acf_values[lag]) for lag in valid_lags])
        
        return np.clip(periodicity_score, 0, 1)
    
    def compute_trend_score(self, series: np.ndarray, period: int = 24) -> float:
        """
        趋势性强度: STL趋势分量能量占比 + Mann-Kendall检验
        
        Args:
            series: 时间序列
            period: 季节周期
        
        Returns:
            趋势性分数 [0, 1]
        """
        try:
            # STL分解
            stl = STL(series, period=period, seasonal=13)
            result = stl.fit()
            
            # 趋势能量占比
            trend_energy = np.var(result.trend)
            total_energy = np.var(series)
            trend_ratio = trend_energy / (total_energy + 1e-8)
            
            # Mann-Kendall趋势检验
            tau, p_value = kendalltau(np.arange(len(series)), series)
            mk_score = 1 - p_value if p_value < 0.05 else 0
            
            # 综合得分
            combined_score = 0.7 * trend_ratio + 0.3 * mk_score
            
            return np.clip(combined_score, 0, 1)
        
        except Exception as e:
            print(f"  ⚠️ 趋势计算失败: {e}")
            return 0.5
    
    def compute_seasonality_score(self, series: np.ndarray, period: int = 24) -> float:
        """
        季节性强度: STL季节分量能量占比
        
        Args:
            series: 时间序列
            period: 季节周期
        
        Returns:
            季节性分数 [0, 1]
        """
        try:
            stl = STL(series, period=period, seasonal=13)
            result = stl.fit()
            
            # 季节能量占比
            seasonal_energy = np.var(result.seasonal)
            total_energy = np.var(series)
            seasonal_ratio = seasonal_energy / (total_energy + 1e-8)
            
            return np.clip(seasonal_ratio, 0, 1)
        
        except Exception as e:
            print(f"  ⚠️ 季节性计算失败: {e}")
            return 0.5
    
    def compute_volatility_score(self, series: np.ndarray, period: int = 24) -> float:
        """
        波动性强度: 残差方差变化率
        
        Args:
            series: 时间序列
            period: 基准周期
        
        Returns:
            波动性分数 [0, 1]
        """
        try:
            # STL分解提取残差
            stl = STL(series, period=period, seasonal=13)
            result = stl.fit()
            resid = result.resid
            
            # 滚动方差
            window = period
            rolling_var = pd.Series(resid).rolling(window).var().dropna()
            
            if len(rolling_var) == 0:
                return 0.5
            
            # 方差的方差（归一化）
            var_of_var = rolling_var.var() / (rolling_var.mean() + 1e-8)
            
            # 归一化到[0,1]
            volatility_score = np.tanh(var_of_var)
            
            return np.clip(volatility_score, 0, 1)
        
        except Exception as e:
            print(f"  ⚠️ 波动性计算失败: {e}")
            return 0.5
    
    def compute_stationarity_score(self, series: np.ndarray) -> float:
        """
        平稳性强度: ADF检验p值映射
        
        Args:
            series: 时间序列
        
        Returns:
            平稳性分数 [0, 1]，越大越平稳
        """
        try:
            # ADF检验
            result = adfuller(series, maxlag=48, regression='c')
            p_value = result[1]
            
            # p值越小越平稳，映射到[0,1]
            stationarity_score = 1 - p_value
            
            return np.clip(stationarity_score, 0, 1)
        
        except Exception as e:
            print(f"  ⚠️ 平稳性计算失败: {e}")
            return 0.5
    
    def get_task_vector(
        self, 
        target_column: str = 'OT',
        sample_length: int = 5000
    ) -> Dict[str, float]:
        """
        获取完整的任务概念向量
        
        Args:
            target_column: 目标列名 (ETTh1有'OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL')
            sample_length: 使用的样本长度
        
        Returns:
            Task[c]: {'periodicity': x, 'trend': y, ...}
        """
        if self.data is None:
            raise ValueError("请先加载数据或提供data_path")
        
        # 提取目标列数据
        if target_column not in self.data.columns:
            print(f"⚠️ 列 '{target_column}' 不存在，使用第2列")
            series = self.data.iloc[:sample_length, 1].values
        else:
            series = self.data[target_column].iloc[:sample_length].values
        
        print(f"\n分析列: {target_column}, 长度: {len(series)}")
        
        # 计算各概念分数
        print("  计算周期性...")
        periodicity = self.compute_periodicity_score(series, freq='h')
        
        print("  计算趋势性...")
        trend = self.compute_trend_score(series, period=24)
        
        print("  计算季节性...")
        seasonality = self.compute_seasonality_score(series, period=24)
        
        print("  计算波动性...")
        volatility = self.compute_volatility_score(series, period=24)
        
        print("  计算平稳性...")
        stationarity = self.compute_stationarity_score(series)
        
        task_vector = {
            'periodicity': periodicity,
            'trend': trend,
            'seasonality': seasonality,
            'volatility': volatility,
            'stationarity': stationarity
        }
        
        return task_vector
    
    def save_task_vector(
        self,
        task_vector: Dict,
        save_path: str = './task_vectors/etth1_task_vector.pkl'
    ):
        """保存任务向量"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(task_vector, f)
        
        print(f"\n✓ 任务向量已保存到: {save_path}")
    
    @staticmethod
    def load_task_vector(path: str) -> Dict:
        """加载任务向量"""
        with open(path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    # 示例用法
    print("="*60)
    print("ETTh1 任务概念向量计算")
    print("="*60)
    
    # 创建画像器
    profiler = ETTh1TaskProfiler(data_path='./datasets/ETT-small/ETTh1.csv')
    
    # 计算任务向量
    task_vector = profiler.get_task_vector(target_column='OT', sample_length=5000)
    
    # 显示结果
    print("\n" + "="*60)
    print("任务概念向量 Task[c]")
    print("="*60)
    for concept, score in task_vector.items():
        bar = '█' * int(score * 20)
        print(f"  {concept:15s}: {score:.4f} {bar}")
    
    # 保存
    profiler.save_task_vector(task_vector)
    
    print("\n✓ 完成!")
