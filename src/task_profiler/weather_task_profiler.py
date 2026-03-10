"""
Weather任务画像器
复用ETTh1的代码，修改数据路径
"""

import sys
sys.path.append('src')

import numpy as np

from task_profiler.etth1_task_profiler import ETTh1TaskProfiler


class WeatherTaskProfiler(ETTh1TaskProfiler):
    """Weather数据集的任务概念向量计算"""

    def _infer_steps_per_day(self) -> int:
        """从 date 列推断一天有多少步（weather.csv 通常是 10min => 144）。"""
        if self.data is None:
            return 24

        if 'date' not in self.data.columns:
            return 24

        try:
            dt = np.diff(self.data['date'].astype('datetime64[ns]').values[:50]).astype('timedelta64[s]')
            dt_seconds = int(np.median(dt).astype(int))
            if dt_seconds <= 0:
                return 24
            return max(1, int(round(86400 / dt_seconds)))
        except Exception:
            return 24

    def get_task_vector(self, target_column: str = 'OT', sample_length: int = 5000):
        """按 Weather 的时间粒度自适配 period 参数。"""
        if self.data is None:
            raise ValueError("请先加载数据或提供data_path")

        if target_column not in self.data.columns:
            print(f"⚠️ 列 '{target_column}' 不存在，使用最后一列")
            series = self.data.iloc[:sample_length, -1].values
        else:
            series = self.data[target_column].iloc[:sample_length].values

        steps_per_day = self._infer_steps_per_day()
        steps_per_week = steps_per_day * 7

        print(f"\n分析列: {target_column}, 长度: {len(series)}")
        print(f"  推断时间粒度: {steps_per_day} steps/day")

        print("  计算周期性...")
        periodicity = self.compute_periodicity_score(series, freq='custom')
        # 覆盖 ETTh1 的 key_lags 逻辑：用推断的 day/week
        try:
            from statsmodels.tsa.stattools import acf
            key_lags = [steps_per_day, 2 * steps_per_day, steps_per_week]
            max_lag = min(max(key_lags) + 1, len(series) // 2)
            acf_values = acf(series, nlags=max_lag, fft=True)
            valid_lags = [lag for lag in key_lags if lag < len(acf_values)]
            periodicity = float(np.clip(np.mean([abs(acf_values[lag]) for lag in valid_lags]), 0, 1)) if valid_lags else 0.0
        except Exception:
            periodicity = float(periodicity)

        print("  计算趋势性...")
        trend = self.compute_trend_score(series, period=steps_per_day)

        print("  计算季节性...")
        seasonality = self.compute_seasonality_score(series, period=steps_per_day)

        print("  计算波动性...")
        volatility = self.compute_volatility_score(series, period=steps_per_day)

        print("  计算平稳性...")
        stationarity = self.compute_stationarity_score(series)

        return {
            'periodicity': float(periodicity),
            'trend': float(trend),
            'seasonality': float(seasonality),
            'volatility': float(volatility),
            'stationarity': float(stationarity),
        }


if __name__ == '__main__':
    print("="*60)
    print("Weather 任务概念向量计算")
    print("="*60)

    profiler = WeatherTaskProfiler(data_path='./dataset/weather/weather.csv')

    task_vector = profiler.get_task_vector(target_column='OT', sample_length=5000)

    print("\n" + "="*60)
    print("任务概念向量 Task[c] - Weather")
    print("="*60)
    for concept, score in task_vector.items():
        bar = '█' * int(score * 20)
        print(f"  {concept:15s}: {score:.4f} {bar}")

    profiler.save_task_vector(
        task_vector,
        save_path='./task_vectors/weather_task_vector.pkl'
    )

    print("\n✓ 完成!")
