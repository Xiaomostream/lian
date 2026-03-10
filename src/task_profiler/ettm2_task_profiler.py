"""
ETTm2任务画像器
复用ETTh1的代码，修改数据路径
"""

import sys
sys.path.append('src')

from task_profiler.etth1_task_profiler import ETTh1TaskProfiler


if __name__ == '__main__':
    print("="*60)
    print("ETTm2 任务概念向量计算")
    print("="*60)
    
    # 创建画像器，指定ETTm2数据
    profiler = ETTh1TaskProfiler(data_path='./datasets/ETT-small/ETTm2.csv')
    
    # 计算任务向量
    task_vector = profiler.get_task_vector(target_column='OT', sample_length=5000)
    
    # 显示结果
    print("\n" + "="*60)
    print("任务概念向量 Task[c] - ETTm2")
    print("="*60)
    for concept, score in task_vector.items():
        bar = '█' * int(score * 20)
        print(f"  {concept:15s}: {score:.4f} {bar}")
    
    # 保存
    profiler.save_task_vector(
        task_vector, 
        save_path='./task_vectors/ettm2_task_vector.pkl'
    )
    
    print("\n✓ 完成!")
