import json
import matplotlib.pyplot as plt
import numpy as np
import os

# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载四个训练结果文件
data_natural = read_json('./statistics/json_baby/training_results_natural.json')
data_parity = read_json('./statistics/json_baby/training_results_parity_negation.json')
data_reversed = read_json('./statistics/json_baby/training_results_reversed.json')

# 提取 perplexity 值
perplexity_natural = data_natural['perplexities']
perplexity_parity = data_parity['perplexities']
perplexity_reversed = data_reversed['perplexities']

# 获取步数
steps_natural = len(perplexity_natural)
steps_parity = len(perplexity_parity)
steps_reversed = len(perplexity_reversed)

print(f"训练步数统计:")
print(f"  Natural: {steps_natural} 步 (最终Perplexity: {perplexity_natural[-1]:.4f})")
print(f"  Parity Negation: {steps_parity} 步 (最终Perplexity: {perplexity_parity[-1]:.4f})")
print(f"  Reversed: {steps_reversed} 步 (最终Perplexity: {perplexity_reversed[-1]:.4f})")

print(f"\n最小困惑度统计:")
print(f"  Natural: {data_natural['min_perplexity']:.4f}")
print(f"  Parity Negation: {data_parity['min_perplexity']:.4f}")
print(f"  Reversed: {data_reversed['min_perplexity']:.4f}")

print(f"\n训练时间统计:")
print(f"  Natural: {data_natural['training_time']:.2f} 秒")
print(f"  Parity Negation: {data_parity['training_time']:.2f} 秒")
print(f"  Reversed: {data_reversed['training_time']:.2f} 秒")

# 1. 所有数据的比较（使用对齐的x轴）
plt.figure(figsize=(14, 8))

# 创建对齐的x轴
max_steps = max(steps_natural, steps_parity, steps_reversed)

# 为每个数据集创建对应的x轴坐标
x_natural = np.linspace(0, max_steps, steps_natural)
x_parity = np.linspace(0, max_steps, steps_parity)
x_reversed = np.linspace(0, max_steps, steps_reversed)

plt.plot(x_natural, perplexity_natural, 
         label=f'Natural (Steps: {steps_natural})', 
         linewidth=2.5, color='blue')
plt.plot(x_parity, perplexity_parity, 
         label=f'Parity Negation (Steps: {steps_parity})', 
         linewidth=2, color='orange')
plt.plot(x_reversed, perplexity_reversed, 
         label=f'Reversed (Steps: {steps_reversed})', 
         linewidth=2, color='red')

plt.title('Perplexity Comparison Across All Training Datasets (Aligned)', fontsize=16, fontweight='bold')
plt.xlabel('Normalized Training Steps', fontsize=14)
plt.ylabel('Perplexity Value', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 使用对数刻度以便观察低值区域
plt.tight_layout()
plt.savefig('all_perplexity_comparison_aligned.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Natural vs Parity Negation (使用最小步数对齐)
plt.figure(figsize=(14, 8))

min_steps = min(steps_natural, steps_parity)
x_common = np.arange(min_steps)

# 截取到相同步数
perplexity_natural_trunc = perplexity_natural[:min_steps]
perplexity_parity_trunc = perplexity_parity[:min_steps]

plt.plot(x_common, perplexity_natural_trunc, 
         label=f'Natural (First {min_steps} steps)', 
         linewidth=3, color='blue')
plt.plot(x_common, perplexity_parity_trunc, 
         label=f'Parity Negation (First {min_steps} steps)', 
         linewidth=3, color='orange', linestyle='--')

plt.title(f'Perplexity Comparison: Natural vs Parity Negation (First {min_steps} Steps)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Perplexity Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('natural_vs_parity_negation_perplexity.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Natural vs Reversed (使用最小步数对齐)
plt.figure(figsize=(14, 8))

min_steps = min(steps_natural, steps_reversed)
x_common = np.arange(min_steps)

# 截取到相同步数
perplexity_natural_trunc = perplexity_natural[:min_steps]
perplexity_reversed_trunc = perplexity_reversed[:min_steps]

plt.plot(x_common, perplexity_natural_trunc, 
         label=f'Natural (First {min_steps} steps)', 
         linewidth=3, color='blue')
plt.plot(x_common, perplexity_reversed_trunc, 
         label=f'Reversed (First {min_steps} steps)', 
         linewidth=3, color='red', linestyle='--')

plt.title(f'Perplexity Comparison: Natural vs Reversed (First {min_steps} Steps)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Perplexity Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('natural_vs_reversed_perplexity.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 所有组在前50步的快速收敛图
plt.figure(figsize=(14, 8))

# 比较前50步（所有模型都有至少50步）
compare_steps = 50
x_compare = np.arange(compare_steps)

plt.plot(x_compare, perplexity_natural[:compare_steps], 
         label=f'Natural', 
         linewidth=3, color='blue')
plt.plot(x_compare, perplexity_parity[:compare_steps], 
         label=f'Parity Negation', 
         linewidth=3, color='orange', linestyle='--')
plt.plot(x_compare, perplexity_reversed[:compare_steps], 
         label=f'Reversed', 
         linewidth=3, color='red', linestyle='--')

plt.title(f'Perplexity Comparison: First {compare_steps} Steps of Training', 
          fontsize=16, fontweight='bold')
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Perplexity Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('first_50_steps_perplexity_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 最终困惑度对比图（条形图）
plt.figure(figsize=(12, 8))

datasets = ['Natural', 'Parity Negation', 'Reversed']
final_perplexities = [
    perplexity_natural[-1],
    perplexity_parity[-1],
    perplexity_reversed[-1]
]
min_perplexities = [
    data_natural['min_perplexity'],
    data_parity['min_perplexity'],
    data_reversed['min_perplexity']
]

x = np.arange(len(datasets))
width = 0.35

plt.bar(x - width/2, final_perplexities, width, label='Final Perplexity', 
        color=['blue', 'green', 'orange', 'red'], alpha=0.8)
plt.bar(x + width/2, min_perplexities, width, label='Minimum Perplexity', 
        color=['blue', 'green', 'orange', 'red'], alpha=0.6, hatch='//')

plt.xlabel('Dataset', fontsize=14)
plt.ylabel('Perplexity Value', fontsize=14)
plt.title('Final vs Minimum Perplexity Comparison', fontsize=16, fontweight='bold')
plt.xticks(x, datasets, rotation=15, ha='right')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, (final, min_val) in enumerate(zip(final_perplexities, min_perplexities)):
    plt.text(i - width/2, final + 0.02, f'{final:.3f}', ha='center', va='bottom', fontsize=10)
    plt.text(i + width/2, min_val + 0.02, f'{min_val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('final_min_perplexity_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*70)
print("所有图表已生成并保存为PNG文件。")
print("="*70)

print(f"\n对比分析:")
print(f"  Natural vs Parity Negation: 对比前{min(steps_natural, steps_parity)}步")
print(f"  Natural vs Reversed: 对比前{min(steps_natural, steps_reversed)}步")

print(f"\n收敛率分析 (初始/最终):")
print(f"  Natural: {perplexity_natural[0]/perplexity_natural[-1]:.2f}x")
print(f"  Parity Negation: {perplexity_parity[0]/perplexity_parity[-1]:.2f}x")
print(f"  Reversed: {perplexity_reversed[0]/perplexity_reversed[-1]:.2f}x")

print(f"\n训练效率 (训练时间/最终困惑度):")
print(f"  Natural: {data_natural['training_time']/perplexity_natural[-1]:.2f} 秒/单位困惑度")
print(f"  Parity Negation: {data_parity['training_time']/perplexity_parity[-1]:.2f} 秒/单位困惑度")
print(f"  Reversed: {data_reversed['training_time']/perplexity_reversed[-1]:.2f} 秒/单位困惑度")
