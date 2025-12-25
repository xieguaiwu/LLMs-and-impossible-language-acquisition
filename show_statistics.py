import json
import matplotlib.pyplot as plt
import numpy as np

# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 加载四个训练结果文件
data_fixed = read_json('./json_results/training_results_fixed_distance.json')
data_parity = read_json('./json_results/training_results_parity.json')
data_natural = read_json('./json_results/training_results_natural.json')
data_reversed = read_json('./json_results/training_results_reversed.json')

# 提取 loss 值
loss_fixed = data_fixed['losses']
loss_parity = data_parity['losses']
loss_natural = data_natural['losses']
loss_reversed = data_reversed['losses']

# 获取步数
steps_fixed = len(loss_fixed)
steps_parity = len(loss_parity)
steps_natural = len(loss_natural)
steps_reversed = len(loss_reversed)

print(f"训练步数统计:")
print(f"  Natural: {steps_natural} 步")
print(f"  Fixed Distance: {steps_fixed} 步")
print(f"  Parity: {steps_parity} 步")
print(f"  Reversed: {steps_reversed} 步")

# 1. 所有数据的比较（使用对齐的x轴）
plt.figure(figsize=(14, 8))

# 创建对齐的x轴
max_steps = max(steps_natural, steps_fixed, steps_parity, steps_reversed)

# 为每个数据集创建对应的x轴坐标
x_natural = np.linspace(0, max_steps, steps_natural)
x_fixed = np.linspace(0, max_steps, steps_fixed)
x_parity = np.linspace(0, max_steps, steps_parity)
x_reversed = np.linspace(0, max_steps, steps_reversed)

plt.plot(x_natural, loss_natural, label=f'Natural (Steps: {steps_natural}, Final: {loss_natural[-1]:.4f})', 
         linewidth=2.5, color='blue')
plt.plot(x_fixed, loss_fixed, label=f'Fixed Distance (Steps: {steps_fixed}, Final: {loss_fixed[-1]:.4f})', 
         linewidth=2, color='green')
plt.plot(x_parity, loss_parity, label=f'Parity (Steps: {steps_parity}, Final: {loss_parity[-1]:.4f})', 
         linewidth=2, color='orange')
plt.plot(x_reversed, loss_reversed, label=f'Reversed (Steps: {steps_reversed}, Final: {loss_reversed[-1]:.4f})', 
         linewidth=2, color='red')

plt.title('Loss Comparison Across All Training Datasets (Aligned)', fontsize=16, fontweight='bold')
plt.xlabel('Normalized Training Steps', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('all_loss_comparison_aligned.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Natural vs Fixed Distance (使用最小步数对齐)
plt.figure(figsize=(14, 8))

min_steps = min(steps_natural, steps_fixed)
x_common = np.arange(min_steps)

# 截取到相同步数
loss_natural_trunc = loss_natural[:min_steps]
loss_fixed_trunc = loss_fixed[:min_steps]

plt.plot(x_common, loss_natural_trunc, label=f'Natural (First {min_steps} steps, Final: {loss_natural_trunc[-1]:.4f})', 
         linewidth=3, color='blue')
plt.plot(x_common, loss_fixed_trunc, label=f'Fixed Distance (First {min_steps} steps, Final: {loss_fixed_trunc[-1]:.4f})', 
         linewidth=3, color='green', linestyle='--')

plt.title(f'Loss Comparison: Natural vs Fixed Distance (First {min_steps} Steps)', fontsize=16, fontweight='bold')
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('natural_vs_fixed_distance_aligned.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Natural vs Parity (使用最小步数对齐)
plt.figure(figsize=(14, 8))

min_steps = min(steps_natural, steps_parity)
x_common = np.arange(min_steps)

# 截取到相同步数
loss_natural_trunc = loss_natural[:min_steps]
loss_parity_trunc = loss_parity[:min_steps]

plt.plot(x_common, loss_natural_trunc, label=f'Natural (First {min_steps} steps, Final: {loss_natural_trunc[-1]:.4f})', 
         linewidth=3, color='blue')
plt.plot(x_common, loss_parity_trunc, label=f'Parity (First {min_steps} steps, Final: {loss_parity_trunc[-1]:.4f})', 
         linewidth=3, color='orange', linestyle='--')

plt.title(f'Loss Comparison: Natural vs Parity (First {min_steps} Steps)', fontsize=16, fontweight='bold')
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('natural_vs_parity_aligned.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Natural vs Reversed (使用最小步数对齐)
plt.figure(figsize=(14, 8))

min_steps = min(steps_natural, steps_reversed)
x_common = np.arange(min_steps)

# 截取到相同步数
loss_natural_trunc = loss_natural[:min_steps]
loss_reversed_trunc = loss_reversed[:min_steps]

plt.plot(x_common, loss_natural_trunc, label=f'Natural (First {min_steps} steps, Final: {loss_natural_trunc[-1]:.4f})', 
         linewidth=3, color='blue')
plt.plot(x_common, loss_reversed_trunc, label=f'Reversed (First {min_steps} steps, Final: {loss_reversed_trunc[-1]:.4f})', 
         linewidth=3, color='red', linestyle='--')

plt.title(f'Loss Comparison: Natural vs Reversed (First {min_steps} Steps)', fontsize=16, fontweight='bold')
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('natural_vs_reversed_aligned.png', dpi=300, bbox_inches='tight')
plt.show()

# 新增：创建比较起始性能的图
plt.figure(figsize=(14, 8))

# 比较前50步（所有模型都有至少50步）
compare_steps = 50
x_compare = np.arange(compare_steps)

plt.plot(x_compare, loss_natural[:compare_steps], label=f'Natural', linewidth=3, color='blue')
plt.plot(x_compare, loss_fixed[:compare_steps], label=f'Fixed Distance', linewidth=3, color='green', linestyle='--')
plt.plot(x_compare, loss_parity[:compare_steps], label=f'Parity', linewidth=3, color='orange', linestyle='--')
plt.plot(x_compare, loss_reversed[:compare_steps], label=f'Reversed', linewidth=3, color='red', linestyle='--')

plt.title(f'Loss Comparison: First {compare_steps} Steps of Training', fontsize=16, fontweight='bold')
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Loss Value', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('first_50_steps_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("所有图表已生成并保存为PNG文件。")
print(f"\n训练步数统计:")
print(f"  Natural: {steps_natural} 步 (最终Loss: {loss_natural[-1]:.4f})")
print(f"  Fixed Distance: {steps_fixed} 步 (最终Loss: {loss_fixed[-1]:.4f})")
print(f"  Parity: {steps_parity} 步 (最终Loss: {loss_parity[-1]:.4f})")
print(f"  Reversed: {steps_reversed} 步 (最终Loss: {loss_reversed[-1]:.4f})")

print(f"\n对比分析:")
print(f"  Natural vs Fixed Distance: 对比前{min(steps_natural, steps_fixed)}步")
print(f"  Natural vs Parity: 对比前{min(steps_natural, steps_parity)}步")
print(f"  Natural vs Reversed: 对比前{min(steps_natural, steps_reversed)}步")
