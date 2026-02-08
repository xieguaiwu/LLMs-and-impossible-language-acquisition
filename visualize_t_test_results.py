import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_json_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def visualize_t_test_results(data, output_dir):
    """
    Visualize Welch's t-test results with grouped bar charts.
    """
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Separate tests by metric
    loss_tests = [t for t in data['tests'] if t['metric_name'] == 'Loss']
    perplexity_tests = [t for t in data['tests'] if t['metric_name'] == 'Perplexity']

    # Create Loss visualization
    fig, ax = plt.subplots(figsize=(12, 7))

    groups = []
    means_group1 = []
    means_group2 = []
    stds_group1 = []
    stds_group2 = []
    p_values = []
    significance_labels = []
    effect_sizes = []
    effect_labels = []

    x = np.arange(len(loss_tests))
    width = 0.35

    for i, test in enumerate(loss_tests):
        comparison = f"{test['name1'].split()[0]}\nvs\n{test['name2'].split()[0]}"
        groups.append(comparison)
        means_group1.append(test['group1']['mean'])
        means_group2.append(test['group2']['mean'])
        stds_group1.append(test['group1']['std'])
        stds_group2.append(test['group2']['std'])
        p_values.append(test['p_value'])
        significance_labels.append('***' if test['significance'] == 'significant' else 'ns')
        effect_sizes.append(abs(test['cohens_d']))
        effect_labels.append(test['effect_size_category'])

    # Create bars
    bars1 = ax.bar(x - width/2, means_group1, width, yerr=stds_group1,
                   capsize=5, label=test['group1']['name'],
                   color='#3498db', alpha=0.8, error_kw={'elinewidth': 1.5})
    bars2 = ax.bar(x + width/2, means_group2, width, yerr=stds_group2,
                   capsize=5, label=test['group2']['name'],
                   color='#e74c3c', alpha=0.8, error_kw={'elinewidth': 1.5})

    # Add significance labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        max_height = max(bar1.get_height(), bar2.get_height())
        max_std = max(stds_group1[i], stds_group2[i])
        ax.text(i, max_height + max_std + 0.1, significance_labels[i],
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Customize plot
    ax.set_xlabel('Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
    ax.set_title("Welch's T-Test Results: Loss Comparisons",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add text annotations
    info_text = f"α = {data['alpha_level']}\n"
    info_text += "Significance: *** p < 0.05, ns: not significant"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    loss_output = os.path.join(output_dir, 't_test_loss_comparison.png')
    plt.savefig(loss_output, dpi=300, bbox_inches='tight')
    print(f"Loss comparison saved to: {loss_output}")
    plt.close()

    # Create Perplexity visualization
    fig, ax = plt.subplots(figsize=(12, 7))

    groups = []
    means_group1 = []
    means_group2 = []
    stds_group1 = []
    stds_group2 = []
    p_values = []
    significance_labels = []

    for i, test in enumerate(perplexity_tests):
        comparison = f"{test['name1'].split()[0]}\nvs\n{test['name2'].split()[0]}"
        groups.append(comparison)
        means_group1.append(test['group1']['mean'])
        means_group2.append(test['group2']['mean'])
        stds_group1.append(test['group1']['std'])
        stds_group2.append(test['group2']['std'])
        p_values.append(test['p_value'])
        significance_labels.append('***' if test['significance'] == 'significant' else 'ns')

    # Create bars
    bars1 = ax.bar(x - width/2, means_group1, width, yerr=stds_group1,
                   capsize=5, label=test['group1']['name'],
                   color='#2ecc71', alpha=0.8, error_kw={'elinewidth': 1.5})
    bars2 = ax.bar(x + width/2, means_group2, width, yerr=stds_group2,
                   capsize=5, label=test['group2']['name'],
                   color='#f39c12', alpha=0.8, error_kw={'elinewidth': 1.5})

    # Add significance labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        max_height = max(bar1.get_height(), bar2.get_height())
        max_std = max(stds_group1[i], stds_group2[i])
        ax.text(i, max_height + max_std + 0.3, significance_labels[i],
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Customize plot
    ax.set_xlabel('Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel('Perplexity Value', fontsize=12, fontweight='bold')
    ax.set_title("Welch's T-Test Results: Perplexity Comparisons",
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add text annotations
    info_text = f"α = {data['alpha_level']}\n"
    info_text += "Significance: *** p < 0.05, ns: not significant"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    perplexity_output = os.path.join(output_dir, 't_test_perplexity_comparison.png')
    plt.savefig(perplexity_output, dpi=300, bbox_inches='tight')
    print(f"Perplexity comparison saved to: {perplexity_output}")
    plt.close()

    # Create summary table visualization
    fig, (ax_loss, ax_perp) = plt.subplots(1, 2, figsize=(16, 8))

    # Loss table
    ax_loss.axis('tight')
    ax_loss.axis('off')

    loss_table_data = [['Comparison', 'Mean (G1)', 'Std (G1)', 'Mean (G2)', 'Std (G2)',
                       't-stat', 'p-value', "Cohen's d", 'Effect', 'Sig']]
    for test in loss_tests:
        g1 = test['group1']
        g2 = test['group2']
        comparison = f"{g1['name']}\nvs\n{g2['name']}"
        loss_table_data.append([
            comparison,
            f"{g1['mean']:.4f}",
            f"{g1['std']:.4f}",
            f"{g2['mean']:.4f}",
            f"{g2['std']:.4f}",
            f"{test['statistic']:.4f}",
            f"{test['p_value']:.2e}",
            f"{test['cohens_d']:.4f}",
            test['effect_size_category'],
            test['significance']
        ])

    table1 = ax_loss.table(cellText=loss_table_data, cellLoc='center',
                          loc='center', colWidths=[0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    table1.auto_set_font_size(False)
    table1.set_fontsize(8)
    table1.scale(1.2, 2.5)

    # Style header row
    for i in range(len(loss_table_data[0])):
        table1[(0, i)].set_facecolor('#3498db')
        table1[(0, i)].set_text_props(weight='bold', color='white')

    ax_loss.set_title('Loss Tests Summary', fontsize=14, fontweight='bold', pad=10)

    # Perplexity table
    ax_perp.axis('tight')
    ax_perp.axis('off')

    perp_table_data = [['Comparison', 'Mean (G1)', 'Std (G1)', 'Mean (G2)', 'Std (G2)',
                        't-stat', 'p-value', "Cohen's d", 'Effect', 'Sig']]
    for test in perplexity_tests:
        g1 = test['group1']
        g2 = test['group2']
        comparison = f"{g1['name']}\nvs\n{g2['name']}"
        perp_table_data.append([
            comparison,
            f"{g1['mean']:.4f}",
            f"{g1['std']:.4f}",
            f"{g2['mean']:.4f}",
            f"{g2['std']:.4f}",
            f"{test['statistic']:.4f}",
            f"{test['p_value']:.2e}",
            f"{test['cohens_d']:.4f}",
            test['effect_size_category'],
            test['significance']
        ])

    table2 = ax_perp.table(cellText=perp_table_data, cellLoc='center',
                           loc='center', colWidths=[0.25, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1.2, 2.5)

    # Style header row
    for i in range(len(perp_table_data[0])):
        table2[(0, i)].set_facecolor('#2ecc71')
        table2[(0, i)].set_text_props(weight='bold', color='white')

    ax_perp.set_title('Perplexity Tests Summary', fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    table_output = os.path.join(output_dir, 't_test_summary_table.png')
    plt.savefig(table_output, dpi=300, bbox_inches='tight')
    print(f"Summary table saved to: {table_output}")
    plt.close()


def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(base_dir, 'statistics', 'welch_t_test_results.json')
    output_dir = os.path.join(base_dir, 'statistics')

    # Load data
    print(f"Loading data from: {json_file}")
    data = load_json_data(json_file)

    # Visualize
    print("\nGenerating visualizations...")
    visualize_t_test_results(data, output_dir)

    print("\nAll visualizations completed!")


if __name__ == '__main__':
    main()