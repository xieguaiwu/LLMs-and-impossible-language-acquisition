import json
import os
from scipy import stats
import numpy as np


def load_json_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def welch_t_test(data1, data2, name1, name2, metric_name):
    """
    Perform Welch's t-test on two independent samples.
    Welch's t-test is used when the two samples have unequal variances.
    """
    data1 = np.array(data1)
    data2 = np.array(data2)

    # Perform Welch's t-test
    statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)

    # Calculate descriptive statistics
    mean1, std1 = np.mean(data1), np.std(data1, ddof=1)
    mean2, std2 = np.mean(data2), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)

    # Calculate Cohen's d (effect size)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std

    print(f"\n{'='*80}")
    print(f"Welch's T-Test: {name1} vs {name2} ({metric_name})")
    print(f"{'='*80}")
    print(f"{name1}:")
    print(f"  Mean: {mean1:.6f}")
    print(f"  Std:  {std1:.6f}")
    print(f"  N:    {n1}")
    print(f"\n{name2}:")
    print(f"  Mean: {mean2:.6f}")
    print(f"  Std:  {std2:.6f}")
    print(f"  N:    {n2}")
    print(f"\nTest Results:")
    print(f"  t-statistic: {statistic:.6f}")
    print(f"  p-value:     {p_value:.6e}")
    print(f"  Cohen's d:   {cohens_d:.6f}")

    # Interpret results
    alpha = 0.05
    if p_value < alpha:
        significance = "significant"
        print(f"  Result: Significant difference (p < {alpha})")
    else:
        significance = "not_significant"
        print(f"  Result: No significant difference (p >= {alpha})")

    # Effect size interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect = "negligible"
    elif abs_d < 0.5:
        effect = "small"
    elif abs_d < 0.8:
        effect = "medium"
    else:
        effect = "large"
    print(f"  Effect size: {effect}")

    return {
        'name1': name1,
        'name2': name2,
        'metric_name': metric_name,
        'statistic': float(statistic),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significance': significance,
        'effect_size_category': effect,
        'alpha': alpha,
        'group1': {
            'name': name1,
            'mean': float(mean1),
            'std': float(std1),
            'n': int(n1)
        },
        'group2': {
            'name': name2,
            'mean': float(mean2),
            'std': float(std2),
            'n': int(n2)
        }
    }


def main():
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'statistics', 'json_baby')
    output_dir = os.path.join(base_dir, 'statistics')

    # Load all data from json_baby directory
    natural = load_json_data(os.path.join(data_dir, 'training_results_natural.json'))
    parity = load_json_data(os.path.join(data_dir, 'training_results_parity_negation.json'))
    reversed = load_json_data(os.path.join(data_dir, 'training_results_reversed.json'))

    print("\n" + "="*80)
    print("WELCH'S T-TEST ANALYSIS FOR BABY DATASET")
    print("="*80)

    # Collect all test results
    all_results = {
        'test_type': "Welch's t-test",
        'alpha_level': 0.05,
        'description': 'Welch\'s t-test for two independent samples with unequal variances on baby dataset',
        'tests': []
    }

    # Perform t-tests on Loss data
    print("\n\n" + "="*80)
    print("LOSS DATA ANALYSIS")
    print("="*80)

    result = welch_t_test(
        natural['losses'],
        parity['losses'],
        'Natural Language',
        'Impossible (Parity Negation)',
        'Loss'
    )
    all_results['tests'].append(result)

    result = welch_t_test(
        natural['losses'],
        reversed['losses'],
        'Natural Language',
        'Impossible (Reversed)',
        'Loss'
    )
    all_results['tests'].append(result)

    result = welch_t_test(
        parity['losses'],
        reversed['losses'],
        'Impossible (Parity Negation)',
        'Impossible (Reversed)',
        'Loss'
    )
    all_results['tests'].append(result)

    # Perform t-tests on Perplexity data
    print("\n\n" + "="*80)
    print("PERPLEXITY DATA ANALYSIS")
    print("="*80)

    result = welch_t_test(
        natural['perplexities'],
        parity['perplexities'],
        'Natural Language',
        'Impossible (Parity Negation)',
        'Perplexity'
    )
    all_results['tests'].append(result)

    result = welch_t_test(
        natural['perplexities'],
        reversed['perplexities'],
        'Natural Language',
        'Impossible (Reversed)',
        'Perplexity'
    )
    all_results['tests'].append(result)

    result = welch_t_test(
        parity['perplexities'],
        reversed['perplexities'],
        'Impossible (Parity Negation)',
        'Impossible (Reversed)',
        'Perplexity'
    )
    all_results['tests'].append(result)

    # Save results to JSON file
    output_file = os.path.join(output_dir, 'welch_t_test_baby_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()