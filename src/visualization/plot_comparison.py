"""
WBAN Optimization - Algorithm Comparison Visualization
Author: Kamil Piejko
Date: 2024

Kompleksowe porównanie algorytmów: statystyki i radar plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats
import logging

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)


def plot_algorithm_comparison_table(
    csv_path: str = 'results/main_scenarios/all_main_scenarios.csv',
    output_path: str = 'results/plots/comparison_table.png',
    scenario: str = 'S2',
    weight_variant: str = 'balanced',
    show: bool = False
):
    """
    Tabela porównawcza algorytmów ze statystykami.
    
    Args:
        csv_path: Ścieżka do CSV
        output_path: Ścieżka zapisu
        scenario: Scenariusz do analizy
        weight_variant: Wariant wag
        show: Czy wyświetlić
    """
    logger.info(f"Generating comparison table...")
    
    # Wczytaj dane
    df = pd.read_csv(csv_path)
    
    # Filtruj
    df = df[(df['scenario'] == scenario) & (df['weight_variant'] == weight_variant)]
    df['E_total_mJ'] = df['E_total'] * 1000
    
    # Statystyki dla każdego algorytmu
    algorithms = ['GA', 'PSO']
    metrics = ['best_fitness', 'E_total_mJ', 'network_lifetime', 'min_link_margin']
    metric_names = ['Fitness', 'Energy (mJ)', 'Lifetime', 'Link Margin (dB)']
    
    stats_data = []
    
    for algorithm in algorithms:
        alg_data = df[df['algorithm'] == algorithm]
        
        for metric, name in zip(metrics, metric_names):
            values = alg_data[metric].values
            
            stats_data.append({
                'Algorithm': algorithm,
                'Metric': name,
                'Mean': values.mean(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max(),
                'Median': np.median(values)
            })
    
    df_stats = pd.DataFrame(stats_data)
    
    # Testy statystyczne (t-test)
    pvalues = []
    for metric in metrics:
        ga_values = df[df['algorithm'] == 'GA'][metric].values
        pso_values = df[df['algorithm'] == 'PSO'][metric].values
        
        t_stat, p_value = stats.ttest_ind(ga_values, pso_values)
        pvalues.append(p_value)
    
    # Wykres - tabela
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Tytuł
    title_text = f'Algorithm Comparison: {scenario}, {weight_variant.replace("_", " ").title()}\n'
    title_text += f'(n = {len(df)//2} runs per algorithm)\n\n'
    ax.text(0.5, 0.95, title_text, ha='center', va='top', 
           fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    # Tabela statystyk
    table_data = []
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ga_stats = df_stats[(df_stats['Algorithm'] == 'GA') & (df_stats['Metric'] == name)].iloc[0]
        pso_stats = df_stats[(df_stats['Algorithm'] == 'PSO') & (df_stats['Metric'] == name)].iloc[0]
        
        # Winner (niższa wartość lepsza dla fitness, energy; wyższa dla lifetime, margin)
        if metric in ['best_fitness', 'E_total_mJ']:
            winner = 'GA' if ga_stats['Mean'] < pso_stats['Mean'] else 'PSO'
        else:
            winner = 'GA' if ga_stats['Mean'] > pso_stats['Mean'] else 'PSO'
        
        # Significance
        sig = '***' if pvalues[i] < 0.001 else '**' if pvalues[i] < 0.01 else '*' if pvalues[i] < 0.05 else 'ns'
        
        table_data.append([
            name,
            f"{ga_stats['Mean']:.4f} ± {ga_stats['Std']:.4f}",
            f"{pso_stats['Mean']:.4f} ± {pso_stats['Std']:.4f}",
            winner,
            f"{pvalues[i]:.4f} {sig}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'GA (mean ± std)', 'PSO (mean ± std)', 'Winner', 'p-value'],
        cellLoc='center',
        loc='center',
        bbox=[0.1, 0.3, 0.8, 0.5]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Formatowanie
    for i in range(len(metrics) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#2E86AB')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 3:  # Winner column
                    winner_text = table_data[i-1][3]
                    if winner_text == 'GA':
                        cell.set_facecolor('#E3F2FD')
                    elif winner_text == 'PSO':
                        cell.set_facecolor('#FCE4EC')
    
    # Legenda znaczności
    legend_text = "Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant"
    ax.text(0.5, 0.15, legend_text, ha='center', va='top',
           fontsize=9, style='italic', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Zapisz
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_radar_comparison(
    csv_path: str = 'results/main_scenarios/all_main_scenarios.csv',
    output_path: str = 'results/plots/comparison_radar.png',
    scenario: str = 'S2',
    weight_variant: str = 'balanced',
    show: bool = False
):
    """
    Radar plot porównujący algorytmy.
    
    Args:
        csv_path: Ścieżka do CSV
        output_path: Ścieżka zapisu
        scenario: Scenariusz
        weight_variant: Wariant wag
        show: Czy wyświetlić
    """
    logger.info(f"Generating radar comparison...")
    
    # Wczytaj dane
    df = pd.read_csv(csv_path)
    df = df[(df['scenario'] == scenario) & (df['weight_variant'] == weight_variant)]
    df['E_total_mJ'] = df['E_total'] * 1000
    
    # Metryki (znormalizowane 0-1, wyższe = lepsze)
    algorithms = ['GA', 'PSO']
    categories = ['Energy\nEfficiency', 'Network\nLifetime', 'Link\nReliability', 'Convergence\nSpeed']
    
    values = {}
    
    for algorithm in algorithms:
        alg_data = df[df['algorithm'] == algorithm]
        
        # Energy efficiency (odwrotność energii, znormalizowana)
        energy_eff = 1.0 / alg_data['E_total_mJ'].mean()
        
        # Lifetime (znormalizowany)
        lifetime = alg_data['network_lifetime'].mean()
        
        # Reliability (link margin, przeskalowany)
        margin = alg_data['min_link_margin'].mean()
        reliability = max(0, (margin + 10) / 20)  # Zakładamy zakres [-10, 10]
        
        # Convergence (odwrotność fitness - niższy fitness = lepsza zbieżność)
        convergence = 1.0 / alg_data['best_fitness'].mean()
        
        # Normalizacja do zakresu [0, 1]
        all_energy = [1.0 / df[df['algorithm'] == a]['E_total_mJ'].mean() for a in algorithms]
        all_lifetime = [df[df['algorithm'] == a]['network_lifetime'].mean() for a in algorithms]
        all_convergence = [1.0 / df[df['algorithm'] == a]['best_fitness'].mean() for a in algorithms]
        
        energy_norm = (energy_eff - min(all_energy)) / (max(all_energy) - min(all_energy) + 1e-9)
        lifetime_norm = (lifetime - min(all_lifetime)) / (max(all_lifetime) - min(all_lifetime) + 1e-9)
        convergence_norm = (convergence - min(all_convergence)) / (max(all_convergence) - min(all_convergence) + 1e-9)
        
        values[algorithm] = [energy_norm, lifetime_norm, reliability, convergence_norm]
    
    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Zamknij polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = {'GA': '#2E86AB', 'PSO': '#A23B72'}
    
    for algorithm in algorithms:
        vals = values[algorithm] + values[algorithm][:1]  # Zamknij
        ax.plot(angles, vals, 'o-', linewidth=2.5, label=algorithm, color=colors[algorithm])
        ax.fill(angles, vals, alpha=0.15, color=colors[algorithm])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    ax.set_title(f'Multi-Metric Algorithm Comparison\n({scenario}, {weight_variant.replace("_", " ").title()})',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    
    plt.tight_layout()
    
    # Zapisz
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    
    if show:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("COMPARISON PLOTS TEST")
    print("="*60)
    
    # Przykładowe dane
    print("\n[TEST] Creating sample comparison data...")
    
    sample_data = []
    for algorithm in ['GA', 'PSO']:
        for run in range(50):
            if algorithm == 'GA':
                fitness = np.random.normal(0.015, 0.002)
                energy = np.random.normal(8.5, 0.5)
                lifetime = int(np.random.normal(1200, 100))
                margin = np.random.normal(2.5, 1.0)
            else:  # PSO
                fitness = np.random.normal(0.013, 0.002)
                energy = np.random.normal(8.0, 0.5)
                lifetime = int(np.random.normal(1350, 100))
                margin = np.random.normal(3.0, 1.0)
            
            sample_data.append({
                'scenario': 'S2',
                'algorithm': algorithm,
                'weight_variant': 'balanced',
                'best_fitness': fitness,
                'E_total': energy / 1000,
                'network_lifetime': lifetime,
                'min_link_margin': margin,
                'run_id': run + 1
            })
    
    df_sample = pd.DataFrame(sample_data)
    csv_path = 'results/plots/sample_comparison_data.csv'
    df_sample.to_csv(csv_path, index=False)
    
    print(f"  Sample data saved to {csv_path}")
    
    # Test 1: Table
    print("\n[TEST 1] Generating comparison table...")
    plot_algorithm_comparison_table(
        csv_path=csv_path,
        output_path='results/plots/comparison_table_test.png',
        scenario='S2',
        show=False
    )
    
    # Test 2: Radar
    print("\n[TEST 2] Generating radar plot...")
    plot_radar_comparison(
        csv_path=csv_path,
        output_path='results/plots/comparison_radar_test.png',
        scenario='S2',
        show=False
    )
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("Plots saved to:")
    print("  - results/plots/comparison_table_test.png")
    print("  - results/plots/comparison_radar_test.png")
    print("="*60 + "\n")
