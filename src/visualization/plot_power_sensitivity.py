"""
WBAN Optimization - Power Sensitivity Analysis
Author: Kamil Piejko
Date: 2024

Analiza wpływu mocy transmisji na metryki sieci.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def plot_power_sensitivity(
    csv_path: str = 'results/power_analysis/all_power_analysis.csv',
    output_path: str = 'results/plots/power_sensitivity.png',
    show: bool = False
):
    """
    Wykres: Wrażliwość metryk na moc transmisji.
    
    Args:
        csv_path: Ścieżka do CSV z power analysis
        output_path: Ścieżka zapisu
        show: Czy wyświetlić
    """
    logger.info(f"Generating power sensitivity plot...")
    
    # Wczytaj dane
    df = pd.read_csv(csv_path)
    
    # Konwertuj energię do mJ
    df['E_total_mJ'] = df['E_total'] * 1000
    
    # Agregacja
    agg_df = df.groupby('P_TX_max').agg({
        'best_fitness': ['mean', 'std'],
        'E_total_mJ': ['mean', 'std'],
        'network_lifetime': ['mean', 'std'],
        'min_link_margin': ['mean', 'std']
    }).reset_index()
    
    # Spłaszcz nazwy kolumn
    agg_df.columns = ['P_TX_max', 
                     'fitness_mean', 'fitness_std',
                     'energy_mean', 'energy_std',
                     'lifetime_mean', 'lifetime_std',
                     'margin_mean', 'margin_std']
    
    # Wykres 2×2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = [
        ('fitness_mean', 'fitness_std', 'Best Fitness', axes[0, 0]),
        ('energy_mean', 'energy_std', 'Energy Consumption (mJ)', axes[0, 1]),
        ('lifetime_mean', 'lifetime_std', 'Network Lifetime (rounds)', axes[1, 0]),
        ('margin_mean', 'margin_std', 'Min Link Margin (dB)', axes[1, 1])
    ]
    
    for mean_col, std_col, ylabel, ax in metrics:
        # Line plot z errorbars
        ax.errorbar(
            agg_df['P_TX_max'],
            agg_df[mean_col],
            yerr=agg_df[std_col],
            marker='o',
            color='#2E86AB',
            capsize=5,
            capthick=2,
            linewidth=2.5,
            markersize=10
        )
        
        ax.set_xlabel('P_TX_max (dBm)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Dodaj wartości
        for _, row in agg_df.iterrows():
            ax.annotate(f'{row[mean_col]:.2f}',
                       xy=(row['P_TX_max'], row[mean_col]),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9)
        
        # Zaznacz rekomendowaną wartość (jeśli margin > 0)
        if mean_col == 'margin_mean':
            # Najniższa moc z dodatnim marginesem
            positive_margin = agg_df[agg_df[mean_col] > 0]
            if not positive_margin.empty:
                optimal_power = positive_margin['P_TX_max'].min()
                ax.axvline(x=optimal_power, color='green', 
                          linestyle='--', linewidth=2, alpha=0.5,
                          label=f'Min power (margin>0): {optimal_power} dBm')
                ax.legend(fontsize=9)
    
    plt.suptitle('Power Sensitivity Analysis (S2, PSO, Balanced)',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Zapisz
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_power_correlation(
    csv_path: str = 'results/power_analysis/all_power_analysis.csv',
    output_path: str = 'results/plots/power_correlation.png',
    show: bool = False
):
    """
    Wykres: Korelacje P_TX vs metryki.
    
    Args:
        csv_path: Ścieżka do CSV
        output_path: Ścieżka zapisu
        show: Czy wyświetlić
    """
    logger.info(f"Generating power correlation plot...")
    
    # Wczytaj dane
    df = pd.read_csv(csv_path)
    df['E_total_mJ'] = df['E_total'] * 1000
    
    # Korelacje
    metrics = ['best_fitness', 'E_total_mJ', 'network_lifetime', 'min_link_margin']
    correlations = df[['P_TX_max'] + metrics].corr()['P_TX_max'][1:]
    
    # Wykres barowy
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#A23B72' if c < 0 else '#2E86AB' for c in correlations.values]
    
    bars = ax.bar(range(len(correlations)), correlations.values, color=colors, alpha=0.8)
    
    ax.set_xticks(range(len(correlations)))
    ax.set_xticklabels(['Fitness', 'Energy (mJ)', 'Lifetime', 'Link Margin'],
                       fontsize=11)
    ax.set_ylabel('Correlation with P_TX_max', fontsize=12, fontweight='bold')
    ax.set_title('Correlation: Transmission Power vs Network Metrics',
                fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Dodaj wartości na barach
    for i, (bar, corr) in enumerate(zip(bars, correlations.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{corr:.3f}',
               ha='center', va='bottom' if height > 0 else 'top',
               fontsize=10, fontweight='bold')
    
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
    print("POWER SENSITIVITY PLOT TEST")
    print("="*60)
    
    # Utwórz przykładowe dane
    print("\n[TEST] Creating sample power analysis data...")
    
    power_levels = [-10, -5, 0, 3, 5]
    sample_data = []
    
    for P_TX in power_levels:
        for run in range(50):
            # Symulowane zależności
            # Energia rośnie z mocą
            energy_base = 5.0 + 0.5 * P_TX
            energy = max(1.0, energy_base + np.random.normal(0, 0.5))
            
            # Lifetime maleje z mocą (więcej energii = krótsza żywotność)
            lifetime_base = 1500 - 20 * P_TX
            lifetime = int(max(100, lifetime_base + np.random.normal(0, 50)))
            
            # Margines rośnie z mocą
            margin_base = -15 + 4 * P_TX
            margin = margin_base + np.random.normal(0, 1)
            
            # Fitness (kombinacja energii i marginesu)
            fitness = 0.7 * (energy / 10.0) + 0.3 * max(0, -margin / 10.0)
            
            sample_data.append({
                'P_TX_max': P_TX,
                'best_fitness': fitness,
                'E_total': energy / 1000,  # Konwersja do J
                'network_lifetime': lifetime,
                'min_link_margin': margin,
                'run_id': run + 1
            })
    
    df_sample = pd.DataFrame(sample_data)
    
    csv_path = 'results/plots/sample_power_data.csv'
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(csv_path, index=False)
    
    print(f"  Sample data saved to {csv_path}")
    
    # Test 1: Sensitivity plot
    print("\n[TEST 1] Generating power sensitivity plot...")
    plot_power_sensitivity(
        csv_path=csv_path,
        output_path='results/plots/power_sensitivity_test.png',
        show=False
    )
    
    # Test 2: Correlation plot
    print("\n[TEST 2] Generating power correlation plot...")
    plot_power_correlation(
        csv_path=csv_path,
        output_path='results/plots/power_correlation_test.png',
        show=False
    )
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("Plots saved to:")
    print("  - results/plots/power_sensitivity_test.png")
    print("  - results/plots/power_correlation_test.png")
    print("="*60 + "\n")
