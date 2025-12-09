"""
WBAN Optimization - Network Lifetime Visualization
Author: Kamil Piejko
Date: 2024

Wykres czasu życia sieci (FND - First Node Dies).
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


def plot_lifetime_comparison(
    csv_path: str = 'results/main_scenarios/all_main_scenarios.csv',
    output_path: str = 'results/plots/lifetime_comparison.png',
    show: bool = False
):
    """
    Wykres: Porównanie czasu życia sieci (box plot).
    
    Args:
        csv_path: Ścieżka do CSV
        output_path: Ścieżka zapisu
        show: Czy wyświetlić
    """
    logger.info(f"Generating network lifetime plot...")
    
    # Wczytaj dane
    df = pd.read_csv(csv_path)
    
    # Filtruj tylko balanced weights
    df = df[df['weight_variant'] == 'balanced']
    
    # Wykres
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    scenarios = ['S1', 'S2', 'S3']
    colors = {'GA': '#2E86AB', 'PSO': '#A23B72'}
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx]
        data = df[df['scenario'] == scenario]
        
        # Box plot
        sns.boxplot(
            data=data,
            x='algorithm',
            y='network_lifetime',
            palette=colors,
            ax=ax
        )
        
        # Dodaj scatter points
        sns.stripplot(
            data=data,
            x='algorithm',
            y='network_lifetime',
            color='black',
            alpha=0.3,
            size=3,
            ax=ax
        )
        
        ax.set_title(f'{scenario} ({data["n_sensors"].iloc[0]} sensors)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Algorithm', fontsize=12)
        
        if idx == 0:
            ax.set_ylabel('Network Lifetime (rounds)', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel('')
        
        ax.grid(True, alpha=0.3, axis='y')
        
        # Dodaj średnie wartości
        for i, algorithm in enumerate(['GA', 'PSO']):
            mean_val = data[data['algorithm'] == algorithm]['network_lifetime'].mean()
            ax.text(i, mean_val, f'μ={mean_val:.0f}',
                   ha='center', va='bottom', fontsize=10,
                   fontweight='bold', color=colors[algorithm])
    
    plt.suptitle('Network Lifetime Comparison (FND - First Node Dies)',
                fontsize=16, fontweight='bold', y=1.02)
    
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
    print("LIFETIME PLOT TEST")
    print("="*60)
    
    # Utwórz przykładowe dane
    print("\n[TEST] Creating sample data...")
    
    sample_data = []
    scenarios = ['S1', 'S2', 'S3']
    n_sensors_map = {'S1': 6, 'S2': 15, 'S3': 25}
    
    for scenario in scenarios:
        n_sensors = n_sensors_map[scenario]
        base_lifetime_ga = 1500 / np.sqrt(n_sensors)
        base_lifetime_pso = base_lifetime_ga * 1.1
        
        for algorithm, base_lt in [('GA', base_lifetime_ga), ('PSO', base_lifetime_pso)]:
            for run in range(50):
                noise = np.random.normal(0, base_lt * 0.1)
                lifetime = int(max(100, base_lt + noise))
                
                sample_data.append({
                    'scenario': scenario,
                    'algorithm': algorithm,
                    'weight_variant': 'balanced',
                    'n_sensors': n_sensors,
                    'network_lifetime': lifetime,
                    'run_id': run + 1
                })
    
    df_sample = pd.DataFrame(sample_data)
    
    csv_path = 'results/plots/sample_data_lifetime.csv'
    df_sample.to_csv(csv_path, index=False)
    
    print(f"  Sample data saved to {csv_path}")
    
    # Wygeneruj wykres
    print("\n[TEST] Generating plot...")
    plot_lifetime_comparison(
        csv_path=csv_path,
        output_path='results/plots/lifetime_comparison_test.png',
        show=False
    )
    
    print("\n" + "="*60)
    print("✓ TEST PASSED")
    print("Plot saved to: results/plots/lifetime_comparison_test.png")
    print("="*60 + "\n")
