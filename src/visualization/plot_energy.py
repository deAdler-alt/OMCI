"""
WBAN Optimization - Energy Consumption Visualization
Author: Kamil Piejko
Date: 2024

Wykres energii vs liczba sensorów dla różnych algorytmów.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Ustaw styl
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def plot_energy_vs_sensors(
    csv_path: str = 'results/main_scenarios/all_main_scenarios.csv',
    output_path: str = 'results/plots/energy_vs_sensors.png',
    weight_variant: str = 'balanced',
    show: bool = False
):
    """
    Wykres: Energia vs liczba sensorów.
    
    Args:
        csv_path: Ścieżka do CSV z wynikami
        output_path: Ścieżka zapisu wykresu
        weight_variant: Który wariant wag pokazać
        show: Czy wyświetlić wykres
    """
    logger.info(f"Generating energy vs sensors plot...")
    
    # Wczytaj dane
    df = pd.read_csv(csv_path)
    
    # Filtruj po wariancie wag
    df = df[df['weight_variant'] == weight_variant]
    
    # Konwertuj energię do mJ
    df['E_total_mJ'] = df['E_total'] * 1000
    
    # Agregacja: średnia i std dla każdej kombinacji (scenario, algorithm)
    agg_df = df.groupby(['scenario', 'algorithm', 'n_sensors']).agg({
        'E_total_mJ': ['mean', 'std']
    }).reset_index()
    
    agg_df.columns = ['scenario', 'algorithm', 'n_sensors', 'mean', 'std']
    
    # Wykres
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Kolory i markery
    colors = {'GA': '#2E86AB', 'PSO': '#A23B72'}
    markers = {'GA': 'o', 'PSO': 's'}
    
    for algorithm in ['GA', 'PSO']:
        data = agg_df[agg_df['algorithm'] == algorithm]
        
        ax.errorbar(
            data['n_sensors'],
            data['mean'],
            yerr=data['std'],
            label=algorithm,
            marker=markers[algorithm],
            color=colors[algorithm],
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=8
        )
    
    ax.set_xlabel('Number of Sensors', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Energy Consumption (mJ)', fontsize=14, fontweight='bold')
    ax.set_title(f'Energy Consumption vs Network Size\n({weight_variant.replace("_", " ").title()})',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Dodaj wartości na wykresie
    for algorithm in ['GA', 'PSO']:
        data = agg_df[agg_df['algorithm'] == algorithm]
        for _, row in data.iterrows():
            ax.annotate(f'{row["mean"]:.2f}',
                       xy=(row['n_sensors'], row['mean']),
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       color=colors[algorithm])
    
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
    print("ENERGY PLOT TEST")
    print("="*60)
    
    # Test z example data
    print("\n[TEST] Creating sample data...")
    
    # Utwórz przykładowe dane
    sample_data = []
    scenarios = ['S1', 'S2', 'S3']
    n_sensors_map = {'S1': 6, 'S2': 15, 'S3': 25}
    
    for scenario in scenarios:
        for algorithm in ['GA', 'PSO']:
            for run in range(10):
                # Symulowane wartości
                n_sensors = n_sensors_map[scenario]
                base_energy = n_sensors * 0.5  # mJ
                noise = np.random.normal(0, 0.1 * base_energy)
                
                if algorithm == 'PSO':
                    base_energy *= 0.95  # PSO trochę lepszy
                
                sample_data.append({
                    'scenario': scenario,
                    'algorithm': algorithm,
                    'weight_variant': 'balanced',
                    'n_sensors': n_sensors,
                    'E_total': (base_energy + noise) / 1000,  # Konwersja do J
                    'run_id': run + 1
                })
    
    df_sample = pd.DataFrame(sample_data)
    
    # Zapisz do CSV
    csv_path = 'results/plots/sample_data.csv'
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(csv_path, index=False)
    
    print(f"  Sample data saved to {csv_path}")
    print(f"  Rows: {len(df_sample)}")
    
    # Wygeneruj wykres
    print("\n[TEST] Generating plot...")
    plot_energy_vs_sensors(
        csv_path=csv_path,
        output_path='results/plots/energy_vs_sensors_test.png',
        weight_variant='balanced',
        show=False
    )
    
    print("\n" + "="*60)
    print("✓ TEST PASSED")
    print("Plot saved to: results/plots/energy_vs_sensors_test.png")
    print("="*60 + "\n")
