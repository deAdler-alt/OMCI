"""
WBAN Optimization - Convergence Curves Visualization
Author: Kamil Piejko
Date: 2024

Wykres krzywych zbieżności GA i PSO.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def plot_convergence_curves(
    csv_path: str = 'results/main_scenarios/all_main_scenarios.csv',
    output_path: str = 'results/plots/convergence_curves.png',
    scenario: str = 'S2',
    weight_variant: str = 'balanced',
    n_runs_to_plot: int = 10,
    show: bool = False
):
    """
    Wykres: Krzywe zbieżności algorytmów.
    
    Args:
        csv_path: Ścieżka do CSV (musi zawierać kolumnę 'convergence_curve')
        output_path: Ścieżka zapisu
        scenario: Który scenariusz pokazać
        weight_variant: Wariant wag
        n_runs_to_plot: Ile runs pokazać (reszta jako cień)
        show: Czy wyświetlić
    """
    logger.info(f"Generating convergence curves plot...")
    
    # Wczytaj dane
    df = pd.read_csv(csv_path)
    
    # Filtruj
    df = df[(df['scenario'] == scenario) & (df['weight_variant'] == weight_variant)]
    
    # Jeśli convergence_curve to string (JSON), parsuj
    if 'convergence_curve' in df.columns and isinstance(df['convergence_curve'].iloc[0], str):
        df['convergence_curve'] = df['convergence_curve'].apply(json.loads)
    
    # Wykres
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    algorithms = ['GA', 'PSO']
    colors = {'GA': '#2E86AB', 'PSO': '#A23B72'}
    
    for idx, algorithm in enumerate(algorithms):
        ax = axes[idx]
        data = df[df['algorithm'] == algorithm]
        
        if 'convergence_curve' not in data.columns or data.empty:
            logger.warning(f"No convergence data for {algorithm}")
            ax.text(0.5, 0.5, 'No convergence data available',
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Weź pierwsze n_runs_to_plot runs
        curves = data['convergence_curve'].head(n_runs_to_plot).tolist()
        
        # Znajdź max długość
        max_len = max(len(c) for c in curves)
        
        # Wyrównaj długości (pad z ostatnią wartością)
        curves_padded = []
        for curve in curves:
            if len(curve) < max_len:
                curve = curve + [curve[-1]] * (max_len - len(curve))
            curves_padded.append(curve)
        
        curves_array = np.array(curves_padded)
        
        # Średnia i std
        mean_curve = curves_array.mean(axis=0)
        std_curve = curves_array.std(axis=0)
        
        iterations = np.arange(len(mean_curve))
        
        # Plot średniej
        ax.plot(iterations, mean_curve, 
               color=colors[algorithm], 
               linewidth=2.5, 
               label=f'{algorithm} (mean)')
        
        # Obszar ±1 std
        ax.fill_between(iterations,
                       mean_curve - std_curve,
                       mean_curve + std_curve,
                       color=colors[algorithm],
                       alpha=0.2,
                       label='±1 std')
        
        # Pojedyncze runs jako cienkie linie
        for curve in curves_padded[:5]:  # Tylko 5 pierwszych dla czytelności
            ax.plot(iterations, curve,
                   color=colors[algorithm],
                   alpha=0.15,
                   linewidth=0.5)
        
        ax.set_xlabel('Iteration / Generation', fontsize=12, fontweight='bold')
        ax.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
        ax.set_title(f'{algorithm} Convergence ({scenario})',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Zaznacz końcową wartość
        final_mean = mean_curve[-1]
        ax.axhline(y=final_mean, color=colors[algorithm],
                  linestyle='--', linewidth=1, alpha=0.5)
        ax.text(len(mean_curve)*0.6, final_mean*1.02,
               f'Final: {final_mean:.6f}',
               fontsize=9, color=colors[algorithm])
    
    plt.suptitle(f'Algorithm Convergence Comparison\n({scenario}, {weight_variant.replace("_", " ").title()})',
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
    print("CONVERGENCE PLOT TEST")
    print("="*60)
    
    # Utwórz przykładowe dane z krzywymi zbieżności
    print("\n[TEST] Creating sample data with convergence curves...")
    
    sample_data = []
    
    for algorithm in ['GA', 'PSO']:
        for run in range(10):
            # Symulowana krzywa zbieżności
            n_iter = 100
            initial_fitness = np.random.uniform(0.1, 0.2)
            
            if algorithm == 'GA':
                # GA: liniowa zbieżność
                curve = initial_fitness * np.exp(-0.03 * np.arange(n_iter))
            else:
                # PSO: szybsza zbieżność na początku
                curve = initial_fitness * np.exp(-0.05 * np.arange(n_iter))
            
            # Dodaj szum
            curve = curve + np.random.normal(0, 0.001, n_iter).cumsum()
            curve = np.maximum.accumulate(curve * -1) * -1  # Monotoniczne maleją
            
            sample_data.append({
                'scenario': 'S2',
                'algorithm': algorithm,
                'weight_variant': 'balanced',
                'convergence_curve': curve.tolist(),
                'best_fitness': curve[-1],
                'run_id': run + 1
            })
    
    df_sample = pd.DataFrame(sample_data)
    
    csv_path = 'results/plots/sample_data_convergence.csv'
    df_sample.to_csv(csv_path, index=False)
    
    print(f"  Sample data saved to {csv_path}")
    
    # Wygeneruj wykres
    print("\n[TEST] Generating plot...")
    plot_convergence_curves(
        csv_path=csv_path,
        output_path='results/plots/convergence_curves_test.png',
        scenario='S2',
        weight_variant='balanced',
        n_runs_to_plot=10,
        show=False
    )
    
    print("\n" + "="*60)
    print("✓ TEST PASSED")
    print("Plot saved to: results/plots/convergence_curves_test.png")
    print("="*60 + "\n")
