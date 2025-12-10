"""
WBAN Optimization - Convergence Plot
Style: IEEE/Scientific Publication
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Styl globalny
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.grid': True,
    'grid.linestyle': '--',
    'lines.linewidth': 2
})

logger = logging.getLogger(__name__)

def plot_convergence_curves(
    csv_path: str = 'results/main_scenarios/all_main_scenarios.csv',
    output_path: str = 'results/plots/convergence_curves.png',
    scenario: str = 'S2',
    weight_variant: str = 'balanced',
    n_runs_to_plot: int = 10,
    show: bool = False
):
    logger.info(f"Generating scientific convergence plot...")
    
    df = pd.read_csv(csv_path)
    df = df[(df['scenario'] == scenario) & (df['weight_variant'] == weight_variant)]
    
    # Parser
    import numpy as np # Ensure numpy is available for eval
    if 'convergence_curve' in df.columns:
        df['convergence_curve'] = df['convergence_curve'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    algorithms = ['GA', 'PSO']
    # Style linii dla publikacji (różne markery)
    styles = {
        'GA': {'color': '#1f77b4', 'marker': '^', 'label': 'Genetic Algorithm'},
        'PSO': {'color': '#d62728', 'marker': 'o', 'label': 'Particle Swarm Opt.'}
    }
    
    for algorithm in algorithms:
        data = df[df['algorithm'] == algorithm]
        if data.empty: continue
        
        curves = data['convergence_curve'].tolist()
        
        # Wyrównaj długości
        max_len = max(len(c) for c in curves)
        curves_padded = [c + [c[-1]]*(max_len-len(c)) for c in curves]
        curves_array = np.array(curves_padded)
        
        # Średnia
        mean_curve = curves_array.mean(axis=0)
        iterations = np.arange(len(mean_curve))
        
        # Rysuj średnią (z markerami co 10 punktów dla czytelności)
        ax.plot(iterations, mean_curve, 
               color=styles[algorithm]['color'],
               label=styles[algorithm]['label'],
               marker=styles[algorithm]['marker'],
               markevery=max(1, len(iterations)//10), # Markery nie za gęsto
               markersize=7,
               linewidth=2)
        
        # Opcjonalnie: Log scale jeśli różnice są ogromne
        # ax.set_yscale('log')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Energy Consumption [J]')
    ax.set_title(f'Convergence Analysis ({scenario})', fontweight='bold')
    
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    plt.close()

if __name__ == '__main__':
    pass