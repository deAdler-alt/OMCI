"""
WBAN/IoMT Optimization - Sensor Placement Visualization
Style: IEEE/Scientific Publication
"""

import numpy as np
import ast
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Ustawienie stylu globalnego dla matplotlib (Scientific Style)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

logger = logging.getLogger(__name__)

def plot_sensor_placement(
    genome: np.ndarray,
    fitness_function,
    output_path: str = 'results/plots/placement.png',
    title: str = 'Network Topology',
    show: bool = False
):
    from src.core.genotype import Genotype
    
    logger.info(f"Generating scientific topology plot...")
    
    # Dekoduj
    sensors, hubs = Genotype.decode(
        genome,
        fitness_function.sensor_config,
        fitness_function.body_model,
        fitness_function.energy_init,
        fitness_function.hub_config
    )
    
    # Oblicz połączenia (dla wizualizacji)
    scale_x = fitness_function.config['space']['dimensions'][0]
    scale_y = fitness_function.config['space']['dimensions'][1]
    
    # Przygotuj wykres
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Konwersja na metry
    h_positions = np.array([h.position * [scale_x, scale_y] for h in hubs])
    s_positions = np.array([s.position * [scale_x, scale_y] for s in sensors])
    
    # 1. Rysuj połączenia (Linie)
    # Dla każdego sensora znajdź najbliższego Huba i narysuj linię
    for s_pos in s_positions:
        # Oblicz dystanse do wszystkich hubów
        dists = [np.linalg.norm(s_pos - h_pos) for h_pos in h_positions]
        closest_idx = np.argmin(dists)
        closest_h_pos = h_positions[closest_idx]
        
        # Rysuj linię (szara, cienka)
        ax.plot([s_pos[0], closest_h_pos[0]], [s_pos[1], closest_h_pos[1]], 
                color='#808080', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)

    # 2. Rysuj Sensory (Kółka)
    # Używamy jednego koloru dla sensorów (np. niebieski) lub zależnego od klastra
    # Tu dla elegancji: niebieskie kropki z czarną obwódką
    ax.scatter(s_positions[:, 0], s_positions[:, 1], 
               c='#2E86AB', edgecolor='white', s=60, marker='o', 
               label='Sensor Node', zorder=2)

    # 3. Rysuj Huby (Gwiazdy / Diamenty)
    # Czerwone lub złote
    ax.scatter(h_positions[:, 0], h_positions[:, 1], 
               c='#D90429', edgecolor='black', s=150, marker='D', 
               label='Cluster Head (CH)', zorder=3)
    
    # 4. Rysuj Stację Bazową (BS)
    bs_pos = np.array(fitness_function.config['space']['base_station']['position'])
    ax.scatter(bs_pos[0], bs_pos[1], 
               c='black', s=200, marker='s', 
               label='Base Station', zorder=4)
    
    # Opcjonalnie: Linie od Hubów do BS (przerywane)
    for h_pos in h_positions:
        ax.plot([h_pos[0], bs_pos[0]], [h_pos[1], bs_pos[1]], 
                color='black', linestyle=':', linewidth=1.5, alpha=0.3)

    # Ustawienia osi
    ax.set_xlim(0, scale_x)
    ax.set_ylim(0, scale_y)
    ax.set_xlabel('Area Width [m]')
    ax.set_ylabel('Area Height [m]')
    
    # Tytuł wewnątrz wykresu lub nad
    ax.set_title(title, pad=15, fontweight='bold')
    
    # Legenda (na dole, pozioma)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), 
              ncol=3, frameon=False)
    
    # Metryki w rogu (elegancki box)
    fitness = fitness_function.evaluate(genome)
    metrics_text = f"Total Energy: {fitness:.4f} J"
    
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='square,pad=0.5', fc='white', ec='black', alpha=1.0))

    plt.tight_layout()
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    plt.close()

def plot_best_placements_from_csv(
    csv_path: str = 'results/main_scenarios/all_main_scenarios.csv',
    config_path: str = 'config/wban_params.yaml',
    output_dir: str = 'results/plots/placements',
    scenarios: list = ['S1', 'S2', 'S3'],
    algorithms: list = ['GA', 'PSO'],
    weight_variant: str = 'balanced'
):
    import sys
    # ... (Setup pathów bez zmian) ...
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    from src.utils.config_loader import load_config
    from src.optimization.fitness_function import FitnessFunction
    
    df = pd.read_csv(csv_path)
    df = df[df['weight_variant'] == weight_variant]
    config = load_config(config_path)
    
    def parse_genome(val):
        if not isinstance(val, str): return val
        try: return ast.literal_eval(val)
        except: return []

    if isinstance(df['best_genome'].iloc[0], str):
        df['best_genome'] = df['best_genome'].apply(parse_genome)

    for scenario in scenarios:
        for algorithm in algorithms:
            subset = df[(df['scenario'] == scenario) & (df['algorithm'] == algorithm)]
            if subset.empty: continue
            
            best_row = subset.loc[subset['best_fitness'].idxmin()]
            best_genome = np.array(best_row['best_genome'])
            fitness_func = FitnessFunction(config, scenario, weight_variant)
            
            # Tytuł naukowy
            n_nodes = fitness_func.n_sensors
            n_chs = fitness_func.scenario_config['n_hubs']
            title = f"Optimized Topology ({algorithm})\nNodes: {n_nodes}, Clusters: {n_chs}"
            
            output_path = f'{output_dir}/placement_{scenario}_{algorithm}.png'
            
            plot_sensor_placement(best_genome, fitness_func, output_path, title)

if __name__ == '__main__':
    pass