"""
WBAN Optimization - Sensor Placement Visualization
Author: Kamil Piejko
Date: 2024

Wizualizacja rozmieszczenia sensorów na ciele z połączeniami LOS/NLOS.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


def plot_sensor_placement(
    genome: np.ndarray,
    fitness_function,
    output_path: str = 'results/plots/placement.png',
    title: str = 'Sensor Placement',
    show: bool = False
):
    """
    Wykres rozmieszczenia sensorów na body model.
    
    Args:
        genome: Genotyp do wizualizacji
        fitness_function: FitnessFunction object (dla decode i body_model)
        output_path: Ścieżka zapisu
        title: Tytuł wykresu
        show: Czy wyświetlić
    """
    from src.core.genotype import Genotype
    
    logger.info(f"Generating sensor placement plot...")
    
    # Dekoduj genotyp
    sensors, hub = Genotype.decode(
        genome,
        fitness_function.n_sensors,
        fitness_function.sensor_config,
        fitness_function.body_model,
        fitness_function.hub_config
    )
    
    # Ewaluacja szczegółowa (do informacji o LOS/NLOS)
    details = fitness_function.evaluate_detailed(genome)
    
    # Wykres
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Tło - body zones jako prostokąty
    body_model = fitness_function.body_model
    
    for zone_name in body_model.zones.keys():
        zone = body_model.get_zone(zone_name)
        
        # Prostokąt strefy
        rect = patches.Rectangle(
            (zone.x_range[0], zone.y_range[0]),
            zone.x_range[1] - zone.x_range[0],
            zone.y_range[1] - zone.y_range[0],
            linewidth=1,
            edgecolor='gray',
            facecolor='lightgray',
            alpha=0.2
        )
        ax.add_patch(rect)
        
        # Etykieta strefy
        center = zone.get_center()
        ax.text(center[0], center[1], zone_name,
               ha='center', va='center',
               fontsize=8, color='gray', alpha=0.5)
    
    # Torso cylinder (jeśli dostępny)
    if hasattr(body_model, 'torso_cylinder'):
        cyl = body_model.torso_cylinder
        circle = patches.Circle(
            (cyl['center'][0], cyl['center'][1]),
            cyl['radius'],
            linewidth=2,
            edgecolor='blue',
            facecolor='none',
            linestyle='--',
            alpha=0.3,
            label='Torso cylinder'
        )
        ax.add_patch(circle)
    
    # Połączenia sensor → hub
    for i, sensor in enumerate(sensors):
        sensor_result = details['sensor_results'][i]
        los_status = sensor_result['los_status']
        
        # Kolor i styl linii zależnie od LOS/NLOS
        if los_status == 'LOS':
            color = 'green'
            linestyle = '-'
            alpha = 0.4
        else:
            color = 'red'
            linestyle = '--'
            alpha = 0.3
        
        ax.plot([sensor.position[0], hub.position[0]],
               [sensor.position[1], hub.position[1]],
               color=color,
               linestyle=linestyle,
               linewidth=1,
               alpha=alpha)
    
    # Sensory jako punkty
    sensor_colors = plt.cm.tab10(np.linspace(0, 1, len(sensors)))
    
    for i, sensor in enumerate(sensors):
        ax.scatter(sensor.position[0], sensor.position[1],
                  s=200,
                  color=sensor_colors[i],
                  edgecolor='black',
                  linewidth=2,
                  zorder=10,
                  label=f'{sensor.type} (S{i+1})')
        
        # Etykieta sensora
        ax.text(sensor.position[0], sensor.position[1] - 0.03,
               f'S{i+1}',
               ha='center', va='top',
               fontsize=9, fontweight='bold')
    
    # Hub jako gwiazda
    ax.scatter(hub.position[0], hub.position[1],
              s=400,
              color='gold',
              marker='*',
              edgecolor='black',
              linewidth=2,
              zorder=11,
              label='Hub')
    
    ax.text(hub.position[0], hub.position[1] - 0.03,
           'HUB',
           ha='center', va='top',
           fontsize=10, fontweight='bold')
    
    # Ustawienia osi
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('X coordinate (normalized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y coordinate (normalized)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Legenda
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
    
    # Informacje o metrykach
    info_text = f"Fitness: {details['best_fitness']:.6f}\n"
    info_text += f"E_total: {details['E_total']*1000:.4f} mJ\n"
    info_text += f"Lifetime: {details['network_lifetime']} rounds\n"
    info_text += f"LOS/NLOS: {details['los_count']}/{details['nlos_count']}\n"
    info_text += f"Min margin: {details['min_link_margin']:.2f} dB"
    
    ax.text(0.02, 0.98, info_text,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Zapisz
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_file}")
    
    if show:
        plt.show()
    
    plt.close()


def plot_best_placements_from_csv(
    csv_path: str = 'results/main_scenarios/all_main_scenarios.csv',
    config_path: str = 'config/wban_params.yaml',
    output_dir: str = 'results/plots/placements',
    scenarios: list = ['S1', 'S2', 'S3'],
    algorithms: list = ['GA', 'PSO'],
    weight_variant: str = 'balanced'
):
    """
    Generuje wykresy placement dla najlepszych rozwiązań z CSV.
    
    Args:
        csv_path: Ścieżka do CSV z wynikami
        config_path: Ścieżka do config
        output_dir: Katalog wynikowy
        scenarios: Lista scenariuszy
        algorithms: Lista algorytmów
        weight_variant: Wariant wag
    """
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.utils.config_loader import load_config
    from src.optimization.fitness_function import FitnessFunction
    
    logger.info(f"Generating placement plots from CSV...")
    
    # Wczytaj dane
    df = pd.read_csv(csv_path)
    
    # Filtruj po wariancie wag
    df = df[df['weight_variant'] == weight_variant]
    
    # Wczytaj config
    config = load_config(config_path)
    
    # Parsuj best_genome jeśli string
    if isinstance(df['best_genome'].iloc[0], str):
        df['best_genome'] = df['best_genome'].apply(json.loads)
    
    for scenario in scenarios:
        for algorithm in algorithms:
            # Znajdź najlepszy run
            subset = df[(df['scenario'] == scenario) & (df['algorithm'] == algorithm)]
            
            if subset.empty:
                logger.warning(f"No data for {scenario}, {algorithm}")
                continue
            
            best_row = subset.loc[subset['best_fitness'].idxmin()]
            best_genome = np.array(best_row['best_genome'])
            
            # Utwórz fitness function
            fitness_func = FitnessFunction(config, scenario, weight_variant)
            
            # Wygeneruj wykres
            title = f'Best Sensor Placement: {scenario}, {algorithm}\n({weight_variant.replace("_", " ").title()})'
            output_path = f'{output_dir}/placement_{scenario}_{algorithm}_{weight_variant}.png'
            
            plot_sensor_placement(
                genome=best_genome,
                fitness_function=fitness_func,
                output_path=output_path,
                title=title,
                show=False
            )
            
            logger.info(f"Generated {output_path}")


if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.utils.config_loader import load_config
    from src.optimization.fitness_function import FitnessFunction
    from src.core.genotype import Genotype
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("PLACEMENT PLOT TEST")
    print("="*60)
    
    # Wczytaj config
    config = load_config()
    
    # Test dla S1
    print("\n[TEST] Generating placement plot for S1...")
    
    fitness_func = FitnessFunction(config, 'S1', 'balanced')
    
    # Wygeneruj losowe rozwiązanie
    rng = np.random.default_rng(42)
    genome = Genotype.generate_random(
        fitness_func.n_sensors,
        fitness_func.sensor_config,
        fitness_func.body_model,
        fitness_func.hub_config,
        rng
    )
    
    plot_sensor_placement(
        genome=genome,
        fitness_function=fitness_func,
        output_path='results/plots/placement_test.png',
        title='Test Sensor Placement (S1, Random)',
        show=False
    )
    
    print("\n" + "="*60)
    print("✓ TEST PASSED")
    print("Plot saved to: results/plots/placement_test.png")
    print("="*60 + "\n")
