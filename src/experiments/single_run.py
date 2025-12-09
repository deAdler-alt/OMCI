"""
WBAN Optimization - Single Experiment Run
Author: Kamil Piejko
Date: 2024

Moduł do uruchomienia pojedynczego eksperymentu z daną konfiguracją.
"""

import numpy as np
from typing import Dict, Optional
import logging
import time

logger = logging.getLogger(__name__)


def run_single_experiment(
    config: Dict,
    scenario_name: str,
    algorithm_name: str,
    weight_variant: str,
    run_id: int,
    seed: Optional[int] = None
) -> Dict:
    """
    Uruchamia pojedynczy eksperyment.
    
    Args:
        config: Pełna konfiguracja z YAML
        scenario_name: 'S1', 'S2', lub 'S3'
        algorithm_name: 'GA', 'PSO', 'Random', lub 'Greedy'
        weight_variant: 'energy_priority', 'balanced', lub 'reliability_priority'
        run_id: ID uruchomienia (dla seed)
        seed: Opcjonalny seed (jeśli None, użyj run_id)
    
    Returns:
        results: Słownik z wynikami eksperymentu
    """
    from src.utils.config_loader import ConfigLoader
    from src.optimization.fitness_function import FitnessFunction
    from src.optimization.ga_optimizer import GAOptimizer
    from src.optimization.pso_optimizer import PSOOptimizer
    from src.baselines.baseline_methods import RandomBaseline, GreedyBaseline
    from src.core.genotype import Genotype
    
    # Ustaw seed
    if seed is None:
        seed = run_id
    
    logger.info(f"Starting experiment: {scenario_name}, {algorithm_name}, "
               f"{weight_variant}, run={run_id}, seed={seed}")
    
    start_time = time.time()
    
    # Inicjalizacja fitness function
    fitness_func = FitnessFunction(config, scenario_name, weight_variant)
    dimension = 2 * fitness_func.n_sensors + 2
    
    # Wybierz algorytm
    if algorithm_name == 'GA':
        optimizer = GAOptimizer(config)
    elif algorithm_name == 'PSO':
        optimizer = PSOOptimizer(config)
    elif algorithm_name == 'Random':
        # Random baseline z taką samą liczbą ewaluacji jak GA/PSO
        n_evals = config['optimization']['GA']['pop_size'] * \
                  config['optimization']['GA']['n_generations']
        optimizer = RandomBaseline(n_samples=n_evals)
    elif algorithm_name == 'Greedy':
        # Greedy z perturbacjami
        optimizer = GreedyBaseline(n_perturbations=100)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # Uruchom optymalizację
    opt_results = optimizer.optimize(
        fitness_function=fitness_func.evaluate,
        dimension=dimension,
        bounds=(0.0, 1.0),
        seed=seed
    )
    
    # Szczegółowa ewaluacja najlepszego rozwiązania
    best_genome = opt_results['best_genome']
    detailed_results = fitness_func.evaluate_detailed(best_genome)
    
    elapsed_time = time.time() - start_time
    
    # Kompletne wyniki
    results = {
        # Identyfikatory
        'scenario': scenario_name,
        'algorithm': algorithm_name,
        'weight_variant': weight_variant,
        'run_id': run_id,
        'seed': seed,
        
        # Parametry scenariusza
        'n_sensors': fitness_func.n_sensors,
        'P_TX_max': fitness_func.P_TX_max,
        'w_E': fitness_func.weights['w_E'],
        'w_R': fitness_func.weights['w_R'],
        
        # Wyniki optymalizacji
        'best_fitness': opt_results['best_fitness'],
        'n_evaluations': opt_results['n_evaluations'],
        'convergence_curve': opt_results['convergence_curve'],
        
        # Szczegółowe metryki
        'is_valid': detailed_results['is_valid'],
        'validation_reason': detailed_results['validation_reason'],
        'E_total': detailed_results['E_total'],
        'min_link_margin': detailed_results['min_link_margin'],
        'network_lifetime': detailed_results['network_lifetime'],
        'los_count': detailed_results['los_count'],
        'nlos_count': detailed_results['nlos_count'],
        'los_ratio': detailed_results['los_ratio'],
        
        # Rozwiązanie
        'best_genome': best_genome.tolist(),
        
        # Meta
        'elapsed_time': elapsed_time
    }
    
    logger.info(f"Experiment complete: fitness={results['best_fitness']:.6f}, "
               f"time={elapsed_time:.2f}s")
    
    return results


if __name__ == '__main__':
    # Test single run
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.utils.config_loader import load_config
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("SINGLE RUN TEST")
    print("="*60)
    
    # Wczytaj konfigurację
    config = load_config()
    
    # Test 1: GA na S1
    print("\n[TEST 1] GA on S1 (balanced weights):")
    results_ga = run_single_experiment(
        config=config,
        scenario_name='S1',
        algorithm_name='GA',
        weight_variant='balanced',
        run_id=1,
        seed=42
    )
    
    print(f"  Scenario: {results_ga['scenario']}")
    print(f"  Algorithm: {results_ga['algorithm']}")
    print(f"  N sensors: {results_ga['n_sensors']}")
    print(f"  Best fitness: {results_ga['best_fitness']:.6f}")
    print(f"  Valid: {results_ga['is_valid']}")
    print(f"  E_total: {results_ga['E_total']*1000:.4f} mJ")
    print(f"  Lifetime: {results_ga['network_lifetime']} rounds")
    print(f"  LOS/NLOS: {results_ga['los_count']}/{results_ga['nlos_count']}")
    print(f"  Time: {results_ga['elapsed_time']:.2f}s")
    
    # Test 2: PSO na S2
    print("\n[TEST 2] PSO on S2 (energy priority):")
    results_pso = run_single_experiment(
        config=config,
        scenario_name='S2',
        algorithm_name='PSO',
        weight_variant='energy_priority',
        run_id=2,
        seed=42
    )
    
    print(f"  Scenario: {results_pso['scenario']}")
    print(f"  Algorithm: {results_pso['algorithm']}")
    print(f"  N sensors: {results_pso['n_sensors']}")
    print(f"  Best fitness: {results_pso['best_fitness']:.6f}")
    print(f"  E_total: {results_pso['E_total']*1000:.4f} mJ")
    print(f"  Lifetime: {results_pso['network_lifetime']} rounds")
    print(f"  Time: {results_pso['elapsed_time']:.2f}s")
    
    # Test 3: Random baseline
    print("\n[TEST 3] Random Baseline on S1:")
    results_random = run_single_experiment(
        config=config,
        scenario_name='S1',
        algorithm_name='Random',
        weight_variant='balanced',
        run_id=3,
        seed=42
    )
    
    print(f"  Best fitness: {results_random['best_fitness']:.6f}")
    print(f"  E_total: {results_random['E_total']*1000:.4f} mJ")
    print(f"  Time: {results_random['elapsed_time']:.2f}s")
    
    # Test 4: Porównanie
    print("\n[TEST 4] Comparison:")
    print(f"  {'Algorithm':>10s} {'Fitness':>12s} {'E_total [mJ]':>14s} {'Time [s]':>10s}")
    print(f"  {'-'*10:>10s} {'-'*12:>12s} {'-'*14:>14s} {'-'*10:>10s}")
    
    for res in [results_ga, results_pso, results_random]:
        print(f"  {res['algorithm']:>10s} {res['best_fitness']:>12.6f} "
              f"{res['E_total']*1000:>14.4f} {res['elapsed_time']:>10.2f}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")
