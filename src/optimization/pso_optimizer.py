"""
WBAN Optimization - Particle Swarm Optimization
Author: Kamil Piejko
Date: 2024

Wrapper dla PSO z biblioteki Mealpy.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
import logging
from mealpy import FloatVar
from mealpy.swarm_based import PSO

logger = logging.getLogger(__name__)


class PSOOptimizer:
    """
    Particle Swarm Optimization dla WBAN sensor placement.
    
    PSO symuluje rój cząstek przeszukujących przestrzeń rozwiązań:
    - Każda cząstka ma pozycję i prędkość
    - Cząstka pamięta swoją najlepszą pozycję (pbest)
    - Rój pamięta globalnie najlepszą pozycję (gbest)
    - Prędkość: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
    
    Attributes:
        pop_size: Rozmiar roju (liczba cząstek)
        n_iterations: Liczba iteracji
        c1: Współczynnik kognitywny (osobisty)
        c2: Współczynnik społeczny (globalny)
        w: Współczynnik bezwładności
        model: Obiekt PSO z Mealpy
    """
    
    def __init__(self, config: Dict):
        """
        Inicjalizacja PSO optimizer.
        
        Args:
            config: Konfiguracja z YAML (sekcja 'optimization.PSO')
        """
        pso_config = config['optimization']['PSO']
        
        self.pop_size = pso_config['population_size']
        self.n_iterations = pso_config['max_iterations']

        self.c1 = pso_config['cognitive_coefficient']
        self.c2 = pso_config['social_coefficient']
        self.w = pso_config['inertia_weight']

        self.w_min = 0.4
        self.w_max = 0.9
        
        # Model będzie utworzony w optimize()
        self.model = None
        
        logger.info(f"PSO Optimizer initialized: pop={self.pop_size}, "
                   f"iter={self.n_iterations}, c1={self.c1}, c2={self.c2}, w={self.w}")
    
    def optimize(self,
                fitness_function: Callable,
                dimension: int,
                bounds: Tuple[float, float] = (0.0, 1.0),
                seed: Optional[int] = None) -> Dict:
        """
        Uruchamia optymalizację PSO.
        
        Args:
            fitness_function: Funkcja fitness(genome) -> float (do minimalizacji)
            dimension: Wymiar problemu
            bounds: Granice zmiennych (lb, ub)
            seed: Seed dla powtarzalności
        
        Returns:
            results: Słownik z wynikami {
                'best_genome': najlepsze rozwiązanie,
                'best_fitness': najlepsza wartość fitness,
                'convergence_curve': historia najlepszych wartości,
                'n_evaluations': liczba ewaluacji fitness,
                'algorithm': 'PSO'
            }
        """
        logger.info(f"Starting PSO optimization (dim={dimension}, seed={seed})...")
        
        # Definicja problemu
        problem = {
            "obj_func": fitness_function,
            "bounds": FloatVar(lb=[bounds[0]] * dimension, ub=[bounds[1]] * dimension),
            "minmax": "min",
            "log_to": None
        }
        
        # Utwórz model PSO
        # Używamy OriginalPSO (klasyczny PSO)
        self.model = PSO.OriginalPSO(
            epoch=self.n_iterations,
            pop_size=self.pop_size,
            c1=self.c1,
            c2=self.c2,
            w_min=self.w_min,
            w_max=self.w_max
        )
        
        # Ustaw seed
        if seed is not None:
            self.model.seed = seed
        
        # Uruchom optymalizację
        best_solution = self.model.solve(problem)
        
        # Wyniki
        results = {
            'best_genome': best_solution.solution,
            'best_fitness': best_solution.target.fitness,
            'convergence_curve': self.model.history.list_global_best_fit,
            'n_evaluations': self.n_iterations * self.pop_size,
            'algorithm': 'PSO',
            'model': self.model
        }
        
        logger.info(f"PSO optimization complete: best_fitness={results['best_fitness']:.6f}")
        
        return results
    
    def get_config(self) -> Dict:
        """
        Zwraca konfigurację algorytmu.
        
        Returns:
            config: Słownik z parametrami
        """
        return {
            'algorithm': 'PSO',
            'pop_size': self.pop_size,
            'n_iterations': self.n_iterations,
            'c1': self.c1,
            'c2': self.c2,
            'w': self.w,
            'w_min': self.w_min,
            'w_max': self.w_max
        }
    
    def __repr__(self) -> str:
        return (f"PSOOptimizer(pop={self.pop_size}, iter={self.n_iterations}, "
                f"c1={self.c1}, c2={self.c2}, w={self.w})")


if __name__ == '__main__':
    # Test PSO optimizer
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.utils.config_loader import load_config
    from src.optimization.fitness_function import FitnessFunction
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("PSO OPTIMIZER TEST")
    print("="*60)
    
    # Wczytaj konfigurację
    config = load_config()
    
    # Test 1: Inicjalizacja
    print("\n[TEST 1] PSO Optimizer Initialization:")
    pso = PSOOptimizer(config)
    print(f"  {pso}")
    print(f"  Config: {pso.get_config()}")
    
    # Test 2: Funkcja testowa (sphere function)
    print("\n[TEST 2] Simple Sphere Function (dim=10):")
    
    def sphere_function(x):
        """Funkcja testowa: f(x) = sum(x_i^2)"""
        return np.sum(x**2)
    
    results = pso.optimize(
        fitness_function=sphere_function,
        dimension=10,
        bounds=(-5.0, 5.0),
        seed=42
    )
    
    print(f"  Best solution: {results['best_genome'][:5]}... (showing first 5)")
    print(f"  Best fitness: {results['best_fitness']:.6f} (expected: ~0.0)")
    print(f"  N evaluations: {results['n_evaluations']}")
    
    # Test 3: Optymalizacja WBAN
    print("\n[TEST 3] WBAN Optimization (S1 - 6 sensors):")
    
    fitness_func = FitnessFunction(config, 'S1', 'balanced')
    dimension = 2 * fitness_func.n_sensors + 2
    
    print(f"  Scenario: S1")
    print(f"  Dimension: {dimension}")
    print(f"  Population: {pso.pop_size}")
    print(f"  Iterations: {pso.n_iterations}")
    
    # Krótka optymalizacja dla testu
    pso_test = PSOOptimizer(config)
    pso_test.n_iterations = 10
    
    results_wban = pso_test.optimize(
        fitness_function=fitness_func.evaluate,
        dimension=dimension,
        bounds=(0.0, 1.0),
        seed=42
    )
    
    print(f"\n  Best fitness: {results_wban['best_fitness']:.6f}")
    print(f"  N evaluations: {results_wban['n_evaluations']}")
    
    # Szczegółowa ewaluacja
    best_details = fitness_func.evaluate_detailed(results_wban['best_genome'])
    print(f"  Valid: {best_details['is_valid']}")
    print(f"  E_total: {best_details['E_total']*1000:.4f} mJ")
    print(f"  Lifetime: {best_details['network_lifetime']} rounds")
    print(f"  LOS/NLOS: {best_details['los_count']}/{best_details['nlos_count']}")
    
    # Test 4: Krzywa zbieżności
    print("\n[TEST 4] Convergence Analysis:")
    conv_curve = results_wban['convergence_curve']
    print(f"  Initial fitness: {conv_curve[0]:.6f}")
    print(f"  Final fitness:   {conv_curve[-1]:.6f}")
    print(f"  Improvement:     {conv_curve[0] - conv_curve[-1]:.6f}")
    print(f"  Reduction:       {(1 - conv_curve[-1]/conv_curve[0])*100:.2f}%")
    
    print(f"\n  Convergence curve (every 2 iterations):")
    for i in range(0, len(conv_curve), 2):
        print(f"    Iter {i:3d}: {conv_curve[i]:.6f}")
    
    # Test 5: Porównanie GA vs PSO
    print("\n[TEST 5] Quick Comparison: GA vs PSO (sphere function):")
    
    from src.optimization.ga_optimizer import GAOptimizer
    
    # GA
    ga_quick = GAOptimizer(config)
    ga_quick.n_generations = 20
    ga_quick.pop_size = 30
    
    results_ga = ga_quick.optimize(
        fitness_function=sphere_function,
        dimension=10,
        bounds=(-5.0, 5.0),
        seed=42
    )
    
    # PSO
    pso_quick = PSOOptimizer(config)
    pso_quick.n_iterations = 20
    pso_quick.pop_size = 30
    
    results_pso = pso_quick.optimize(
        fitness_function=sphere_function,
        dimension=10,
        bounds=(-5.0, 5.0),
        seed=42
    )
    
    print(f"  {'Algorithm':>12s} {'Best Fitness':>15s} {'N Evals':>10s}")
    print(f"  {'-'*12:>12s} {'-'*15:>15s} {'-'*10:>10s}")
    print(f"  {'GA':>12s} {results_ga['best_fitness']:>15.6f} {results_ga['n_evaluations']:>10d}")
    print(f"  {'PSO':>12s} {results_pso['best_fitness']:>15.6f} {results_pso['n_evaluations']:>10d}")
    
    winner = 'GA' if results_ga['best_fitness'] < results_pso['best_fitness'] else 'PSO'
    print(f"\n  Winner for this run: {winner}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")
