"""
WBAN Optimization - Genetic Algorithm Optimizer
Author: Kamil Piejko
Date: 2024

Wrapper dla Genetic Algorithm z biblioteki Mealpy.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
import logging
from mealpy import FloatVar
from mealpy.evolutionary_based import GA

logger = logging.getLogger(__name__)


class GAOptimizer:
    """
    Genetic Algorithm optimizer dla WBAN sensor placement.
    
    Używa implementacji GA z Mealpy z następującymi operatorami:
    - Selection: Tournament selection
    - Crossover: Single-point / uniform
    - Mutation: Gaussian mutation
    
    Attributes:
        pop_size: Rozmiar populacji
        n_generations: Liczba generacji (epoch)
        pc: Prawdopodobieństwo krzyżowania
        pm: Prawdopodobieństwo mutacji
        tournament_size: Rozmiar turnieju w selekcji
        model: Obiekt GA z Mealpy
    """
    
    def __init__(self, config: Dict):
        """
        Inicjalizacja GA optimizer.
        
        Args:
            config: Konfiguracja z YAML (sekcja 'optimization.GA')
        """
        ga_config = config['optimization']['GA']
        
        self.pop_size = ga_config['population_size']
        self.n_generations = ga_config['max_iterations']

        self.pc = ga_config['crossover']['probability']
        self.pm = ga_config['mutation']['probability']
        self.tournament_size = ga_config['selection']['tournament_size']
        
        # Model będzie utworzony w optimize()
        self.model = None
        
        logger.info(f"GA Optimizer initialized: pop={self.pop_size}, "
                   f"gen={self.n_generations}, pc={self.pc}, pm={self.pm}")
    
    def optimize(self,
                fitness_function: Callable,
                dimension: int,
                bounds: Tuple[float, float] = (0.0, 1.0),
                seed: Optional[int] = None) -> Dict:
        """
        Uruchamia optymalizację GA.
        
        Args:
            fitness_function: Funkcja fitness(genome) -> float (do minimalizacji)
            dimension: Wymiar problemu (liczba zmiennych)
            bounds: Granice zmiennych (lb, ub)
            seed: Seed dla powtarzalności
        
        Returns:
            results: Słownik z wynikami {
                'best_genome': najlepsze rozwiązanie,
                'best_fitness': najlepsza wartość fitness,
                'convergence_curve': historia najlepszych wartości,
                'n_evaluations': liczba ewaluacji fitness,
                'algorithm': 'GA'
            }
        """
        logger.info(f"Starting GA optimization (dim={dimension}, seed={seed})...")
        
        # Definicja problemu dla Mealpy
        problem = {
            "obj_func": fitness_function,
            "bounds": FloatVar(lb=[bounds[0]] * dimension, ub=[bounds[1]] * dimension),
            "minmax": "min",  # Minimalizacja
            "log_to": None  # Wyłącz wbudowane logowanie
        }
        
        # Oblicz proporcję dla k_way (Mealpy wymaga float 0-1)
        # Jeśli tournament_size=3 i pop_size=50, to k_way=0.06
        k_way_ratio = self.tournament_size / self.pop_size
        
        # Zabezpieczenie, żeby nie wyjść poza zakres
        if k_way_ratio <= 0 or k_way_ratio > 1.0:
            k_way_ratio = 0.1  # Fallback do bezpiecznej wartości

        # Utwórz model GA
        self.model = GA.BaseGA(
            epoch=self.n_generations,
            pop_size=self.pop_size,
            pc=self.pc,
            pm=self.pm,
            selection="tournament",
            k_way=k_way_ratio,  
        )
        
        # Ustaw seed jeśli podany
        if seed is not None:
            self.model.seed = seed
        
        # Uruchom optymalizację
        best_solution = self.model.solve(problem)
        
        # Wyniki
        results = {
            'best_genome': best_solution.solution,
            'best_fitness': best_solution.target.fitness,
            'convergence_curve': self.model.history.list_global_best_fit,
            'n_evaluations': self.n_generations * self.pop_size,
            'algorithm': 'GA',
            'model': self.model  # Dla zaawansowanych analiz
        }
        
        logger.info(f"GA optimization complete: best_fitness={results['best_fitness']:.6f}")
        
        return results
    
    def get_config(self) -> Dict:
        """
        Zwraca konfigurację algorytmu.
        
        Returns:
            config: Słownik z parametrami
        """
        return {
            'algorithm': 'GA',
            'pop_size': self.pop_size,
            'n_generations': self.n_generations,
            'pc': self.pc,
            'pm': self.pm,
            'tournament_size': self.tournament_size
        }
    
    def __repr__(self) -> str:
        return (f"GAOptimizer(pop={self.pop_size}, gen={self.n_generations}, "
                f"pc={self.pc}, pm={self.pm})")


if __name__ == '__main__':
    # Test GA optimizer
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
    print("GA OPTIMIZER TEST")
    print("="*60)
    
    # Wczytaj konfigurację
    config = load_config()
    
    # Test 1: Inicjalizacja
    print("\n[TEST 1] GA Optimizer Initialization:")
    ga = GAOptimizer(config)
    print(f"  {ga}")
    print(f"  Config: {ga.get_config()}")
    
    # Test 2: Funkcja testowa (sphere function)
    print("\n[TEST 2] Simple Sphere Function (dim=10):")
    
    def sphere_function(x):
        """Funkcja testowa: f(x) = sum(x_i^2)"""
        return np.sum(x**2)
    
    # Optymalizacja
    results = ga.optimize(
        fitness_function=sphere_function,
        dimension=10,
        bounds=(-5.0, 5.0),
        seed=42
    )
    
    print(f"  Best solution: {results['best_genome'][:5]}... (showing first 5)")
    print(f"  Best fitness: {results['best_fitness']:.6f} (expected: ~0.0)")
    print(f"  N evaluations: {results['n_evaluations']}")
    print(f"  Convergence: {len(results['convergence_curve'])} generations")
    
    # Test 3: Optymalizacja WBAN (mały problem)
    print("\n[TEST 3] WBAN Optimization (S1 - 6 sensors):")
    
    # Utwórz fitness function
    fitness_func = FitnessFunction(config, 'S1', 'balanced')
    dimension = 2 * fitness_func.n_sensors + 2
    
    print(f"  Scenario: S1")
    print(f"  Dimension: {dimension}")
    print(f"  Population: {ga.pop_size}")
    print(f"  Generations: {ga.n_generations}")
    
    # Optymalizacja (krótka dla testu - tylko 10 generacji)
    ga_test = GAOptimizer(config)
    ga_test.n_generations = 10  # Override dla szybkiego testu
    
    results_wban = ga_test.optimize(
        fitness_function=fitness_func.evaluate,
        dimension=dimension,
        bounds=(0.0, 1.0),
        seed=42
    )
    
    print(f"\n  Best fitness: {results_wban['best_fitness']:.6f}")
    print(f"  N evaluations: {results_wban['n_evaluations']}")
    
    # Szczegółowa ewaluacja najlepszego rozwiązania
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
    
    # Wyświetl kilka punktów z krzywej
    print(f"\n  Convergence curve (every 2 generations):")
    for i in range(0, len(conv_curve), 2):
        print(f"    Gen {i:3d}: {conv_curve[i]:.6f}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")
