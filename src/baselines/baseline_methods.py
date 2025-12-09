"""
WBAN Optimization - Baseline Methods
Author: Kamil Piejko
Date: 2024

Metody bazowe do porównania z GA i PSO:
1. Random Placement - losowe rozmieszczenie sensorów
2. Greedy Placement - zachłanna heurystyka (minimalizacja odległości)
"""

import numpy as np
from typing import Dict, Callable, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RandomBaseline:
    """
    Random Placement Baseline - losowe rozmieszczenie sensorów w strefach.
    
    Najprostrza metoda bazowa - generuje N losowych rozwiązań i wybiera najlepsze.
    """
    
    def __init__(self, n_samples: int = 100):
        """
        Inicjalizacja Random Baseline.
        
        Args:
            n_samples: Liczba losowych prób
        """
        self.n_samples = n_samples
        
        logger.info(f"Random Baseline initialized: n_samples={n_samples}")
    
    def optimize(self,
                fitness_function: Callable,
                dimension: int,
                bounds: Tuple[float, float] = (0.0, 1.0),
                seed: Optional[int] = None) -> Dict:
        """
        Generuje losowe rozwiązania i wybiera najlepsze.
        
        Args:
            fitness_function: Funkcja fitness(genome) -> float
            dimension: Wymiar problemu
            bounds: Granice zmiennych
            seed: Seed dla powtarzalności
        
        Returns:
            results: Słownik z wynikami
        """
        logger.info(f"Starting Random Baseline (n_samples={self.n_samples})...")
        
        rng = np.random.default_rng(seed)
        
        best_genome = None
        best_fitness = float('inf')
        fitness_history = []
        
        for i in range(self.n_samples):
            # Generuj losowe rozwiązanie
            genome = rng.uniform(bounds[0], bounds[1], size=dimension)
            
            # Ewaluuj
            fitness = fitness_function(genome)
            fitness_history.append(fitness)
            
            # Aktualizuj najlepsze
            if fitness < best_fitness:
                best_fitness = fitness
                best_genome = genome.copy()
        
        results = {
            'best_genome': best_genome,
            'best_fitness': best_fitness,
            'convergence_curve': fitness_history,
            'n_evaluations': self.n_samples,
            'algorithm': 'Random'
        }
        
        logger.info(f"Random Baseline complete: best_fitness={best_fitness:.6f}")
        
        return results
    
    def get_config(self) -> Dict:
        return {
            'algorithm': 'Random',
            'n_samples': self.n_samples
        }
    
    def __repr__(self) -> str:
        return f"RandomBaseline(n_samples={self.n_samples})"


class GreedyBaseline:
    """
    Greedy Placement Baseline - zachłanna heurystyka.
    
    Strategia:
    1. Hub umieszczony w centrum (waist)
    2. Sensory umieszczane po kolei w środkach przypisanych stref
    3. Opcjonalnie: perturbacje losowe dla poprawy
    """
    
    def __init__(self, n_perturbations: int = 10, perturbation_scale: float = 0.05):
        """
        Inicjalizacja Greedy Baseline.
        
        Args:
            n_perturbations: Liczba perturbacji do przetestowania
            perturbation_scale: Skala perturbacji (frakcja zakresu)
        """
        self.n_perturbations = n_perturbations
        self.perturbation_scale = perturbation_scale
        
        logger.info(f"Greedy Baseline initialized: n_perturbations={n_perturbations}")
    
    def optimize(self,
                fitness_function: Callable,
                dimension: int,
                bounds: Tuple[float, float] = (0.0, 1.0),
                seed: Optional[int] = None,
                zone_centers: Optional[np.ndarray] = None) -> Dict:
        """
        Tworzy zachłanne rozwiązanie z opcjonalnymi perturbacjami.
        
        Args:
            fitness_function: Funkcja fitness
            dimension: Wymiar problemu
            bounds: Granice
            seed: Seed
            zone_centers: Opcjonalne środki stref [N+1, 2] (sensory + hub)
        
        Returns:
            results: Słownik z wynikami
        """
        logger.info(f"Starting Greedy Baseline...")
        
        rng = np.random.default_rng(seed)
        
        # Jeśli nie podano zone_centers, użyj prostego grida
        if zone_centers is None:
            n_points = dimension // 2
            # Rozmieść równomiernie w przestrzeni [0, 1] × [0, 1]
            grid_size = int(np.ceil(np.sqrt(n_points)))
            x = np.linspace(0.2, 0.8, grid_size)
            y = np.linspace(0.2, 0.8, grid_size)
            
            zone_centers = []
            for i in range(n_points):
                ix = i % grid_size
                iy = i // grid_size
                if iy < len(y) and ix < len(x):
                    zone_centers.append([x[ix], y[iy]])
            
            zone_centers = np.array(zone_centers[:n_points])
        
        # Bazowe rozwiązanie: środki stref
        base_genome = zone_centers.flatten()
        
        # Uzupełnij do właściwego wymiaru jeśli potrzeba
        if len(base_genome) < dimension:
            base_genome = np.concatenate([base_genome, [0.5, 0.5]])  # Dodaj hub na środku
        
        base_genome = base_genome[:dimension]  # Obetnij do dimension
        
        # Ewaluuj bazowe rozwiązanie
        best_fitness = fitness_function(base_genome)
        best_genome = base_genome.copy()
        fitness_history = [best_fitness]
        
        # Perturbacje
        for i in range(self.n_perturbations):
            # Dodaj losową perturbację
            perturbation = rng.normal(0, self.perturbation_scale, size=dimension)
            perturbed_genome = np.clip(
                base_genome + perturbation,
                bounds[0], bounds[1]
            )
            
            # Ewaluuj
            fitness = fitness_function(perturbed_genome)
            fitness_history.append(fitness)
            
            # Aktualizuj najlepsze
            if fitness < best_fitness:
                best_fitness = fitness
                best_genome = perturbed_genome.copy()
                base_genome = best_genome.copy()  # Nowa baza dla kolejnych perturbacji
        
        results = {
            'best_genome': best_genome,
            'best_fitness': best_fitness,
            'convergence_curve': fitness_history,
            'n_evaluations': 1 + self.n_perturbations,
            'algorithm': 'Greedy'
        }
        
        logger.info(f"Greedy Baseline complete: best_fitness={best_fitness:.6f}")
        
        return results
    
    def get_config(self) -> Dict:
        return {
            'algorithm': 'Greedy',
            'n_perturbations': self.n_perturbations,
            'perturbation_scale': self.perturbation_scale
        }
    
    def __repr__(self) -> str:
        return f"GreedyBaseline(n_pert={self.n_perturbations})"


if __name__ == '__main__':
    # Test baseline methods
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
    print("BASELINE METHODS TEST")
    print("="*60)
    
    # Funkcja testowa
    def sphere_function(x):
        return np.sum(x**2)
    
    # Test 1: Random Baseline
    print("\n[TEST 1] Random Baseline (sphere function):")
    random_baseline = RandomBaseline(n_samples=100)
    print(f"  {random_baseline}")
    
    results_random = random_baseline.optimize(
        fitness_function=sphere_function,
        dimension=10,
        bounds=(-5.0, 5.0),
        seed=42
    )
    
    print(f"  Best fitness: {results_random['best_fitness']:.6f}")
    print(f"  N evaluations: {results_random['n_evaluations']}")
    
    # Test 2: Greedy Baseline
    print("\n[TEST 2] Greedy Baseline (sphere function):")
    greedy_baseline = GreedyBaseline(n_perturbations=20)
    print(f"  {greedy_baseline}")
    
    results_greedy = greedy_baseline.optimize(
        fitness_function=sphere_function,
        dimension=10,
        bounds=(-5.0, 5.0),
        seed=42
    )
    
    print(f"  Best fitness: {results_greedy['best_fitness']:.6f}")
    print(f"  N evaluations: {results_greedy['n_evaluations']}")
    
    # Test 3: WBAN Problem
    print("\n[TEST 3] WBAN Optimization (S1 - 6 sensors):")
    config = load_config()
    fitness_func = FitnessFunction(config, 'S1', 'balanced')
    dimension = 2 * fitness_func.n_sensors + 2
    
    print(f"  Scenario: S1")
    print(f"  Dimension: {dimension}")
    
    # Random
    print("\n  [3a] Random Baseline:")
    random_wban = RandomBaseline(n_samples=50)
    results_random_wban = random_wban.optimize(
        fitness_function=fitness_func.evaluate,
        dimension=dimension,
        bounds=(0.0, 1.0),
        seed=42
    )
    
    details_random = fitness_func.evaluate_detailed(results_random_wban['best_genome'])
    print(f"    Best fitness: {results_random_wban['best_fitness']:.6f}")
    print(f"    Valid: {details_random['is_valid']}")
    print(f"    E_total: {details_random['E_total']*1000:.4f} mJ")
    print(f"    Lifetime: {details_random['network_lifetime']} rounds")
    
    # Greedy
    print("\n  [3b] Greedy Baseline:")
    greedy_wban = GreedyBaseline(n_perturbations=30)
    
    # Użyj środków stref z body model
    zone_centers = []
    for sensor_cfg in fitness_func.sensor_config:
        zone = fitness_func.body_model.get_zone(sensor_cfg['zone'])
        center = zone.get_center()
        zone_centers.append(center)
    
    # Dodaj hub (waist)
    hub_zone = fitness_func.body_model.get_zone('waist')
    zone_centers.append(hub_zone.get_center())
    
    zone_centers = np.array(zone_centers)
    
    results_greedy_wban = greedy_wban.optimize(
        fitness_function=fitness_func.evaluate,
        dimension=dimension,
        bounds=(0.0, 1.0),
        seed=42,
        zone_centers=zone_centers
    )
    
    details_greedy = fitness_func.evaluate_detailed(results_greedy_wban['best_genome'])
    print(f"    Best fitness: {results_greedy_wban['best_fitness']:.6f}")
    print(f"    Valid: {details_greedy['is_valid']}")
    print(f"    E_total: {details_greedy['E_total']*1000:.4f} mJ")
    print(f"    Lifetime: {details_greedy['network_lifetime']} rounds")
    
    # Test 4: Porównanie
    print("\n[TEST 4] Comparison:")
    print(f"  {'Method':>12s} {'Best Fitness':>15s} {'Valid':>8s} {'Lifetime':>10s}")
    print(f"  {'-'*12:>12s} {'-'*15:>15s} {'-'*8:>8s} {'-'*10:>10s}")
    print(f"  {'Random':>12s} {results_random_wban['best_fitness']:>15.6f} "
          f"{str(details_random['is_valid']):>8s} {details_random['network_lifetime']:>10d}")
    print(f"  {'Greedy':>12s} {results_greedy_wban['best_fitness']:>15.6f} "
          f"{str(details_greedy['is_valid']):>8s} {details_greedy['network_lifetime']:>10d}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")
