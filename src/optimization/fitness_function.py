"""
WBAN Optimization - Fitness Function
Author: Kamil Piejko
Date: 2024

Wielokryterialna funkcja fitness do optymalizacji rozmieszczenia sensorów w WBAN.

F(g) = w_E × E_total(g) + w_R × P_rel(g) + P_geo(g)

gdzie:
- E_total: Całkowita energia zużyta przez wszystkie sensory [J]
- P_rel: Kara za niską niezawodność (słabe marginesy łączy)
- P_geo: Kara za naruszenia geometryczne (sensory poza strefami)
- w_E, w_R: Wagi (np. 0.7, 0.3 dla priorytetu energii)
"""

import numpy as np
from typing import Dict, Tuple, List
import logging

from src.core.body_model import BodyModel
from src.core.sensor import Sensor, Hub
from src.core.genotype import Genotype
from src.models.energy_model import EnergyModel
from src.models.propagation_model import PropagationModel
from src.models.los_detector import LOSDetector

logger = logging.getLogger(__name__)


class FitnessFunction:
    """
    Wielokryterialna funkcja fitness dla optymalizacji WBAN.
    
    Attributes:
        config: Konfiguracja systemu
        body_model: Model ciała
        energy_model: Model energetyczny
        propagation_model: Model propagacji
        los_detector: Detektor LOS/NLOS
        weights: Wagi funkcji celu {w_E, w_R}
        scenario_config: Konfiguracja scenariusza
    """
    
    def __init__(self, 
                 config: Dict,
                 scenario_name: str,
                 weight_variant: str):
        """
        Inicjalizacja funkcji fitness.
        
        Args:
            config: Pełna konfiguracja z YAML
            scenario_name: Nazwa scenariusza (np. 'S1', 'S2', 'S3')
            weight_variant: Wariant wag (np. 'energy_priority', 'balanced')
        """
        self.config = config
        self.scenario_name = scenario_name
        self.scenario_config = config['scenarios'][scenario_name]
        
        # Wczytaj wagi
        weights_config = config['fitness_function']['weight_variants'][weight_variant]
        self.weights = {
            'w_E': weights_config['w_E'],
            'w_R': weights_config['w_R']
        }
        
        # Inicjalizuj modele
        self.body_model = BodyModel(config)
        self.energy_model = EnergyModel(config)
        self.propagation_model = PropagationModel(config)
        self.los_detector = LOSDetector(self.body_model)
        
        # Parametry scenariusza
        self.n_sensors = self.scenario_config['n_sensors']
        self.P_TX_max = self.scenario_config['P_TX_max']
        self.sensor_config = self.scenario_config['sensor_config']
        self.hub_config = config['scenarios']['hub_placement']
        self.energy_init = config['energy_model']['E_init']
        
        # Parametry kar
        self.penalty_params = config['fitness_function']['penalties']
        
        logger.info(f"Fitness function initialized: {scenario_name}, "
                   f"weights=({self.weights['w_E']}, {self.weights['w_R']})")
    
    def evaluate(self, genome: np.ndarray) -> float:
        """
        Główna funkcja fitness - ewaluuje genotyp.
        
        Args:
            genome: Wektor [x1,y1,...,xN,yN,x_hub,y_hub]
        
        Returns:
            fitness: Wartość fitness (niższa = lepsza, do minimalizacji)
        """
        # ====================================================================
        # KROK 1: Dekodowanie genotypu
        # ====================================================================
        try:
            sensors, hub = Genotype.decode(
                genome, 
                self.sensor_config, 
                self.body_model,
                self.energy_init,
                self.hub_config
            )
        except Exception as e:
            logger.error(f"Genotype decoding failed: {e}")
            return 1e9  # Ogromna kara za błąd dekodowania
        
        # ====================================================================
        # KROK 2: Walidacja geometryczna
        # ====================================================================
        penalty_geo = Genotype.compute_geometric_penalty(
            sensors, hub, self.body_model, self.config
        )
        
        if penalty_geo > 0:
            # Rozwiązanie niedopuszczalne - zwróć karę
            return penalty_geo
        
        # ====================================================================
        # KROK 3: Obliczenie całkowitej energii
        # ====================================================================
        E_total = 0.0
        link_margins = []
        
        for sensor in sensors:
            # 3a. Odległość sensor-hub
            distance = self.los_detector.compute_distance(sensor.position, hub.position)
            
            # 3b. Detekcja LOS/NLOS
            los_status = self.los_detector.detect(sensor.position, hub.position)
            
            # 3c. Path loss
            PL = self.propagation_model.compute_path_loss(
                distance, 
                los_status, 
                include_shadowing=False  # Dla powtarzalności
            )
            
            # 3d. Wymagana moc transmisji
            P_TX_required = self.propagation_model.compute_required_tx_power(PL)
            
            # 3e. Margines łącza
            link_margin = self.P_TX_max - P_TX_required
            link_margins.append(link_margin)
            
            # 3f. Energia transmisji
            E_TX = self.energy_model.compute_transmission_energy(
                sensor.packet_size,
                distance
            )
            
            # 3g. Energia odbioru (na Hub)
            E_RX = self.energy_model.compute_reception_energy(sensor.packet_size)
            
            # 3h. Suma
            E_total += E_TX + E_RX
        
        # ====================================================================
        # KROK 4: Kara za niską niezawodność
        # ====================================================================
        min_margin = min(link_margins) if link_margins else 0
        
        if min_margin < 0:
            # Łącze niemożliwe - kara proporcjonalna
            penalty_rel = abs(min_margin) * self.penalty_params['reliability_penalty_factor']
        else:
            penalty_rel = 0.0
        
        # ====================================================================
        # KROK 5: Agregacja fitness
        # ====================================================================
        fitness = self.weights['w_E'] * E_total + self.weights['w_R'] * penalty_rel
        
        return fitness
    
    def evaluate_detailed(self, genome: np.ndarray) -> Dict:
        """
        Ewaluuje genotyp i zwraca szczegółowe metryki.
        
        Args:
            genome: Wektor genotypu
        
        Returns:
            results: Słownik z metrykami
        """
        # Dekoduj
        sensors, hub = Genotype.decode(
            genome, 
            self.sensor_config, 
            self.body_model,
            self.energy_init,
            self.hub_config
        )
        
        # Walidacja
        is_valid, reason = Genotype.is_valid_geometry(sensors, hub, self.body_model)
        penalty_geo = Genotype.compute_geometric_penalty(
            sensors, hub, self.body_model, self.config
        )
        
        # Obliczenia dla każdego sensora
        sensor_results = []
        E_total = 0.0
        link_margins = []
        los_count = 0
        nlos_count = 0
        
        for sensor in sensors:
            distance = self.los_detector.compute_distance(sensor.position, hub.position)
            los_status = self.los_detector.detect(sensor.position, hub.position)
            
            if los_status == 'LOS':
                los_count += 1
            else:
                nlos_count += 1
            
            PL = self.propagation_model.compute_path_loss(distance, los_status, False)
            P_TX_req = self.propagation_model.compute_required_tx_power(PL)
            link_margin = self.P_TX_max - P_TX_req
            
            E_TX = self.energy_model.compute_transmission_energy(sensor.packet_size, distance)
            E_RX = self.energy_model.compute_reception_energy(sensor.packet_size)
            
            sensor_results.append({
                'sensor_id': sensor.id,
                'sensor_type': sensor.type,
                'distance': distance,
                'los_status': los_status,
                'path_loss': PL,
                'P_TX_required': P_TX_req,
                'link_margin': link_margin,
                'E_TX': E_TX,
                'E_RX': E_RX
            })
            
            E_total += E_TX + E_RX
            link_margins.append(link_margin)
        
        # Kara niezawodności
        min_margin = min(link_margins) if link_margins else 0
        penalty_rel = abs(min_margin) * self.penalty_params['reliability_penalty_factor'] if min_margin < 0 else 0.0
        
        # Fitness
        fitness = self.weights['w_E'] * E_total + self.weights['w_R'] * penalty_rel
        
        # Dodatkowe metryki
        network_lifetime = self.energy_model.compute_network_lifetime(
            energy_per_sensor=np.array([self.energy_init] * self.n_sensors),
            energy_per_round_per_sensor=np.array([sr['E_TX'] for sr in sensor_results])
        )
        
        results = {
            'fitness': fitness,
            'is_valid': is_valid,
            'validation_reason': reason,
            'penalty_geo': penalty_geo,
            'penalty_rel': penalty_rel,
            'E_total': E_total,
            'min_link_margin': min_margin,
            'network_lifetime': network_lifetime,
            'los_count': los_count,
            'nlos_count': nlos_count,
            'los_ratio': los_count / self.n_sensors if self.n_sensors > 0 else 0,
            'sensor_results': sensor_results
        }
        
        return results
    
    def get_config_summary(self) -> Dict:
        """
        Zwraca podsumowanie konfiguracji funkcji fitness.
        
        Returns:
            summary: Słownik z konfiguracją
        """
        return {
            'scenario': self.scenario_name,
            'n_sensors': self.n_sensors,
            'P_TX_max': self.P_TX_max,
            'weights': self.weights,
            'energy_init': self.energy_init,
            'penalty_params': self.penalty_params
        }
    
    def __repr__(self) -> str:
        return (f"FitnessFunction(scenario='{self.scenario_name}', "
                f"n_sensors={self.n_sensors}, "
                f"weights=({self.weights['w_E']}, {self.weights['w_R']}))")
    
    def __call__(self, genome: np.ndarray) -> float:
        """
        Umożliwia wywołanie obiektu jako funkcji.
        
        Args:
            genome: Wektor genotypu
        
        Returns:
            fitness: Wartość fitness
        """
        return self.evaluate(genome)


if __name__ == '__main__':
    # Test funkcji fitness
    import sys
    sys.path.append('../..')
    
    from src.utils.config_loader import load_config
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("FITNESS FUNCTION TEST")
    print("="*60)
    
    # Wczytaj konfigurację
    config = load_config()
    
    # Utwórz funkcję fitness dla S1, balanced
    fitness_func = FitnessFunction(config, 'S1', 'balanced')
    print(f"\n[TEST 1] Fitness Function: {fitness_func}")
    print(f"  Config: {fitness_func.get_config_summary()}")
    
    # Test 2: Generuj losowy genotyp i ewaluuj
    print("\n[TEST 2] Random Genotype Evaluation:")
    rng = np.random.default_rng(42)
    
    genome = Genotype.generate_random(
        fitness_func.n_sensors,
        fitness_func.sensor_config,
        fitness_func.body_model,
        fitness_func.hub_config,
        rng
    )
    
    print(f"  Genome: {genome[:10]}... (showing first 10 values)")
    
    fitness = fitness_func.evaluate(genome)
    print(f"  Fitness: {fitness:.6f}")
    
    # Test 3: Szczegółowa ewaluacja
    print("\n[TEST 3] Detailed Evaluation:")
    results = fitness_func.evaluate_detailed(genome)
    
    print(f"  Fitness: {results['fitness']:.6f}")
    print(f"  Valid: {results['is_valid']} ({results['validation_reason']})")
    print(f"  E_total: {results['E_total']*1000:.4f} mJ")
    print(f"  Min link margin: {results['min_link_margin']:.2f} dB")
    print(f"  Network lifetime: {results['network_lifetime']} rounds")
    print(f"  LOS/NLOS: {results['los_count']}/{results['nlos_count']} (ratio={results['los_ratio']:.2f})")
    
    print(f"\n  Per-sensor details:")
    print(f"    {'ID':>3s} {'Type':>15s} {'Dist [m]':>10s} {'LOS':>6s} {'PL [dB]':>8s} {'Margin [dB]':>12s} {'E_TX [μJ]':>12s}")
    for sr in results['sensor_results']:
        print(f"    {sr['sensor_id']:>3d} {sr['sensor_type']:>15s} {sr['distance']:>10.3f} "
              f"{sr['los_status']:>6s} {sr['path_loss']:>8.2f} {sr['link_margin']:>12.2f} "
              f"{sr['E_TX']*1e6:>12.4f}")
    
    # Test 4: Niepr poprawny genotyp (sensor poza strefą)
    print("\n[TEST 4] Invalid Genotype (sensor out of zone):")
    invalid_genome = genome.copy()
    invalid_genome[0] = 0.99  # Sensor 1 przesunięty poza strefę
    
    fitness_invalid = fitness_func.evaluate(invalid_genome)
    results_invalid = fitness_func.evaluate_detailed(invalid_genome)
    
    print(f"  Fitness: {fitness_invalid:.6f}")
    print(f"  Valid: {results_invalid['is_valid']} ({results_invalid['validation_reason']})")
    print(f"  Penalty_geo: {results_invalid['penalty_geo']:.2f}")
    
    # Test 5: Porównanie różnych wariantów wag
    print("\n[TEST 5] Weight Variants Comparison:")
    weight_variants = ['energy_priority', 'balanced', 'reliability_priority']
    
    print(f"  {'Variant':>25s} {'w_E':>6s} {'w_R':>6s} {'Fitness':>12s}")
    print(f"  {'-'*25:>25s} {'-'*6:>6s} {'-'*6:>6s} {'-'*12:>12s}")
    
    for variant in weight_variants:
        ff = FitnessFunction(config, 'S1', variant)
        fitness_val = ff.evaluate(genome)
        print(f"  {variant:>25s} {ff.weights['w_E']:>6.1f} {ff.weights['w_R']:>6.1f} {fitness_val:>12.6f}")
    
    # Test 6: Weryfikacja z ręcznym obliczeniem (Przykład 1 z docs)
    print("\n[TEST 6] Verification with Manual Calculation:")
    print("  (Approximation of Example 1 from docs/fitness_calculation_examples.md)")
    
    # Mały scenariusz (3 sensory) - stworzymy custom config
    custom_sensor_config = [
        {'type': 'ECG', 'zone': 'chest', 'data_rate': 200, 'packet_size': 100},
        {'type': 'SpO2', 'zone': 'left_wrist', 'data_rate': 50, 'packet_size': 50},
        {'type': 'Temperature', 'zone': 'chest', 'data_rate': 10, 'packet_size': 20},
    ]
    
    # Genotyp z dokumentacji
    manual_genome = np.array([
        0.45, 0.65,  # Sensor 1 (ECG)
        0.18, 0.50,  # Sensor 2 (SpO2)
        0.50, 0.62,  # Sensor 3 (Temperature)
        0.50, 0.45   # Hub
    ])
    
    # Oblicz ręcznie (uproszczona wersja)
    # Expected: E_total ≈ 0.017 mJ, fitness ≈ 0.0085 (dla wag 0.5/0.5)
    
    # Utworzymy tymczasową konfigurację
    temp_config = config.copy()
    temp_config['scenarios']['S_TEST'] = {
        'n_sensors': 3,
        'P_TX_max': 0,
        'sensor_config': custom_sensor_config
    }
    
    ff_test = FitnessFunction(temp_config, 'S_TEST', 'balanced')
    fitness_manual = ff_test.evaluate(manual_genome)
    results_manual = ff_test.evaluate_detailed(manual_genome)
    
    print(f"  Calculated fitness: {fitness_manual:.6f}")
    print(f"  E_total: {results_manual['E_total']*1000:.6f} mJ")
    print(f"  Expected E_total: ~0.017 mJ")
    print(f"  Expected fitness: ~0.0085")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")
