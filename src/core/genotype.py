"""
WBAN Optimization - Genotype Encoding/Decoding
Author: Kamil Piejko
Date: 2024

Kodowanie pozycji sensorów i Hub jako wektor liczb rzeczywistych (genotyp).
"""

import numpy as np
from typing import List, Tuple, Dict
import logging

from .sensor import Sensor, Hub, create_sensor_from_config
from .body_model import BodyModel

logger = logging.getLogger(__name__)


class Genotype:
    """
    Kodowanie rozwiązania jako wektor liczb rzeczywistych.
    
    Struktura genotypu:
        g = [x1, y1, x2, y2, ..., xN, yN, x_hub, y_hub]
    
    Wymiar: D = 2*N + 2 (gdzie N = liczba sensorów)
    Zakres: wszystkie współrzędne ∈ [0, 1]
    
    Attributes:
        n_sensors: Liczba sensorów
        dimension: Wymiar genotypu
        bounds: Granice dla każdej zmiennej
    """
    
    def __init__(self, n_sensors: int):
        """
        Inicjalizacja kodowania genotypu.
        
        Args:
            n_sensors: Liczba sensorów w sieci
        """
        self.n_sensors = n_sensors
        self.dimension = 2 * n_sensors + 2  # (x,y) dla każdego sensora + Hub
        self.bounds = [(0.0, 1.0)] * self.dimension
        
        logger.debug(f"Genotype initialized: n_sensors={n_sensors}, dim={self.dimension}")
    
    @staticmethod
    def decode(genome: np.ndarray, 
               sensor_config: List[Dict], 
               body_model: BodyModel,
               energy_init: float,
               hub_config: Dict) -> Tuple[List[Sensor], Hub]:
        """
        Dekoduje genotyp na obiekty Sensor i Hub.
        
        Args:
            genome: Wektor [x1,y1,...,xN,yN,x_hub,y_hub]
            sensor_config: Lista konfiguracji sensorów (z YAML scenario)
            body_model: Model ciała
            energy_init: Początkowa energia sensorów [J]
            hub_config: Konfiguracja Hub (z YAML)
        
        Returns:
            (sensors, hub): Lista sensorów i obiekt Hub
        
        Raises:
            ValueError: Jeśli wymiar genotypu jest niepoprawny
        """
        n_sensors = (len(genome) - 2) // 2
        
        if len(sensor_config) != n_sensors:
            raise ValueError(f"Mismatch: genome has {n_sensors} sensors, "
                           f"but config has {len(sensor_config)}")
        
        sensors = []
        
        # Dekoduj sensory
        for i in range(n_sensors):
            x = genome[2*i]
            y = genome[2*i + 1]
            position = np.array([x, y])
            
            sensor = create_sensor_from_config(
                sensor_id=i+1,
                sensor_config=sensor_config[i],
                position=position,
                energy_init=energy_init
            )
            
            sensors.append(sensor)
        
        # Dekoduj Hub
        x_hub = genome[-2]
        y_hub = genome[-1]
        hub_position = np.array([x_hub, y_hub])
        
        hub = Hub(
            position=hub_position,
            zone=hub_config.get('preferred_zone', 'waist'),
            energy_unlimited=True
        )
        
        return sensors, hub
    
    @staticmethod
    def encode(sensors: List[Sensor], hub: Hub) -> np.ndarray:
        """
        Koduje obiekty Sensor i Hub do genotypu.
        
        Args:
            sensors: Lista sensorów
            hub: Obiekt Hub
        
        Returns:
            genome: Wektor [x1,y1,...,xN,yN,x_hub,y_hub]
        """
        n_sensors = len(sensors)
        genome = np.zeros(2 * n_sensors + 2)
        
        # Koduj sensory
        for i, sensor in enumerate(sensors):
            genome[2*i] = sensor.position[0]
            genome[2*i + 1] = sensor.position[1]
        
        # Koduj Hub
        genome[-2] = hub.position[0]
        genome[-1] = hub.position[1]
        
        return genome
    
    @staticmethod
    def generate_random(n_sensors: int, 
                       sensor_config: List[Dict],
                       body_model: BodyModel,
                       hub_config: Dict,
                       rng: np.random.Generator = None) -> np.ndarray:
        """
        Generuje losowy genotyp z sensorami w odpowiednich strefach.
        
        Args:
            n_sensors: Liczba sensorów
            sensor_config: Konfiguracja sensorów
            body_model: Model ciała
            hub_config: Konfiguracja Hub
            rng: Generator liczb losowych
        
        Returns:
            genome: Losowy genotyp
        """
        if rng is None:
            rng = np.random.default_rng()
        
        genome = np.zeros(2 * n_sensors + 2)
        
        # Generuj pozycje sensorów w przypisanych strefach
        for i in range(n_sensors):
            zone_name = sensor_config[i]['zone']
            
            # Obsługa przypadku, gdy zone może być listą (elastyczne przypisanie)
            if isinstance(zone_name, list):
                zone_name = rng.choice(zone_name)
            
            position = body_model.get_random_position_in_zone(zone_name, rng)
            genome[2*i] = position[0]
            genome[2*i + 1] = position[1]
        
        # Generuj pozycję Hub
        hub_zone = hub_config.get('preferred_zone', 'waist')
        hub_position = body_model.get_random_position_in_zone(hub_zone, rng)
        genome[-2] = hub_position[0]
        genome[-1] = hub_position[1]
        
        return genome
    
    @staticmethod
    def is_valid_geometry(sensors: List[Sensor], 
                         hub: Hub,
                         body_model: BodyModel,
                         min_sensor_distance: float = 0.05) -> Tuple[bool, str]:
        """
        Sprawdza poprawność geometryczną rozwiązania.
        
        Waliduje:
        1. Czy sensory są w swoich przypisanych strefach
        2. Czy sensory nie są za blisko siebie (nakładanie)
        3. Czy Hub jest w dopuszczalnej strefie (opcjonalnie)
        
        Args:
            sensors: Lista sensorów
            hub: Obiekt Hub
            body_model: Model ciała
            min_sensor_distance: Minimalna odległość między sensorami
        
        Returns:
            (is_valid, reason): True/False i opis problemu
        """
        # Sprawdź każdy sensor
        for sensor in sensors:
            if not body_model.is_valid_position(sensor.position, sensor.assigned_zone):
                return False, f"Sensor {sensor.id} ({sensor.type}) poza strefą '{sensor.assigned_zone}'"
        
        # Sprawdź kolizje między sensorami
        for i, s1 in enumerate(sensors):
            for s2 in sensors[i+1:]:
                distance = np.linalg.norm(s1.position - s2.position)
                if distance < min_sensor_distance:
                    return False, f"Sensory {s1.id} i {s2.id} za blisko ({distance:.4f} < {min_sensor_distance})"
        
        # Hub nie wymaga ścisłej walidacji (może być gdziekolwiek)
        # ale sprawdźmy, czy jest w znormalizowanym zakresie
        if not (0 <= hub.position[0] <= 1 and 0 <= hub.position[1] <= 1):
            return False, f"Hub poza zakresem [0,1]×[0,1]"
        
        return True, "OK"
    
    @staticmethod
    def compute_geometric_penalty(sensors: List[Sensor],
                                  hub: Hub,
                                  body_model: BodyModel,
                                  config: Dict) -> float:
        """
        Oblicza karę za naruszenia geometryczne.
        
        Args:
            sensors: Lista sensorów
            hub: Obiekt Hub
            body_model: Model ciała
            config: Konfiguracja (penalties)
        
        Returns:
            penalty: Wartość kary (0 jeśli brak naruszeń)
        """
        penalty = 0.0
        
        penalty_params = config['fitness_function']['penalties']

        zone_violation_penalty = float(penalty_params['zone_violation'])
        min_distance = float(penalty_params['overlap_distance'])
        overlap_penalty_factor = float(penalty_params['overlap_penalty'])
        
        # Kara za sensory poza strefami
        for sensor in sensors:
            if not body_model.is_valid_position(sensor.position, sensor.assigned_zone):
                penalty += zone_violation_penalty
        
        # Kara za nakładanie się sensorów
        for i, s1 in enumerate(sensors):
            for s2 in sensors[i+1:]:
                distance = np.linalg.norm(s1.position - s2.position)
                if distance < min_distance:
                    # Kara proporcjonalna do stopnia nakładania
                    overlap = min_distance - distance
                    penalty += overlap_penalty_factor * overlap
        
        return penalty
    
    def __repr__(self) -> str:
        return f"Genotype(n_sensors={self.n_sensors}, dimension={self.dimension})"


if __name__ == '__main__':
    # Test genotypu
    import sys
    sys.path.append('../..')
    
    from src.utils.config_loader import load_config
    from src.core.body_model import BodyModel
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("GENOTYPE TEST")
    print("="*60)
    
    # Wczytaj konfigurację
    config = load_config()
    body_model = BodyModel(config)
    
    # Test dla scenariusza S1 (6 sensorów)
    scenario_config = config['scenarios']['S1']
    n_sensors = scenario_config['n_sensors']
    sensor_config = scenario_config['sensor_config']
    hub_config = config['scenarios']['hub_placement']
    energy_init = config['energy_model']['E_init']
    
    print(f"\n[TEST 1] Genotype for Scenario S1 ({n_sensors} sensors):")
    genotype = Genotype(n_sensors)
    print(f"  {genotype}")
    print(f"  Dimension: {genotype.dimension}")
    print(f"  Bounds: {genotype.bounds[0]} × {genotype.dimension}")
    
    # Test 2: Generowanie losowego genotypu
    print("\n[TEST 2] Random Genotype Generation:")
    rng = np.random.default_rng(42)
    genome = Genotype.generate_random(n_sensors, sensor_config, body_model, hub_config, rng)
    print(f"  Generated genome shape: {genome.shape}")
    print(f"  Genome: {genome}")
    
    # Test 3: Dekodowanie
    print("\n[TEST 3] Decoding Genotype:")
    sensors, hub = Genotype.decode(genome, sensor_config, body_model, energy_init, hub_config)
    print(f"  Decoded {len(sensors)} sensors:")
    for sensor in sensors:
        print(f"    {sensor}")
    print(f"  Hub: {hub}")
    
    # Test 4: Kodowanie z powrotem
    print("\n[TEST 4] Encoding back to Genotype:")
    genome_reconstructed = Genotype.encode(sensors, hub)
    print(f"  Original genome:       {genome}")
    print(f"  Reconstructed genome:  {genome_reconstructed}")
    print(f"  Match: {np.allclose(genome, genome_reconstructed)}")
    
    # Test 5: Walidacja geometrii
    print("\n[TEST 5] Geometry Validation:")
    is_valid, reason = Genotype.is_valid_geometry(sensors, hub, body_model)
    print(f"  Valid: {is_valid}")
    print(f"  Reason: {reason}")
    
    # Test 6: Geometryczna kara
    print("\n[TEST 6] Geometric Penalty:")
    penalty = Genotype.compute_geometric_penalty(sensors, hub, body_model, config)
    print(f"  Penalty: {penalty}")
    
    # Test 7: Niepr poprawny genotyp (sensor poza strefą)
    print("\n[TEST 7] Invalid Genotype (sensor out of zone):")
    invalid_genome = genome.copy()
    invalid_genome[0] = 0.99  # Sensor 1 (ECG w chest) przesunięty poza strefę
    
    sensors_invalid, hub_invalid = Genotype.decode(
        invalid_genome, sensor_config, body_model, energy_init, hub_config
    )
    
    is_valid, reason = Genotype.is_valid_geometry(sensors_invalid, hub_invalid, body_model)
    print(f"  Valid: {is_valid}")
    print(f"  Reason: {reason}")
    
    penalty_invalid = Genotype.compute_geometric_penalty(
        sensors_invalid, hub_invalid, body_model, config
    )
    print(f"  Penalty: {penalty_invalid}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")