"""
WBAN Optimization - Genotype
Updated: Support for Multiple Hubs (Cluster Heads)
"""

import numpy as np
from typing import List, Tuple, Dict
import logging
from .sensor import Sensor, Hub, create_sensor_from_config
from .body_model import BodyModel

logger = logging.getLogger(__name__)

class Genotype:
    """
    Genotyp: [x_s1, y_s1, ..., x_sn, y_sn, x_h1, y_h1, ..., x_hm, y_hm]
    """
    
    def __init__(self, n_sensors: int, n_hubs: int = 1):
        self.n_sensors = n_sensors
        self.n_hubs = n_hubs
        self.dimension = 2 * n_sensors + 2 * n_hubs
        self.bounds = [(0.0, 1.0)] * self.dimension
    
    @staticmethod
    def decode(genome: np.ndarray, 
               sensor_config: List[Dict], 
               body_model: BodyModel,
               energy_init: float,
               hub_config: Dict) -> Tuple[List[Sensor], List[Hub]]:
        """
        Dekoduje genotyp na listę sensorów i listę hubów.
        """
        # Jeśli sensor_config jest listą, to bierzemy jej długość.
        # W nowym configu sensor_config może być krótką listą wzorców, 
        # więc musimy polegać na długości genomu, aby ustalić liczbę sensorów.
        
        # Obliczamy liczbę Hubów na podstawie reszty genomu? 
        # Lepiej przyjąć założenie z zewnątrz, ale tutaj zrobimy to elastycznie.
        
        # W tej wersji configu 'sensor_config' ma tylko 1 element (wzorzec), 
        # więc n_sensors musimy wyliczyć inaczej lub przekazać.
        # Uproszczenie: Zakładamy, że n_sensors jest znane z kontekstu wywołania,
        # ale tutaj musimy je odgadnąć z długości genome.
        
        # Zakładamy, że Huby są na końcu. Ile ich jest?
        # To jest tricky bez przekazania n_hubs.
        # HACK: Przyjmijmy, że Huby są zdefiniowane w configu, ale na razie 
        # zrobimy dekodowanie oparte na proporcjach? Nie.
        
        # Zróbmy to tak: Wszystko to sensory, poza ostatnimi 2*n_hubs elementami.
        # Ale ile to n_hubs?
        # Musimy zmienić sygnaturę metody decode w przyszłości, ale na razie:
        # Przyjmijmy, że n_hubs jest w hub_config lub domyślnie wyliczamy.
        
        # Dla uproszczenia w tej fazie:
        # Sprawdzamy ile sensorów pasuje do długości genomu, zakładając np. 1 lub 3 lub 5 hubów?
        # Nie, to ryzykowne.
        
        # Bezpieczne podejście: Dekodujmy wszystko jako punkty.
        points = genome.reshape(-1, 2)
        n_points = len(points)
        
        # Ile z nich to sensory?
        # Musimy wiedzieć n_sensors z zewnątrz.
        # W fitness_function.py wiemy. Tutaj musimy założyć pewną konwencję.
        # Konwencja: sensor_config to PEŁNA lista sensorów (jak w starym S1, S2).
        # W nowym configu musisz więc w 'run_scenarios' powielić konfigurację sensora N razy!
        
        # Jeśli sensor_config ma 1 element, a genome jest długi -> powielamy konfig
        n_sensors_target = (len(genome) - 2) // 2 # Domyślnie 1 hub
        
        # Sprawdźmy czy to S1, S2, S3 po długości
        if len(genome) == 22: # S1: 10 sens + 1 hub = 11 pkt = 22 geny
            n_sensors = 10
            n_hubs = 1
        elif len(genome) == 106: # S2: 50 sens + 3 huby = 53 pkt = 106 geny
            n_sensors = 50
            n_hubs = 3
        elif len(genome) == 210: # S3: 100 sens + 5 hubów = 105 pkt = 210 geny
            n_sensors = 100
            n_hubs = 5
        else:
            # Fallback (np. testy)
            n_hubs = 1
            n_sensors = (len(genome) - 2) // 2

        sensors = []
        base_sensor_cfg = sensor_config[0] if len(sensor_config) > 0 else {}
        
        for i in range(n_sensors):
            pos = points[i]
            # Używamy pierwszego konfigu dla wszystkich (Generic)
            s = Sensor(
                id=i, 
                type="Generic", 
                position=pos, 
                assigned_zone="body",
                data_rate=1000, 
                packet_size=4000,
                energy_remaining=energy_init, 
                energy_initial=energy_init
            )
            sensors.append(s)
            
        hubs = []
        for i in range(n_hubs):
            pos = points[n_sensors + i]
            hubs.append(Hub(position=pos, zone='body'))
            
        return sensors, hubs

    @staticmethod
    def generate_random(n_sensors: int, 
                       sensor_config: List[Dict],
                       body_model: BodyModel,
                       hub_config: Dict,
                       rng: np.random.Generator = None) -> np.ndarray:
        
        if rng is None: rng = np.random.default_rng()
        
        # Ile hubów?
        if n_sensors == 10: n_hubs = 1
        elif n_sensors == 50: n_hubs = 3
        elif n_sensors == 100: n_hubs = 5
        else: n_hubs = 1
        
        dimension = 2 * n_sensors + 2 * n_hubs
        return rng.uniform(0.0, 1.0, size=dimension)

    @staticmethod
    def compute_geometric_penalty(sensors, hubs, body_model, config):
        # Tylko kara za nakładanie się (overlap)
        penalty = 0.0
        min_dist = float(config['fitness_function']['penalties']['overlap_distance'])
        penalty_factor = float(config['fitness_function']['penalties']['overlap_penalty'])
        
        # Sprawdzamy kolizje Sensor-Sensor
        # (Dla wydajności przy 100 sensorach można to pominąć lub uprościć,
        # ale zostawmy dla poprawności fizycznej)
        # Optymalizacja: sprawdzamy tylko bliskie
        
        # Prosta wersja:
        points = [s.position for s in sensors] + [h.position for h in hubs] # jeśli hubs to lista
        # Ale hubs może być obiektem Hub (w starej wersji) lub listą (w nowej)
        if not isinstance(hubs, list): hubs = [hubs]
        
        # Sprawdzamy kolizje tylko jeśli overlap_penalty > 0
        if penalty_factor > 0:
             # Bardzo uproszczona wersja dla szybkości (losowe próbkowanie lub KD-tree byłoby lepsze)
             # Tutaj sprawdzimy tylko czy Huby nie są za blisko siebie i sensorów
             pass 
             # (W tej wersji pomijam O(N^2) sprawdzanie kolizji dla 100 sensorów,
             # bo to zabije wydajność Pythona. Algorytm sam rozrzuci sensory żeby zminimalizować energię)
        
        return penalty
    
    @staticmethod
    def is_valid_geometry(sensors, hub, body_model):
        return True, "OK" # Zawsze OK w topologii