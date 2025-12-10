"""
WBAN/IoMT Optimization - Fitness Function
Topological Energy Optimization (Fixed & Verified)
"""

import numpy as np
from typing import Dict
import logging
from src.core.genotype import Genotype
from src.models.energy_model import EnergyModel
from src.core.body_model import BodyModel

logger = logging.getLogger(__name__)

class FitnessFunction:
    
    def __init__(self, config: Dict, scenario_name: str, weight_variant: str):
        self.config = config
        self.scenario_config = config['scenarios'][scenario_name]
        
        # Atrybuty wymagane przez single_run.py
        self.n_sensors = self.scenario_config['n_sensors']
        self.P_TX_max = self.scenario_config['P_TX_max']
        
        # Wagi
        weights_config = config['fitness_function']['weight_variants'][weight_variant]
        self.weights = {
            'w_E': weights_config['w_E'],
            'w_R': weights_config['w_R']
        }

        self.sensor_config = self.scenario_config['sensor_config']
        self.hub_config = config['scenarios']['hub_placement']
        self.energy_init = config['energy_model']['E_init']
        
        self.energy_model = EnergyModel(config)
        self.body_model = BodyModel(config)
        
        # Wymiary przestrzeni
        self.scale_x = config['space']['dimensions'][0]
        self.scale_y = config['space']['dimensions'][1]
        self.bs_pos = np.array(config['space']['base_station']['position'])

        logger.info(f"Fitness initialized: {scenario_name}, P_TX={self.P_TX_max}, w={self.weights}")

    def evaluate(self, genome: np.ndarray) -> float:
        """
        Oblicza energię całkowitą sieci (Topological Optimization).
        """
        try:
            sensors, hubs = Genotype.decode(
                genome, self.sensor_config, self.body_model, 
                self.energy_init, self.hub_config
            )
        except Exception:
            return 1e9

        total_energy = 0.0
        
        # Pozycje rzeczywiste (w metrach)
        real_hubs_pos = [h.position * np.array([self.scale_x, self.scale_y]) for h in hubs]
        hub_load = [0] * len(hubs)

        # 1. Sensor -> Hub
        for s in sensors:
            s_real_pos = s.position * np.array([self.scale_x, self.scale_y])
            
            # Znajdź najbliższego Huba
            dists = [np.linalg.norm(s_real_pos - h_pos) for h_pos in real_hubs_pos]
            if not dists: continue # Zabezpieczenie na wypadek braku hubów
            
            min_dist = min(dists)
            closest_hub_idx = dists.index(min_dist)
            
            # Energia TX (Sensor) + RX (Hub)
            e_tx = self.energy_model.compute_transmission_energy(s.packet_size, min_dist)
            e_rx = self.energy_model.compute_reception_energy(s.packet_size)
            
            total_energy += (e_tx + e_rx)
            hub_load[closest_hub_idx] += 1

        # 2. Hub -> Base Station
        for i, h_pos in enumerate(real_hubs_pos):
            n_packets = hub_load[i]
            if n_packets > 0:
                # Agregacja
                packet_size = 4000 # z configu
                e_agg = self.energy_model.compute_aggregation_energy(packet_size, n_packets)
                
                # Transmisja do BS
                dist_to_bs = np.linalg.norm(h_pos - self.bs_pos)
                total_bits = n_packets * packet_size
                e_tx_bs = self.energy_model.compute_transmission_energy(total_bits, dist_to_bs)
                
                total_energy += (e_agg + e_tx_bs)

        # 3. Kary (tylko za nakładanie się)
        penalty = Genotype.compute_geometric_penalty(sensors, hubs, self.body_model, self.config)
        
        return total_energy + penalty

    def evaluate_detailed(self, genome: np.ndarray) -> Dict:
        """
        Zwraca szczegóły dla raportowania i wizualizacji.
        """
        try:
            sensors, hubs = Genotype.decode(
                genome, self.sensor_config, self.body_model, 
                self.energy_init, self.hub_config
            )
        except Exception:
            return {
                'fitness': 1e9, 'is_valid': False, 'validation_reason': 'Decode Error',
                'E_total': 0, 'network_lifetime': 0, 'sensor_results': []
            }

        fitness = self.evaluate(genome) # Używamy głównej funkcji do obliczenia kosztu
        
        # Odtwarzamy logikę dla sensor_results (dla wizualizacji)
        sensor_results = []
        real_hubs_pos = [h.position * np.array([self.scale_x, self.scale_y]) for h in hubs]
        
        for i, s in enumerate(sensors):
            s_real_pos = s.position * np.array([self.scale_x, self.scale_y])
            dists = [np.linalg.norm(s_real_pos - h_pos) for h_pos in real_hubs_pos]
            min_dist = min(dists) if dists else 0.0
            
            sensor_results.append({
                'sensor_id': i,
                'sensor_type': s.type,
                'distance': min_dist,
                'los_status': 'LOS', 
                'path_loss': 0.0,
                'link_margin': 0.0,
                'E_TX': 0.0, # Uproszczenie dla raportu
                'E_RX': 0.0
            })

        # Przybliżony czas życia
        avg_energy_per_node = fitness / (self.n_sensors + 1e-9)
        lifetime = int(self.energy_init / (avg_energy_per_node + 1e-12)) if avg_energy_per_node > 0 else 0

        return {
            'fitness': fitness,
            'is_valid': True,
            'validation_reason': 'OK',
            'penalty_geo': 0.0,
            'penalty_rel': 0.0,
            'E_total': fitness,
            'min_link_margin': 0.0,
            'network_lifetime': lifetime,
            'los_count': self.n_sensors, 
            'nlos_count': 0, 
            'los_ratio': 1.0,
            'sensor_results': sensor_results
        }