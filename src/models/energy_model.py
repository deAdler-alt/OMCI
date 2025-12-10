"""
WBAN Optimization - Energy Model
Updated for Topological Optimization (Clustering)
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class EnergyModel:
    """
    First Order Radio Model dla IoMT/WBAN.
    """
    
    def __init__(self, config: Dict):
        """
        Inicjalizacja modelu energii.
        """
        energy_config = config['energy_model']
        
        self.E_elec_TX = energy_config['E_elec_TX']  # J/bit
        self.E_elec_RX = energy_config['E_elec_RX']  # J/bit
        self.E_amp_fs = energy_config['E_amp_fs']    # J/bit/m²
        self.E_amp_mp = energy_config['E_amp_mp']    # J/bit/m⁴
        self.E_da = energy_config.get('E_da', 5e-9)  # J/bit/signal (agregacja)
        self.d_threshold = energy_config['d_threshold']  # m
        
        logger.info(f"Energy model initialized. d_threshold={self.d_threshold:.1f}m")
    
    def compute_transmission_energy(self, packet_size: int, distance: float) -> float:
        """
        Oblicza energię transmisji pakietu (TX).
        E_TX = E_elec*k + E_amp*k*d^n
        """
        if packet_size <= 0 or distance < 0:
            return 0.0
        
        k = packet_size
        d = distance
        
        # Energia elektroniki
        E_elec = self.E_elec_TX * k
        
        # Energia wzmacniacza
        if d < self.d_threshold:
            # Free Space model (d²)
            E_amp = self.E_amp_fs * k * (d ** 2)
        else:
            # Multipath model (d⁴)
            E_amp = self.E_amp_mp * k * (d ** 4)
        
        return E_elec + E_amp
    
    def compute_reception_energy(self, packet_size: int) -> float:
        """
        Oblicza energię odbioru pakietu (RX).
        E_RX = E_elec*k
        """
        return self.E_elec_RX * packet_size
    
    def compute_aggregation_energy(self, packet_size: int, n_signals: int) -> float:
        """
        Oblicza energię agregacji danych w Hubie.
        E_DA = E_da * n_signals * k
        """
        return self.E_da * n_signals * packet_size
        
    # Metody pomocnicze (zachowane dla kompatybilności)
    def compute_network_lifetime(self, energy_per_sensor, energy_per_round):
        rounds = energy_per_sensor / (energy_per_round + 1e-12) # unikaj dzielenia przez 0
        return int(np.min(rounds))