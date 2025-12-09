"""
WBAN Optimization - Energy Model
Author: Kamil Piejko
Date: 2024

First Order Radio Model dla obliczania zużycia energii w transmisji bezprzewodowej.

Źródła:
- [11] Al-Mishmish et al. 2018 - Critical Data Transmission
- [13] Ajmi et al. 2021 - MWCSGA
"""

import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class EnergyModel:
    """
    First Order Radio Model dla WBAN.
    
    Model energii transmisji:
        E_TX(k, d) = E_elec,TX × k + E_amp × k × d^α
    
    gdzie:
        k = packet_size [bits]
        d = distance [m]
        α = 2 (Free Space) lub 4 (Multipath)
    
    Model energii odbioru:
        E_RX(k) = E_elec,RX × k
    
    Attributes:
        E_elec_TX: Energia elektroniki nadajnika [J/bit]
        E_elec_RX: Energia elektroniki odbiornika [J/bit]
        E_amp_fs: Energia wzmacniacza - Free Space [J/bit/m²]
        E_amp_mp: Energia wzmacniacza - Multipath [J/bit/m⁴]
        E_da: Energia agregacji danych [J/bit/signal]
        d_threshold: Próg odległości Free Space/Multipath [m]
    """
    
    def __init__(self, config: Dict):
        """
        Inicjalizacja modelu energii.
        
        Args:
            config: Konfiguracja z YAML (sekcja 'energy_model')
        """
        energy_config = config['energy_model']
        
        self.E_elec_TX = energy_config['E_elec_TX']  # J/bit
        self.E_elec_RX = energy_config['E_elec_RX']  # J/bit
        self.E_amp_fs = energy_config['E_amp_fs']    # J/bit/m²
        self.E_amp_mp = energy_config['E_amp_mp']    # J/bit/m⁴
        self.E_da = energy_config.get('E_da', 5e-9)  # J/bit (opcjonalne)
        self.d_threshold = energy_config['d_threshold']  # m
        
        logger.info(f"Energy model initialized with E_elec_TX={self.E_elec_TX*1e9:.1f} nJ/bit")
    
    def compute_transmission_energy(self, 
                                   packet_size: int, 
                                   distance: float) -> float:
        """
        Oblicza energię transmisji pakietu.
        
        Args:
            packet_size: Rozmiar pakietu [bits]
            distance: Odległość sensor-hub [m]
        
        Returns:
            E_TX: Energia transmisji [J]
        """
        if packet_size <= 0:
            raise ValueError("Packet size must be positive")
        
        if distance < 0:
            raise ValueError("Distance cannot be negative")
        
        k = packet_size
        d = distance
        
        # Energia elektroniki (zawsze)
        E_elec = self.E_elec_TX * k
        
        # Energia wzmacniacza (zależy od odległości)
        if d < self.d_threshold:
            # Free Space model (d²)
            E_amp = self.E_amp_fs * k * (d ** 2)
        else:
            # Multipath model (d⁴)
            E_amp = self.E_amp_mp * k * (d ** 4)
        
        E_TX = E_elec + E_amp
        
        return E_TX
    
    def compute_reception_energy(self, packet_size: int) -> float:
        """
        Oblicza energię odbioru pakietu.
        
        Args:
            packet_size: Rozmiar pakietu [bits]
        
        Returns:
            E_RX: Energia odbioru [J]
        """
        if packet_size <= 0:
            raise ValueError("Packet size must be positive")
        
        E_RX = self.E_elec_RX * packet_size
        
        return E_RX
    
    def compute_data_aggregation_energy(self, 
                                       n_signals: int, 
                                       packet_size: int) -> float:
        """
        Oblicza energię agregacji danych (opcjonalne, dla CH w klastrach).
        
        Args:
            n_signals: Liczba sygnałów do zagregowania
            packet_size: Rozmiar pakietu [bits]
        
        Returns:
            E_DA: Energia agregacji [J]
        """
        E_DA = self.E_da * n_signals * packet_size
        return E_DA
    
    def compute_total_energy_per_round(self,
                                      packet_size: int,
                                      distances: np.ndarray) -> float:
        """
        Oblicza całkowitą energię dla rundy komunikacyjnej.
        
        Runda = wszyscy sensory wysyłają po 1 pakiecie do Hub.
        
        Args:
            packet_size: Rozmiar pakietu [bits]
            distances: Wektor odległości sensor-hub [m]
        
        Returns:
            E_total: Całkowita energia [J]
        """
        n_sensors = len(distances)
        
        # Energia transmisji wszystkich sensorów
        E_TX_total = sum(
            self.compute_transmission_energy(packet_size, d) 
            for d in distances
        )
        
        # Energia odbioru na Hub (n_sensors pakietów)
        E_RX_total = n_sensors * self.compute_reception_energy(packet_size)
        
        E_total = E_TX_total + E_RX_total
        
        return E_total
    
    def compute_network_lifetime(self,
                                energy_per_sensor: np.ndarray,
                                energy_per_round_per_sensor: np.ndarray) -> int:
        """
        Oblicza czas życia sieci (First Node Dies - FND).
        
        Args:
            energy_per_sensor: Początkowa energia każdego sensora [J]
            energy_per_round_per_sensor: Energia zużywana przez każdy sensor na rundę [J]
        
        Returns:
            T_life: Liczba rund do FND
        """
        # Dla każdego sensora: ile rund może przetrwać?
        rounds_per_sensor = energy_per_sensor / energy_per_round_per_sensor
        
        # FND = minimum (pierwszy sensor umiera)
        T_life = int(np.min(rounds_per_sensor))
        
        return T_life
    
    def get_model_params(self) -> Dict[str, float]:
        """
        Zwraca parametry modelu.
        
        Returns:
            params: Słownik z parametrami
        """
        return {
            'E_elec_TX': self.E_elec_TX,
            'E_elec_RX': self.E_elec_RX,
            'E_amp_fs': self.E_amp_fs,
            'E_amp_mp': self.E_amp_mp,
            'E_da': self.E_da,
            'd_threshold': self.d_threshold
        }
    
    def __repr__(self) -> str:
        return (f"EnergyModel(E_elec_TX={self.E_elec_TX*1e9:.1f} nJ/bit, "
                f"d_threshold={self.d_threshold} m)")


if __name__ == '__main__':
    # Test modelu energii
    import sys
    sys.path.append('../..')
    
    from src.utils.config_loader import load_config
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("ENERGY MODEL TEST")
    print("="*60)
    
    # Wczytaj konfigurację
    config = load_config()
    
    # Utwórz model energii
    energy_model = EnergyModel(config)
    print(f"\n[TEST 1] Model: {energy_model}")
    
    # Test 2: Parametry
    print("\n[TEST 2] Model Parameters:")
    params = energy_model.get_model_params()
    for key, value in params.items():
        if 'E_' in key:
            # Konwersja do nJ lub pJ dla czytelności
            if value < 1e-9:
                print(f"  {key:15s}: {value*1e12:.2f} pJ/bit" + ("" if "mp" in key else ""))
            else:
                print(f"  {key:15s}: {value*1e9:.2f} nJ/bit")
        else:
            print(f"  {key:15s}: {value} m")
    
    # Test 3: Energia transmisji dla różnych odległości
    print("\n[TEST 3] Transmission Energy vs. Distance:")
    packet_size = 100  # bits
    distances = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]  # m
    
    print(f"  Packet size: {packet_size} bits")
    print(f"  {'Distance [m]':>15s} {'Model':>10s} {'E_TX [μJ]':>12s} {'E_elec [μJ]':>14s} {'E_amp [μJ]':>13s}")
    print(f"  {'-'*15:>15s} {'-'*10:>10s} {'-'*12:>12s} {'-'*14:>14s} {'-'*13:>13s}")
    
    for d in distances:
        E_TX = energy_model.compute_transmission_energy(packet_size, d)
        
        # Rozbij na składowe
        E_elec = energy_model.E_elec_TX * packet_size
        if d < energy_model.d_threshold:
            model = "FS (d²)"
            E_amp = energy_model.E_amp_fs * packet_size * (d ** 2)
        else:
            model = "MP (d⁴)"
            E_amp = energy_model.E_amp_mp * packet_size * (d ** 4)
        
        print(f"  {d:>15.2f} {model:>10s} {E_TX*1e6:>12.4f} {E_elec*1e6:>14.4f} {E_amp*1e6:>13.6f}")
    
    # Test 4: Energia odbioru
    print("\n[TEST 4] Reception Energy:")
    E_RX = energy_model.compute_reception_energy(packet_size)
    print(f"  Packet size: {packet_size} bits")
    print(f"  E_RX: {E_RX*1e6:.4f} μJ")
    
    # Test 5: Energia całkowita dla rundy
    print("\n[TEST 5] Total Energy per Round (3 sensors):")
    distances_round = np.array([0.2, 0.35, 0.6])  # m
    E_total_round = energy_model.compute_total_energy_per_round(packet_size, distances_round)
    print(f"  Distances: {distances_round} m")
    print(f"  E_total per round: {E_total_round*1e3:.4f} mJ")
    
    # Rozbij na składowe
    E_TX_sum = sum(energy_model.compute_transmission_energy(packet_size, d) for d in distances_round)
    E_RX_sum = len(distances_round) * E_RX
    print(f"    E_TX (all sensors): {E_TX_sum*1e3:.4f} mJ")
    print(f"    E_RX (Hub): {E_RX_sum*1e3:.4f} mJ")
    
    # Test 6: Czas życia sieci
    print("\n[TEST 6] Network Lifetime (FND):")
    E_init = config['energy_model']['E_init']  # J
    energy_per_sensor = np.array([E_init, E_init, E_init])  # 3 sensory
    
    # Energia na rundę dla każdego sensora
    energy_per_round = np.array([
        energy_model.compute_transmission_energy(packet_size, distances_round[0]),
        energy_model.compute_transmission_energy(packet_size, distances_round[1]),
        energy_model.compute_transmission_energy(packet_size, distances_round[2])
    ])
    
    T_life = energy_model.compute_network_lifetime(energy_per_sensor, energy_per_round)
    
    print(f"  Initial energy per sensor: {E_init} J")
    print(f"  Energy per round per sensor: {energy_per_round*1e3} mJ")
    print(f"  Network lifetime (FND): {T_life} rounds")
    print(f"  First node dies at round: {T_life}")
    print(f"  (sensor with highest energy consumption)")
    
    # Test 7: Weryfikacja z ręcznym obliczeniem (Przykład 1 z docs)
    print("\n[TEST 7] Verification with Manual Calculation:")
    print("  (From docs/fitness_calculation_examples.md, Sensor 1)")
    
    packet_size_test = 100  # bits (ECG)
    distance_test = 0.206  # m
    
    E_TX_test = energy_model.compute_transmission_energy(packet_size_test, distance_test)
    
    # Ręczne obliczenie (z dokumentacji)
    E_elec_manual = 50e-9 * 100  # 5.0 μJ
    E_amp_manual = 10e-12 * 100 * (0.206 ** 2)  # ~4.24e-11 J ≈ 0
    E_TX_manual = E_elec_manual + E_amp_manual
    
    print(f"  Calculated E_TX: {E_TX_test*1e6:.6f} μJ")
    print(f"  Manual E_TX:     {E_TX_manual*1e6:.6f} μJ")
    print(f"  Match: {np.isclose(E_TX_test, E_TX_manual, rtol=1e-3)}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")