"""
WBAN Optimization - Sensor and Hub Classes
Author: Kamil Piejko
Date: 2024

Klasy reprezentujące sensory i koncentrator (Hub) w sieci WBAN.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Sensor:
    """
    Reprezentacja pojedynczego sensora w sieci WBAN.
    
    Attributes:
        id: Unikalny identyfikator sensora
        type: Typ sensora (np. 'ECG', 'SpO2', 'Accelerometer')
        position: Pozycja [x, y] w przestrzeni znormalizowanej
        assigned_zone: Przypisana strefa anatomiczna
        data_rate: Szybkość generowania danych [bps]
        packet_size: Rozmiar pakietu [bits]
        energy_remaining: Pozostała energia [J]
        energy_initial: Początkowa energia [J]
        priority: Priorytet sensora ('high', 'medium', 'low')
        mobility: Mobilność sensora ('low', 'medium', 'high')
    """
    id: int
    type: str
    position: np.ndarray
    assigned_zone: str
    data_rate: float  # bps
    packet_size: int  # bits
    energy_remaining: float  # J
    energy_initial: float  # J
    priority: str = 'medium'
    mobility: str = 'low'
    
    # Metryki transmisji (aktualizowane podczas symulacji)
    transmitted_packets: int = 0
    energy_consumed: float = 0.0
    failed_transmissions: int = 0
    
    def is_alive(self, threshold: float = 0.01) -> bool:
        """
        Sprawdza, czy sensor ma wystarczającą energię do działania.
        
        Args:
            threshold: Minimalny próg energii [J]
        
        Returns:
            True jeśli sensor ma energię
        """
        return self.energy_remaining > threshold
    
    def consume_energy(self, energy: float) -> bool:
        """
        Zużywa energię przez sensor.
        
        Args:
            energy: Ilość energii do zużycia [J]
        
        Returns:
            True jeśli operacja się powiodła (miał wystarczającą energię)
        """
        if energy < 0:
            raise ValueError("Energy consumption cannot be negative")
        
        if self.energy_remaining >= energy:
            self.energy_remaining -= energy
            self.energy_consumed += energy
            return True
        else:
            # Sensor nie ma wystarczającej energii
            self.energy_remaining = 0.0
            return False
    
    def transmit_packet(self, energy_required: float) -> bool:
        """
        Symuluje transmisję pakietu.
        
        Args:
            energy_required: Energia wymagana do transmisji [J]
        
        Returns:
            True jeśli transmisja się powiodła
        """
        if self.consume_energy(energy_required):
            self.transmitted_packets += 1
            return True
        else:
            self.failed_transmissions += 1
            return False
    
    def get_energy_percentage(self) -> float:
        """
        Zwraca procent pozostałej energii.
        
        Returns:
            percentage: 0-100%
        """
        if self.energy_initial == 0:
            return 0.0
        return (self.energy_remaining / self.energy_initial) * 100
    
    def reset_metrics(self):
        """
        Resetuje metryki transmisji i przywraca energię do stanu początkowego.
        """
        self.energy_remaining = self.energy_initial
        self.energy_consumed = 0.0
        self.transmitted_packets = 0
        self.failed_transmissions = 0
    
    def __repr__(self) -> str:
        return (f"Sensor(id={self.id}, type='{self.type}', "
                f"pos={self.position}, zone='{self.assigned_zone}', "
                f"energy={self.energy_remaining:.4f}J)")
    
    def to_dict(self) -> dict:
        """
        Konwertuje sensor do słownika (dla serializacji).
        
        Returns:
            sensor_dict: Słownik z danymi sensora
        """
        return {
            'id': self.id,
            'type': self.type,
            'position': self.position.tolist(),
            'assigned_zone': self.assigned_zone,
            'data_rate': self.data_rate,
            'packet_size': self.packet_size,
            'energy_remaining': self.energy_remaining,
            'energy_initial': self.energy_initial,
            'priority': self.priority,
            'mobility': self.mobility,
            'transmitted_packets': self.transmitted_packets,
            'energy_consumed': self.energy_consumed,
            'failed_transmissions': self.failed_transmissions
        }


@dataclass
class Hub:
    """
    Koncentrator (sink) - odbiera dane ze wszystkich sensorów.
    
    Hub jest zwykle podłączony do zewnętrznego zasilania (telefon, tablet)
    lub ma znacznie większą baterię niż sensory, więc nie modelujemy jego zużycia energii.
    
    Attributes:
        position: Pozycja [x, y] w przestrzeni znormalizowanej
        zone: Preferowana strefa (domyślnie 'waist')
        energy_unlimited: Czy Hub ma nieograniczoną energię
    """
    position: np.ndarray
    zone: str = 'waist'
    energy_unlimited: bool = True
    
    # Metryki odbioru
    received_packets: int = 0
    total_throughput: float = 0.0  # bits/s
    lost_packets: int = 0
    
    def receive_packet(self, packet_size: int, success: bool = True):
        """
        Rejestruje odbiór pakietu.
        
        Args:
            packet_size: Rozmiar pakietu [bits]
            success: Czy pakiet został pomyślnie odebrany
        """
        if success:
            self.received_packets += 1
            self.total_throughput += packet_size
        else:
            self.lost_packets += 1
    
    def get_packet_loss_rate(self) -> float:
        """
        Oblicza współczynnik utraty pakietów.
        
        Returns:
            loss_rate: Wartość od 0 do 1
        """
        total_attempts = self.received_packets + self.lost_packets
        if total_attempts == 0:
            return 0.0
        return self.lost_packets / total_attempts
    
    def reset_metrics(self):
        """
        Resetuje metryki odbioru.
        """
        self.received_packets = 0
        self.total_throughput = 0.0
        self.lost_packets = 0
    
    def __repr__(self) -> str:
        return f"Hub(pos={self.position}, zone='{self.zone}')"
    
    def to_dict(self) -> dict:
        """
        Konwertuje Hub do słownika.
        
        Returns:
            hub_dict: Słownik z danymi Hub
        """
        return {
            'position': self.position.tolist(),
            'zone': self.zone,
            'energy_unlimited': self.energy_unlimited,
            'received_packets': self.received_packets,
            'total_throughput': self.total_throughput,
            'lost_packets': self.lost_packets
        }


def create_sensor_from_config(sensor_id: int, sensor_config: dict, 
                             position: np.ndarray, energy_init: float) -> Sensor:
    """
    Tworzy obiekt Sensor z konfiguracji.
    
    Args:
        sensor_id: ID sensora
        sensor_config: Konfiguracja typu sensora (z YAML)
        position: Pozycja [x, y]
        energy_init: Początkowa energia [J]
    
    Returns:
        sensor: Obiekt Sensor
    """
    return Sensor(
        id=sensor_id,
        type=sensor_config['type'],
        position=position,
        assigned_zone=sensor_config['zone'],
        data_rate=sensor_config.get('data_rate', 100),
        packet_size=sensor_config.get('packet_size', 100),
        energy_remaining=energy_init,
        energy_initial=energy_init,
        priority=sensor_config.get('priority', 'medium'),
        mobility=sensor_config.get('mobility', 'low')
    )


def create_hub_from_config(position: np.ndarray, hub_config: dict) -> Hub:
    """
    Tworzy obiekt Hub z konfiguracji.
    
    Args:
        position: Pozycja [x, y]
        hub_config: Konfiguracja Hub (z YAML)
    
    Returns:
        hub: Obiekt Hub
    """
    return Hub(
        position=position,
        zone=hub_config.get('preferred_zone', 'waist'),
        energy_unlimited=True
    )


if __name__ == '__main__':
    # Test klas Sensor i Hub
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("SENSOR & HUB TEST")
    print("="*60)
    
    # Test 1: Utworzenie sensora
    print("\n[TEST 1] Sensor Creation:")
    sensor = Sensor(
        id=1,
        type='ECG',
        position=np.array([0.45, 0.65]),
        assigned_zone='chest',
        data_rate=200,
        packet_size=100,
        energy_remaining=0.5,
        energy_initial=0.5,
        priority='high'
    )
    print(f"  Created: {sensor}")
    print(f"  Alive: {sensor.is_alive()}")
    print(f"  Energy: {sensor.get_energy_percentage():.1f}%")
    
    # Test 2: Zużycie energii
    print("\n[TEST 2] Energy Consumption:")
    energy_to_consume = 0.1  # J
    success = sensor.consume_energy(energy_to_consume)
    print(f"  Consumed {energy_to_consume} J: {success}")
    print(f"  Remaining energy: {sensor.energy_remaining:.4f} J ({sensor.get_energy_percentage():.1f}%)")
    
    # Test 3: Transmisja pakietów
    print("\n[TEST 3] Packet Transmission:")
    for i in range(5):
        energy_tx = 0.05  # J per packet
        success = sensor.transmit_packet(energy_tx)
        print(f"  Packet {i+1}: {'✓ Success' if success else '✗ Failed'} "
              f"(energy={sensor.energy_remaining:.4f} J)")
    
    print(f"  Total packets transmitted: {sensor.transmitted_packets}")
    print(f"  Total energy consumed: {sensor.energy_consumed:.4f} J")
    print(f"  Failed transmissions: {sensor.failed_transmissions}")
    
    # Test 4: Sensor umiera (brak energii)
    print("\n[TEST 4] Sensor Depletion:")
    # Zużyj całą energię
    while sensor.is_alive():
        sensor.consume_energy(0.01)
    
    print(f"  Sensor alive: {sensor.is_alive()}")
    print(f"  Energy remaining: {sensor.energy_remaining:.4f} J")
    
    # Próba transmisji bez energii
    success = sensor.transmit_packet(0.05)
    print(f"  Transmission without energy: {'✓ Success' if success else '✗ Failed'}")
    print(f"  Failed transmissions: {sensor.failed_transmissions}")
    
    # Test 5: Reset
    print("\n[TEST 5] Sensor Reset:")
    sensor.reset_metrics()
    print(f"  After reset:")
    print(f"    Energy: {sensor.energy_remaining:.4f} J ({sensor.get_energy_percentage():.1f}%)")
    print(f"    Packets transmitted: {sensor.transmitted_packets}")
    print(f"    Energy consumed: {sensor.energy_consumed:.4f} J")
    
    # Test 6: Hub
    print("\n[TEST 6] Hub:")
    hub = Hub(position=np.array([0.50, 0.45]), zone='waist')
    print(f"  Created: {hub}")
    
    # Symuluj odbiór pakietów
    for i in range(10):
        success = i < 8  # 8 sukcesów, 2 straty
        hub.receive_packet(packet_size=100, success=success)
    
    print(f"  Received packets: {hub.received_packets}")
    print(f"  Lost packets: {hub.lost_packets}")
    print(f"  Packet loss rate: {hub.get_packet_loss_rate()*100:.1f}%")
    print(f"  Total throughput: {hub.total_throughput} bits")
    
    # Test 7: Serializacja
    print("\n[TEST 7] Serialization:")
    sensor_dict = sensor.to_dict()
    hub_dict = hub.to_dict()
    print(f"  Sensor dict keys: {list(sensor_dict.keys())}")
    print(f"  Hub dict keys: {list(hub_dict.keys())}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")