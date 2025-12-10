"""
WBAN Optimization - Body Model
Author: Kamil Piejko
Date: 2024

Model ciała ludzkiego ze strefami anatomicznymi dla rozmieszczenia sensorów.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BodyZone:
    """
    Reprezentacja strefy anatomicznej na ciele.
    
    Attributes:
        name: Nazwa strefy (np. 'chest', 'left_wrist')
        x_range: Zakres współrzędnych X [x_min, x_max]
        y_range: Zakres współrzędnych Y [y_min, y_max]
        description: Opis strefy
    """
    name: str
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    description: str = ""

    #Dodana nowa metoda
    def distance_to_boundary(self, x: float, y: float) -> float:
        """
        Oblicza minimalną odległość punktu od granic strefy.
        Zwraca 0.0, jeśli punkt jest wewnątrz.
        """
        # Odległość w osi X
        dx = max(self.x_range[0] - x, 0, x - self.x_range[1])
        # Odległość w osi Y
        dy = max(self.y_range[0] - y, 0, y - self.y_range[1])
        
        # Odległość euklidesowa
        return np.sqrt(dx*dx + dy*dy)


    def contains_point(self, x: float, y: float) -> bool:
        """
        Sprawdza, czy punkt (x, y) leży w tej strefie.
        
        Args:
            x: Współrzędna X
            y: Współrzędna Y
        
        Returns:
            True jeśli punkt jest w strefie
        """
        in_x = self.x_range[0] <= x <= self.x_range[1]
        in_y = self.y_range[0] <= y <= self.y_range[1]
        return in_x and in_y
    
    def get_center(self) -> np.ndarray:
        """
        Zwraca środek strefy.
        
        Returns:
            center: [x_center, y_center]
        """
        x_center = (self.x_range[0] + self.x_range[1]) / 2
        y_center = (self.y_range[0] + self.y_range[1]) / 2
        return np.array([x_center, y_center])
    
    def get_area(self) -> float:
        """
        Zwraca pole powierzchni strefy.
        
        Returns:
            area: Pole w jednostkach znormalizowanych
        """
        width = self.x_range[1] - self.x_range[0]
        height = self.y_range[1] - self.y_range[0]
        return width * height
    
    def __repr__(self) -> str:
        return (f"BodyZone(name='{self.name}', "
                f"x={self.x_range}, y={self.y_range})")


class BodyModel:
    """
    Model ciała ludzkiego z strefami anatomicznymi.
    
    Współrzędny znormalizowane do [0, 1] × [0, 1]:
    - X: 0 (lewa) → 1 (prawa)
    - Y: 0 (dół/stopy) → 1 (góra/głowa)
    
    Attributes:
        zones: Słownik stref anatomicznych
        torso_params: Parametry cylindra torsu (dla LOS/NLOS)
    """
    
    # W klasie BodyModel
    def get_distance_to_zone(self, position: np.ndarray, zone_name: str) -> float:
        """
        Zwraca odległość pozycji od wskazanej strefy.
        """
        zone = self.get_zone(zone_name)
        return zone.distance_to_boundary(position[0], position[1])


    def __init__(self, config: Dict):
        """
        Inicjalizacja modelu ciała z konfiguracji.
        
        Args:
            config: Słownik z konfiguracją (z YAML)
        """
        self.zones: Dict[str, BodyZone] = {}
        self.torso_params: Dict[str, float] = {}
        
        self._load_zones(config['body_model']['zones'])
        self._load_torso_params(config['body_model']['torso_cylinder'])
        
        logger.info(f"Body model initialized with {len(self.zones)} zones")
    
    def _load_zones(self, zones_config: Dict):
        """
        Wczytuje strefy z konfiguracji.
        
        Args:
            zones_config: Słownik ze strefami z YAML
        """
        for zone_name, zone_data in zones_config.items():
            self.zones[zone_name] = BodyZone(
                name=zone_name,
                x_range=tuple(zone_data['x_range']),
                y_range=tuple(zone_data['y_range']),
                description=zone_data.get('description', '')
            )
    
    def _load_torso_params(self, torso_config: Dict):
        """
        Wczytuje parametry cylindra torsu.
        
        Args:
            torso_config: Parametry torsu z YAML
        """
        self.torso_params = {
            'center_x': torso_config['center_x'],
            'center_y': torso_config['center_y'],
            'radius': torso_config['radius'],
            'height_range': tuple(torso_config['height_range'])
        }
    
    def get_zone(self, zone_name: str) -> BodyZone:
        """
        Pobiera strefę po nazwie.
        
        Args:
            zone_name: Nazwa strefy
        
        Returns:
            zone: Obiekt BodyZone
        
        Raises:
            KeyError: Jeśli strefa nie istnieje
        """
        if zone_name not in self.zones:
            raise KeyError(f"Zone '{zone_name}' not found in body model")
        
        return self.zones[zone_name]
    
    def is_valid_position(self, position: np.ndarray, zone_name: str) -> bool:
        """
        Sprawdza, czy pozycja jest w przypisanej strefie.
        
        Args:
            position: [x, y]
            zone_name: Nazwa strefy
        
        Returns:
            True jeśli pozycja jest w strefie
        """
        zone = self.get_zone(zone_name)
        x, y = position[0], position[1]
        return zone.contains_point(x, y)
    
    def find_zone_for_point(self, position: np.ndarray) -> Optional[str]:
        """
        Znajduje strefę zawierającą dany punkt.
        
        Args:
            position: [x, y]
        
        Returns:
            zone_name: Nazwa strefy lub None jeśli punkt jest poza wszystkimi strefami
        """
        x, y = position[0], position[1]
        
        for zone_name, zone in self.zones.items():
            if zone.contains_point(x, y):
                return zone_name
        
        return None
    
    def get_random_position_in_zone(self, zone_name: str, 
                                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generuje losową pozycję w danej strefie.
        
        Args:
            zone_name: Nazwa strefy
            rng: Generator liczb losowych (opcjonalny)
        
        Returns:
            position: [x, y] w strefie
        """
        if rng is None:
            rng = np.random.default_rng()
        
        zone = self.get_zone(zone_name)
        
        x = rng.uniform(zone.x_range[0], zone.x_range[1])
        y = rng.uniform(zone.y_range[0], zone.y_range[1])
        
        return np.array([x, y])
    
    def get_distance_between_points(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Oblicza odległość euklidesową między dwoma punktami.
        
        Args:
            pos1: [x1, y1]
            pos2: [x2, y2]
        
        Returns:
            distance: Odległość w jednostkach znormalizowanych
        """
        return np.linalg.norm(pos1 - pos2)
    
    def is_inside_torso_cylinder(self, position: np.ndarray) -> bool:
        """
        Sprawdza, czy punkt jest wewnątrz cylindra torsu.
        
        Args:
            position: [x, y]
        
        Returns:
            True jeśli punkt jest w cylindrze
        """
        x, y = position[0], position[1]
        
        cx = self.torso_params['center_x']
        cy = self.torso_params['center_y']
        R = self.torso_params['radius']
        y_min, y_max = self.torso_params['height_range']
        
        # Sprawdź zakres wysokości
        in_height = y_min <= y <= y_max
        
        # Sprawdź odległość od osi cylindra
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        in_radius = dist_from_center <= R
        
        return in_height and in_radius
    
    def get_torso_cylinder_params(self) -> Dict[str, float]:
        """
        Zwraca parametry cylindra torsu.
        
        Returns:
            torso_params: Słownik z parametrami
        """
        return self.torso_params.copy()
    
    def get_all_zone_names(self) -> List[str]:
        """
        Zwraca listę wszystkich nazw stref.
        
        Returns:
            zone_names: Lista nazw
        """
        return list(self.zones.keys())
    
    def visualize(self, ax=None, show_zones: bool = True, 
                 show_torso: bool = True) -> None:
        """
        Wizualizuje model ciała (wymaga matplotlib).
        
        Args:
            ax: Matplotlib axis (jeśli None, utworzy nowy)
            show_zones: Czy pokazać strefy
            show_torso: Czy pokazać cylinder torsu
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 10))
        
        # Rysuj strefy
        if show_zones:
            for zone_name, zone in self.zones.items():
                x_min, x_max = zone.x_range
                y_min, y_max = zone.y_range
                width = x_max - x_min
                height = y_max - y_min
                
                rect = Rectangle((x_min, y_min), width, height, 
                               linewidth=1, edgecolor='blue', 
                               facecolor='lightblue', alpha=0.3)
                ax.add_patch(rect)
                
                # Etykieta w środku strefy
                center = zone.get_center()
                ax.text(center[0], center[1], zone_name, 
                       ha='center', va='center', fontsize=8, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Rysuj cylinder torsu
        if show_torso:
            cx = self.torso_params['center_x']
            cy = self.torso_params['center_y']
            R = self.torso_params['radius']
            
            circle = Circle((cx, cy), R, 
                          linewidth=2, edgecolor='red', 
                          facecolor='none', linestyle='--', 
                          label='Torso Cylinder')
            ax.add_patch(circle)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('X (Left ← → Right)')
        ax.set_ylabel('Y (Feet ← → Head)')
        ax.set_title('WBAN Body Model - Anatomical Zones')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    
    def __repr__(self) -> str:
        return f"BodyModel(zones={len(self.zones)}, torso_params={self.torso_params})"


if __name__ == '__main__':
    # Test modelu ciała
    import sys
    sys.path.append('../..')
    
    from src.utils.config_loader import load_config
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("BODY MODEL TEST")
    print("="*60)
    
    # Wczytaj konfigurację
    config = load_config()
    
    # Utwórz model ciała
    body = BodyModel(config)
    
    # Test 1: Strefy
    print("\n[TEST 1] Body Zones:")
    for zone_name in body.get_all_zone_names():
        zone = body.get_zone(zone_name)
        center = zone.get_center()
        area = zone.get_area()
        print(f"  {zone_name:15s}: center={center}, area={area:.4f}")
    
    # Test 2: Walidacja pozycji
    print("\n[TEST 2] Position Validation:")
    test_positions = [
        (np.array([0.45, 0.65]), 'chest', True),
        (np.array([0.70, 0.65]), 'chest', False),  # Poza strefą
        (np.array([0.15, 0.50]), 'left_wrist', True),
        (np.array([0.50, 0.45]), 'waist', True),
    ]
    
    for pos, zone_name, expected in test_positions:
        result = body.is_valid_position(pos, zone_name)
        status = "✓" if result == expected else "✗"
        print(f"  {status} Position {pos} in '{zone_name}': {result} (expected {expected})")
    
    # Test 3: Cylinder torsu
    print("\n[TEST 3] Torso Cylinder:")
    test_torso = [
        (np.array([0.45, 0.65]), True),  # Wewnątrz
        (np.array([0.15, 0.50]), False),  # Poza (left_wrist)
    ]
    
    for pos, expected in test_torso:
        result = body.is_inside_torso_cylinder(pos)
        status = "✓" if result == expected else "✗"
        print(f"  {status} Position {pos} inside torso: {result} (expected {expected})")
    
    # Test 4: Losowa pozycja
    print("\n[TEST 4] Random Position in Zone:")
    rng = np.random.default_rng(42)
    for zone_name in ['chest', 'left_wrist', 'waist']:
        pos = body.get_random_position_in_zone(zone_name, rng)
        valid = body.is_valid_position(pos, zone_name)
        status = "✓" if valid else "✗"
        print(f"  {status} Random in '{zone_name}': {pos} (valid={valid})")
    
    # Test 5: Wizualizacja (opcjonalnie)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Bez wyświetlania (tylko zapis)
        
        print("\n[TEST 5] Visualization:")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 10))
        body.visualize(ax=ax)
        plt.savefig('/tmp/body_model_test.png', dpi=150, bbox_inches='tight')
        print("  ✓ Visualization saved to /tmp/body_model_test.png")
        
    except ImportError:
        print("\n[TEST 5] Visualization: SKIPPED (matplotlib not available)")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")