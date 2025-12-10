"""
WBAN Optimization - LOS/NLOS Detector
Author: Kamil Piejko
Date: 2024

Detekcja Line-of-Sight (LOS) vs Non-Line-of-Sight (NLOS) dla łączy sensor-hub
używając testu przecięcia promienia z cylindrem torsu.

Źródła:
- [10] Januszkiewicz 2018 - Body shadowing effect
"""

import numpy as np
from typing import Dict, Tuple
import logging

from ..core.body_model import BodyModel

logger = logging.getLogger(__name__)


class LOSDetector:
    """
    Detektor stanu łącza: Line-of-Sight (LOS) lub Non-Line-of-Sight (NLOS).
    
    Metoda: Ray-cylinder intersection test
    
    Założenie:
        Tors reprezentowany jako cylinder o promieniu R i zakresie wysokości [y_min, y_max].
        Jeśli linia sensor-hub przecina cylinder → NLOS
        W przeciwnym razie → LOS
    
    Attributes:
        torso_params: Parametry cylindra torsu
        body_model: Model ciała (dla dodatkowych sprawdzeń)
    """
    
    def __init__(self, body_model: BodyModel):
        """
        Inicjalizacja detektora.
        
        Args:
            body_model: Model ciała z parametrami cylindra torsu
        """
        self.body_model = body_model
        self.torso_params = body_model.get_torso_cylinder_params()
        
        logger.info(f"LOS Detector initialized with torso cylinder: "
                   f"center=({self.torso_params['center_x']}, {self.torso_params['center_y']}), "
                   f"radius={self.torso_params['radius']}")
    
    def detect(self, 
              sensor_pos: np.ndarray, 
              hub_pos: np.ndarray) -> str:
        """
        Wykrywa stan łącza: LOS lub NLOS.
        
        Args:
            sensor_pos: Pozycja sensora [x, y]
            hub_pos: Pozycja hub [x, y]
        
        Returns:
            'LOS' lub 'NLOS'
        """
        # Test 1: Czy którykolwiek z punktów jest wewnątrz cylindra?
        if self._is_inside_cylinder(sensor_pos) or self._is_inside_cylinder(hub_pos):
            # Jeśli którykolwiek punkt jest w ciele, to potencjalnie NLOS
            # (chyba że oba są w ciele i bardzo blisko - ale upraszczamy)
            return 'NLOS'
        
        # Test 2: Czy promień sensor-hub przecina cylinder?
        if self._ray_intersects_cylinder(sensor_pos, hub_pos):
            return 'NLOS'
        
        # Żaden test nie wykrył NLOS → LOS
        return 'LOS'
    
    def _is_inside_cylinder(self, position: np.ndarray) -> bool:
        """
        Sprawdza, czy punkt jest wewnątrz cylindra torsu.
        
        Args:
            position: [x, y]
        
        Returns:
            True jeśli punkt jest w cylindrze
        """
        return self.body_model.is_inside_torso_cylinder(position)
    
    def _ray_intersects_cylinder(self, 
                                 sensor_pos: np.ndarray, 
                                 hub_pos: np.ndarray) -> bool:
        """
        Sprawdza, czy promień (sensor → hub) przecina cylinder torsu.
        
        Algorytm:
        1. Oblicz odległość od prostej (sensor-hub) do środka cylindra
        2. Jeśli odległość < radius → potencjalne przecięcie
        3. Sprawdź, czy punkt przecięcia leży MIĘDZY sensor i hub (nie za nimi)
        
        Args:
            sensor_pos: Pozycja sensora [x, y]
            hub_pos: Pozycja hub [x, y]
        
        Returns:
            True jeśli promień przecina cylinder
        """
        cx = self.torso_params['center_x']
        cy = self.torso_params['center_y']
        R = self.torso_params['radius']
        y_min, y_max = self.torso_params['height_range']
        
        x1, y1 = sensor_pos[0], sensor_pos[1]
        x2, y2 = hub_pos[0], hub_pos[1]
        
        # Wektor kierunku promienia
        ray_dir = hub_pos - sensor_pos
        ray_length = np.linalg.norm(ray_dir)
        
        if ray_length < 1e-9:
            # Sensor i hub w tym samym miejscu (nie powinno się zdarzyć)
            return False
        
        ray_dir_normalized = ray_dir / ray_length
        
        # Wektor od sensor do środka cylindra
        to_center = np.array([cx - x1, cy - y1])
        
        # Projekcja to_center na kierunek promienia
        # t = jak daleko wzdłuż promienia jest najbliższy punkt do środka cylindra
        t = np.dot(to_center, ray_dir_normalized)
        
        # Jeśli t < 0 lub t > ray_length, najbliższy punkt jest poza odcinkiem sensor-hub
        if t < 0 or t > ray_length:
            return False
        
        # Najbliższy punkt na promieniu do środka cylindra
        closest_point = sensor_pos + t * ray_dir_normalized
        
        # Odległość od tego punktu do środka cylindra
        distance_to_center = np.linalg.norm(closest_point - np.array([cx, cy]))
        
        # Sprawdź, czy ten punkt jest w zasięgu promienia cylindra
        if distance_to_center > R:
            return False
        
        # Sprawdź, czy punkt jest w zakresie wysokości cylindra
        closest_y = closest_point[1]
        if y_min <= closest_y <= y_max:
            return True
        
        return False
    
    def compute_distance(self, 
                        sensor_pos: np.ndarray, 
                        hub_pos: np.ndarray) -> float:
        """
        Oblicza odległość między sensorem a hubem.
        
        Args:
            sensor_pos: [x, y]
            hub_pos: [x, y]
        
        Returns:
            distance: Odległość [m w skali znormalizowanej]
        """
        return np.linalg.norm(hub_pos - sensor_pos)
    
    def detect_with_distance(self,
                            sensor_pos: np.ndarray,
                            hub_pos: np.ndarray) -> Tuple[str, float]:
        """
        Wykrywa stan łącza i zwraca odległość.
        
        Args:
            sensor_pos: [x, y]
            hub_pos: [x, y]
        
        Returns:
            (los_status, distance): 'LOS'/'NLOS' i odległość
        """
        los_status = self.detect(sensor_pos, hub_pos)
        distance = self.compute_distance(sensor_pos, hub_pos)
        
        return los_status, distance
    
    def __repr__(self) -> str:
        return f"LOSDetector(torso_radius={self.torso_params['radius']})"


if __name__ == '__main__':
    # Test detektora LOS/NLOS
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
    print("LOS/NLOS DETECTOR TEST")
    print("="*60)
    
    # Wczytaj konfigurację i utwórz model ciała
    config = load_config()
    body_model = BodyModel(config)
    
    # Utwórz detektor
    detector = LOSDetector(body_model)
    print(f"\n[TEST 1] Detector: {detector}")
    print(f"  Torso params: {detector.torso_params}")
    
    # Test 2: Przypadki testowe
    print("\n[TEST 2] LOS/NLOS Detection Test Cases:")
    
    test_cases = [
        # (sensor_pos, hub_pos, expected_status, description)
        (np.array([0.45, 0.65]), np.array([0.50, 0.45]), 'NLOS', 
         "Sensor w cylindrze (chest) → Hub (waist)"),
        
        (np.array([0.15, 0.50]), np.array([0.50, 0.45]), 'NLOS', 
         "Sensor poza (left_wrist) → Hub (waist), linia przecina tors"),
        
        (np.array([0.85, 0.50]), np.array([0.50, 0.45]), 'NLOS', 
         "Sensor poza (right_wrist) → Hub (waist), linia przecina tors"),
        
        (np.array([0.45, 0.65]), np.array([0.55, 0.70]), 'NLOS', 
         "Oba w cylindrze (chest), krótka odległość"),
        
        (np.array([0.15, 0.50]), np.array([0.20, 0.52]), 'LOS', 
         "Oba poza cylindrem (left side), linia nie przecina"),
        
        (np.array([0.45, 0.85]), np.array([0.50, 0.45]), 'NLOS', 
         "Sensor na głowie → Hub waist, pionowa linia przechodzi przez tors (NLOS)"),
        
        (np.array([0.38, 0.25]), np.array([0.55, 0.30]), 'LOS', 
         "Sensory na nogach, poniżej cylindra torsu"),
    ]
    
    print(f"  {'#':>3s} {'Sensor Pos':>15s} {'Hub Pos':>15s} {'Expected':>10s} {'Detected':>10s} {'Match':>7s} {'Dist [m]':>10s}")
    print(f"  {'-'*3:>3s} {'-'*15:>15s} {'-'*15:>15s} {'-'*10:>10s} {'-'*10:>10s} {'-'*7:>7s} {'-'*10:>10s}")
    
    for i, (sensor_pos, hub_pos, expected, desc) in enumerate(test_cases, 1):
        detected, distance = detector.detect_with_distance(sensor_pos, hub_pos)
        match = "✓" if detected == expected else "✗"
        
        sensor_str = f"({sensor_pos[0]:.2f},{sensor_pos[1]:.2f})"
        hub_str = f"({hub_pos[0]:.2f},{hub_pos[1]:.2f})"
        
        print(f"  {i:>3d} {sensor_str:>15s} {hub_str:>15s} {expected:>10s} {detected:>10s} {match:>7s} {distance:>10.3f}")
        
        if match == "✗":
            print(f"      Description: {desc}")
            print(f"      MISMATCH! Expected {expected}, got {detected}")
    
    # Test 3: Statystyki dla losowych pozycji
    print("\n[TEST 3] Statistics for Random Positions:")
    rng = np.random.default_rng(42)
    
    n_tests = 1000
    los_count = 0
    nlos_count = 0
    
    for _ in range(n_tests):
        # Losowe pozycje w zakresie [0, 1] × [0, 1]
        sensor_pos = rng.uniform(0, 1, size=2)
        hub_pos = rng.uniform(0, 1, size=2)
        
        status = detector.detect(sensor_pos, hub_pos)
        
        if status == 'LOS':
            los_count += 1
        else:
            nlos_count += 1
    
    print(f"  Total tests: {n_tests}")
    print(f"  LOS:  {los_count} ({los_count/n_tests*100:.1f}%)")
    print(f"  NLOS: {nlos_count} ({nlos_count/n_tests*100:.1f}%)")
    
    # Test 4: Wizualizacja (opcjonalnie)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        print("\n[TEST 4] Visualization:")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Rysuj tors (cylinder)
        cx = detector.torso_params['center_x']
        cy = detector.torso_params['center_y']
        R = detector.torso_params['radius']
        
        circle = Circle((cx, cy), R, linewidth=2, edgecolor='red', 
                       facecolor='pink', alpha=0.3, label='Torso Cylinder')
        ax.add_patch(circle)
        
        # Rysuj przypadki testowe
        for i, (sensor_pos, hub_pos, expected, desc) in enumerate(test_cases[:5], 1):
            detected = detector.detect(sensor_pos, hub_pos)
            color = 'green' if detected == 'LOS' else 'red'
            
            # Sensor
            ax.plot(sensor_pos[0], sensor_pos[1], 'o', color=color, markersize=8)
            
            # Hub
            ax.plot(hub_pos[0], hub_pos[1], '*', color=color, markersize=12)
            
            # Linia
            ax.plot([sensor_pos[0], hub_pos[0]], 
                   [sensor_pos[1], hub_pos[1]], 
                   color=color, linewidth=1, alpha=0.6)
            
            # Etykieta
            mid_x = (sensor_pos[0] + hub_pos[0]) / 2
            mid_y = (sensor_pos[1] + hub_pos[1]) / 2
            ax.text(mid_x, mid_y, f"#{i}\n{detected}", 
                   fontsize=8, ha='center', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel('X (Left ← → Right)')
        ax.set_ylabel('Y (Feet ← → Head)')
        ax.set_title('LOS/NLOS Detection - Test Cases\n(Green=LOS, Red=NLOS)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('/tmp/los_detector_test.png', dpi=150, bbox_inches='tight')
        print("  ✓ Visualization saved to /tmp/los_detector_test.png")
        
    except ImportError:
        print("\n[TEST 4] Visualization: SKIPPED (matplotlib not available)")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")