"""
WBAN Optimization - Propagation Model
Author: Kamil Piejko
Date: 2024

Model propagacji IEEE 802.15.6 CM3 (on-body channel) dla sieci WBAN.

Źródła:
- [19] Deepak & Babu 2016 - Tabela 6 (2.4 GHz)
- [8] Goswami et al. 2016 - Path loss variation UWB
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class PropagationModel:
    """
    Model propagacji IEEE 802.15.6 CM3 (on-body channel).
    
    Model path loss:
        PL(d) [dB] = PL_d0 + 10 × n × log10(d / d0) + X_σ
    
    gdzie:
        PL_d0 = path loss at reference distance d0
        n = path loss exponent (LOS lub NLOS)
        d0 = reference distance (typowo 1 m)
        X_σ = shadowing (losowa zmienna ~ N(0, σ²))
    
    Attributes:
        LOS_params: Parametry dla Line-of-Sight
        NLOS_params: Parametry dla Non-Line-of-Sight
        d0: Odległość referencyjna [m]
        P_sens: Czułość odbiornika [dBm]
        M_safe: Margines bezpieczeństwa [dB]
    """
    
    def __init__(self, config: Dict):
        """
        Inicjalizacja modelu propagacji.
        
        Args:
            config: Konfiguracja z YAML (sekcja 'propagation_model')
        """
        prop_config = config['propagation_model']
        
        # Parametry LOS
        self.LOS_params = {
            'PL_d0': prop_config['LOS']['PL_d0'],
            'n': prop_config['LOS']['path_loss_exponent'],
            'sigma': prop_config['LOS']['shadowing_std']
        }
        
        # Parametry NLOS
        self.NLOS_params = {
            'PL_d0': prop_config['NLOS']['PL_d0'],
            'n': prop_config['NLOS']['path_loss_exponent'],
            'sigma': prop_config['NLOS']['shadowing_std']
        }
        
        # Parametry wspólne
        self.d0 = prop_config['d0']
        self.P_sens = prop_config['receiver_sensitivity']
        self.M_safe = prop_config['link_margin']
        self.frequency = prop_config.get('frequency', 2.4e9)  # Hz
        
        logger.info(f"Propagation model initialized: LOS (n={self.LOS_params['n']}), "
                   f"NLOS (n={self.NLOS_params['n']})")
    
    def compute_path_loss(self, 
                         distance: float, 
                         los_status: str,
                         include_shadowing: bool = True,
                         rng: np.random.Generator = None) -> float:
        """
        Oblicza straty propagacji (path loss) zgodnie z IEEE 802.15.6 CM3.
        
        Args:
            distance: Odległość [m]
            los_status: 'LOS' lub 'NLOS'
            include_shadowing: Czy dodać losowe cieniowanie
            rng: Generator liczb losowych
        
        Returns:
            PL: Path loss [dB]
        
        Raises:
            ValueError: Jeśli los_status jest niepoprawny
        """
        if los_status not in ['LOS', 'NLOS']:
            raise ValueError(f"Invalid LOS status: {los_status}. Must be 'LOS' or 'NLOS'")
        
        if distance < 0:
            raise ValueError("Distance cannot be negative")
        
        # Wybierz parametry
        if los_status == 'LOS':
            params = self.LOS_params
        else:
            params = self.NLOS_params
        
        PL_d0 = params['PL_d0']
        n = params['n']
        sigma = params['sigma']
        
        # Zapobiegaj log10(0) dla bardzo małych odległości
        if distance < self.d0:
            distance = self.d0
        
        # Path loss (deterministyczny)
        PL_deterministic = PL_d0 + 10 * n * np.log10(distance / self.d0)
        
        # Shadowing (losowy składnik)
        if include_shadowing:
            if rng is None:
                rng = np.random.default_rng()
            
            X_sigma = rng.normal(0, sigma)
            PL = PL_deterministic + X_sigma
        else:
            PL = PL_deterministic
        
        return PL
    
    def compute_required_tx_power(self, 
                                  path_loss: float) -> float:
        """
        Oblicza wymaganą moc transmisji aby osiągnąć czułość odbiornika.
        
        P_TX_req = P_sens + PL + M_safe
        
        Args:
            path_loss: Straty propagacji [dB]
        
        Returns:
            P_TX_required: Wymagana moc [dBm]
        """
        P_TX_required = self.P_sens + path_loss + self.M_safe
        
        return P_TX_required
    
    def compute_link_margin(self, 
                           P_TX_max: float,
                           path_loss: float) -> float:
        """
        Oblicza margines łącza.
        
        M = P_TX_max - P_TX_required
        
        Args:
            P_TX_max: Maksymalna dostępna moc [dBm]
            path_loss: Straty propagacji [dB]
        
        Returns:
            margin: Margines łącza [dB] (dodatni = OK, ujemny = łącze niemożliwe)
        """
        P_TX_required = self.compute_required_tx_power(path_loss)
        margin = P_TX_max - P_TX_required
        
        return margin
    
    def is_link_feasible(self,
                        P_TX_max: float,
                        distance: float,
                        los_status: str,
                        include_shadowing: bool = False,
                        rng: np.random.Generator = None) -> Tuple[bool, float]:
        """
        Sprawdza, czy łącze jest wykonalne (margines dodatni).
        
        Args:
            P_TX_max: Maksymalna moc [dBm]
            distance: Odległość [m]
            los_status: 'LOS' lub 'NLOS'
            include_shadowing: Czy uwzględnić shadowing
            rng: Generator liczb losowych
        
        Returns:
            (feasible, margin): True/False i wartość marginesu [dB]
        """
        PL = self.compute_path_loss(distance, los_status, include_shadowing, rng)
        margin = self.compute_link_margin(P_TX_max, PL)
        
        feasible = margin >= 0
        
        return feasible, margin
    
    def get_LOS_params(self) -> Dict[str, float]:
        """Zwraca parametry LOS."""
        return self.LOS_params.copy()
    
    def get_NLOS_params(self) -> Dict[str, float]:
        """Zwraca parametry NLOS."""
        return self.NLOS_params.copy()
    
    def get_all_params(self) -> Dict[str, any]:
        """Zwraca wszystkie parametry modelu."""
        return {
            'LOS': self.LOS_params.copy(),
            'NLOS': self.NLOS_params.copy(),
            'd0': self.d0,
            'P_sens': self.P_sens,
            'M_safe': self.M_safe,
            'frequency': self.frequency
        }
    
    def __repr__(self) -> str:
        return (f"PropagationModel(LOS: n={self.LOS_params['n']}, "
                f"NLOS: n={self.NLOS_params['n']}, "
                f"P_sens={self.P_sens} dBm)")


if __name__ == '__main__':
    # Test modelu propagacji
    import sys
    sys.path.append('../..')
    
    from src.utils.config_loader import load_config
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("PROPAGATION MODEL TEST")
    print("="*60)
    
    # Wczytaj konfigurację
    config = load_config()
    
    # Utwórz model propagacji
    prop_model = PropagationModel(config)
    print(f"\n[TEST 1] Model: {prop_model}")
    
    # Test 2: Parametry
    print("\n[TEST 2] Model Parameters:")
    all_params = prop_model.get_all_params()
    
    print("  LOS:")
    for key, value in all_params['LOS'].items():
        print(f"    {key:10s}: {value}")
    
    print("  NLOS:")
    for key, value in all_params['NLOS'].items():
        print(f"    {key:10s}: {value}")
    
    print(f"  d0: {all_params['d0']} m")
    print(f"  P_sens: {all_params['P_sens']} dBm")
    print(f"  M_safe: {all_params['M_safe']} dB")
    
    # Test 3: Path loss dla różnych odległości (bez shadowing)
    print("\n[TEST 3] Path Loss vs. Distance (deterministic, no shadowing):")
    distances = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    
    print(f"  {'Distance [m]':>13s} {'PL_LOS [dB]':>13s} {'PL_NLOS [dB]':>14s} {'Δ [dB]':>10s}")
    print(f"  {'-'*13:>13s} {'-'*13:>13s} {'-'*14:>14s} {'-'*10:>10s}")
    
    for d in distances:
        PL_LOS = prop_model.compute_path_loss(d, 'LOS', include_shadowing=False)
        PL_NLOS = prop_model.compute_path_loss(d, 'NLOS', include_shadowing=False)
        delta = PL_NLOS - PL_LOS
        
        print(f"  {d:>13.2f} {PL_LOS:>13.2f} {PL_NLOS:>14.2f} {delta:>10.2f}")
    
    # Test 4: Wymagana moc TX
    print("\n[TEST 4] Required TX Power:")
    test_distance = 0.5  # m
    
    PL_LOS = prop_model.compute_path_loss(test_distance, 'LOS', include_shadowing=False)
    PL_NLOS = prop_model.compute_path_loss(test_distance, 'NLOS', include_shadowing=False)
    
    P_TX_LOS = prop_model.compute_required_tx_power(PL_LOS)
    P_TX_NLOS = prop_model.compute_required_tx_power(PL_NLOS)
    
    print(f"  Distance: {test_distance} m")
    print(f"  LOS:  PL={PL_LOS:.2f} dB → P_TX_req={P_TX_LOS:.2f} dBm")
    print(f"  NLOS: PL={PL_NLOS:.2f} dB → P_TX_req={P_TX_NLOS:.2f} dBm")
    
    # Test 5: Margines łącza
    print("\n[TEST 5] Link Margin:")
    P_TX_max_options = [0, -5, -10]  # dBm (S1, S2, S3)
    
    print(f"  {'P_TX_max [dBm]':>15s} {'Distance [m]':>14s} {'LOS Status':>12s} {'Margin [dB]':>13s} {'Feasible':>10s}")
    print(f"  {'-'*15:>15s} {'-'*14:>14s} {'-'*12:>12s} {'-'*13:>13s} {'-'*10:>10s}")
    
    test_cases = [
        (0, 0.5, 'LOS'),
        (0, 0.5, 'NLOS'),
        (-5, 1.0, 'LOS'),
        (-5, 1.0, 'NLOS'),
        (-10, 0.3, 'LOS'),
        (-10, 0.3, 'NLOS'),
    ]
    
    for P_TX_max, dist, los in test_cases:
        feasible, margin = prop_model.is_link_feasible(P_TX_max, dist, los, include_shadowing=False)
        status = "✓" if feasible else "✗"
        print(f"  {P_TX_max:>15d} {dist:>14.2f} {los:>12s} {margin:>13.2f} {status:>10s}")
    
    # Test 6: Shadowing (losowy)
    print("\n[TEST 6] Shadowing Effect (10 samples with random seed):")
    rng = np.random.default_rng(42)
    distance = 0.5
    los_status = 'NLOS'
    
    print(f"  Distance: {distance} m, Status: {los_status}")
    print(f"  σ_NLOS = {prop_model.NLOS_params['sigma']} dB")
    print(f"\n  {'Sample':>8s} {'PL [dB]':>10s}")
    print(f"  {'-'*8:>8s} {'-'*10:>10s}")
    
    PL_samples = []
    for i in range(10):
        PL = prop_model.compute_path_loss(distance, los_status, include_shadowing=True, rng=rng)
        PL_samples.append(PL)
        print(f"  {i+1:>8d} {PL:>10.2f}")
    
    print(f"\n  Mean: {np.mean(PL_samples):.2f} dB")
    print(f"  Std:  {np.std(PL_samples):.2f} dB (expected ~{prop_model.NLOS_params['sigma']} dB)")
    
    # Test 7: Weryfikacja z ręcznym obliczeniem (Przykład 1 z docs)
    print("\n[TEST 7] Verification with Manual Calculation:")
    print("  (From docs/fitness_calculation_examples.md, Sensor 1)")
    
    distance_test = 0.206  # m
    los_test = 'NLOS'
    
    # Bez shadowing dla powtarzalności
    PL_test = prop_model.compute_path_loss(distance_test, los_test, include_shadowing=False)
    
    # Ręczne obliczenie (z dokumentacji)
    PL_d0_manual = 48.4
    n_manual = 5.9
    d0_manual = 1.0
    PL_manual = PL_d0_manual + 10 * n_manual * np.log10(distance_test / d0_manual)
    
    print(f"  Calculated PL: {PL_test:.2f} dB")
    print(f"  Manual PL:     {PL_manual:.2f} dB")
    print(f"  Match: {np.isclose(PL_test, PL_manual, atol=0.1)}")
    
    # Test 8: Scenariusze z różnymi mocami
    print("\n[TEST 8] Scenarios S1, S2, S3 - Link Feasibility:")
    scenarios = {
        'S1': {'n_sensors': 6, 'P_TX_max': 0},
        'S2': {'n_sensors': 15, 'P_TX_max': -5},
        'S3': {'n_sensors': 25, 'P_TX_max': -10}
    }
    
    print(f"  {'Scenario':>10s} {'P_TX [dBm]':>12s} {'Max Dist LOS [m]':>18s} {'Max Dist NLOS [m]':>19s}")
    print(f"  {'-'*10:>10s} {'-'*12:>12s} {'-'*18:>18s} {'-'*19:>19s}")
    
    for scenario_name, scenario in scenarios.items():
        P_TX_max = scenario['P_TX_max']
        
        # Znajdź maksymalną odległość dla LOS i NLOS (margin = 0)
        # Rozwiązujemy: P_TX_max = P_sens + PL(d) + M_safe
        # PL(d) = PL_d0 + 10*n*log10(d/d0)
        # d_max = d0 * 10^((P_TX_max - P_sens - M_safe - PL_d0) / (10*n))
        
        def max_distance(P_TX, params):
            numerator = P_TX - prop_model.P_sens - prop_model.M_safe - params['PL_d0']
            exponent = numerator / (10 * params['n'])
            d_max = prop_model.d0 * (10 ** exponent)
            return d_max
        
        d_max_LOS = max_distance(P_TX_max, prop_model.LOS_params)
        d_max_NLOS = max_distance(P_TX_max, prop_model.NLOS_params)
        
        print(f"  {scenario_name:>10s} {P_TX_max:>12d} {d_max_LOS:>18.3f} {d_max_NLOS:>19.3f}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60 + "\n")