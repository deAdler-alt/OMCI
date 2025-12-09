"""
WBAN Optimization - Configuration Loader
Author: Kamil Piejko
Date: 2024

Moduł do wczytywania i walidacji konfiguracji z pliku YAML.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Loader konfiguracji systemu WBAN z pliku YAML.
    
    Attributes:
        config_path (Path): Ścieżka do pliku konfiguracyjnego
        config (Dict): Wczytana konfiguracja
    """
    
    def __init__(self, config_path: str = "config/wban_params.yaml"):
        """
        Inicjalizacja loadera.
        
        Args:
            config_path: Ścieżka do pliku YAML
        """
        self.config_path = Path(config_path)
        self.config: Optional[Dict[str, Any]] = None
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
    
    def load(self) -> Dict[str, Any]:
        """
        Wczytuje konfigurację z pliku YAML.
        
        Returns:
            config: Słownik z konfiguracją
        
        Raises:
            ValueError: Jeśli plik jest uszkodzony lub niepoprawny
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
            # Walidacja podstawowych sekcji
            self._validate_structure()
            
            return self.config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
    
    def _validate_structure(self):
        """
        Waliduje strukturę konfiguracji - sprawdza wymagane sekcje.
        
        Raises:
            ValueError: Jeśli brakuje wymaganych sekcji
        """
        required_sections = [
            'body_model',
            'sensor_types',
            'energy_model',
            'propagation_model',
            'scenarios',
            'optimization',
            'fitness_function',
            'experiments'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
        
        logger.info("Configuration structure validated ✓")
    
    def get_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Pobiera konfigurację konkretnego scenariusza.
        
        Args:
            scenario_name: Nazwa scenariusza (np. 'S1', 'S2', 'S3')
        
        Returns:
            scenario_config: Konfiguracja scenariusza
        
        Raises:
            KeyError: Jeśli scenariusz nie istnieje
        """
        if scenario_name not in self.config['scenarios']:
            raise KeyError(f"Scenario {scenario_name} not found in config")
        
        return self.config['scenarios'][scenario_name]
    
    def get_body_zones(self) -> Dict[str, Dict[str, list]]:
        """
        Pobiera definicje stref anatomicznych.
        
        Returns:
            zones: Słownik ze strefami ciała
        """
        return self.config['body_model']['zones']
    
    def get_energy_params(self) -> Dict[str, float]:
        """
        Pobiera parametry modelu energetycznego.
        
        Returns:
            energy_params: Parametry energii
        """
        return self.config['energy_model']
    
    def get_propagation_params(self) -> Dict[str, Any]:
        """
        Pobiera parametry modelu propagacji.
        
        Returns:
            propagation_params: Parametry propagacji
        """
        return self.config['propagation_model']
    
    def get_algorithm_params(self, algorithm: str) -> Dict[str, Any]:
        """
        Pobiera parametry algorytmu optymalizacyjnego.
        
        Args:
            algorithm: Nazwa algorytmu ('GA' lub 'PSO')
        
        Returns:
            algorithm_params: Parametry algorytmu
        
        Raises:
            KeyError: Jeśli algorytm nie istnieje
        """
        if algorithm not in self.config['optimization']:
            raise KeyError(f"Algorithm {algorithm} not found in config")
        
        return self.config['optimization'][algorithm]
    
    def get_fitness_weights(self, variant_name: str) -> Dict[str, float]:
        """
        Pobiera wagi funkcji fitness dla danego wariantu.
        
        Args:
            variant_name: Nazwa wariantu (np. 'energy_priority', 'balanced')
        
        Returns:
            weights: Słownik z wagami {w_E, w_R}
        
        Raises:
            KeyError: Jeśli wariant nie istnieje
        """
        variants = self.config['fitness_function']['weight_variants']
        
        if variant_name not in variants:
            raise KeyError(f"Weight variant {variant_name} not found")
        
        variant = variants[variant_name]
        return {'w_E': variant['w_E'], 'w_R': variant['w_R']}
    
    def __getitem__(self, key: str) -> Any:
        """
        Umożliwia dostęp do konfiguracji jak do słownika.
        
        Args:
            key: Klucz w konfiguracji
        
        Returns:
            value: Wartość dla klucza
        """
        if self.config is None:
            self.load()
        
        return self.config[key]
    
    def __repr__(self) -> str:
        """Reprezentacja tekstowa obiektu."""
        if self.config is None:
            return f"ConfigLoader(path={self.config_path}, loaded=False)"
        else:
            n_scenarios = len(self.config.get('scenarios', {}))
            return f"ConfigLoader(path={self.config_path}, scenarios={n_scenarios})"


def load_config(config_path: str = "config/wban_params.yaml") -> Dict[str, Any]:
    """
    Funkcja pomocnicza do szybkiego wczytania konfiguracji.
    
    Args:
        config_path: Ścieżka do pliku YAML
    
    Returns:
        config: Słownik z konfiguracją
    
    Example:
        >>> config = load_config()
        >>> print(config['scenarios']['S1']['n_sensors'])
        6
    """
    loader = ConfigLoader(config_path)
    return loader.load()


if __name__ == '__main__':
    # Test loadera
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Wczytaj konfigurację
        loader = ConfigLoader()
        config = loader.load()
        
        print("\n" + "="*60)
        print("CONFIG LOADER TEST")
        print("="*60)
        
        # Test 1: Scenariusze
        print("\n[TEST 1] Scenarios:")
        for scenario_name in ['S1', 'S2', 'S3']:
            scenario = loader.get_scenario(scenario_name)
            print(f"  {scenario_name}: {scenario['n_sensors']} sensors, "
                  f"P_TX={scenario['P_TX_max']} dBm")
        
        # Test 2: Parametry energetyczne
        print("\n[TEST 2] Energy Parameters:")
        energy = loader.get_energy_params()
        print(f"  E_elec_TX: {energy['E_elec_TX']*1e9:.1f} nJ/bit")
        print(f"  E_init: {energy['E_init']} J")
        
        # Test 3: Parametry propagacji
        print("\n[TEST 3] Propagation Parameters:")
        prop = loader.get_propagation_params()
        print(f"  LOS: n={prop['LOS']['path_loss_exponent']}, "
              f"σ={prop['LOS']['shadowing_std']} dB")
        print(f"  NLOS: n={prop['NLOS']['path_loss_exponent']}, "
              f"σ={prop['NLOS']['shadowing_std']} dB")
        
        # Test 4: Wagi fitness
        print("\n[TEST 4] Fitness Weights:")
        for variant in ['energy_priority', 'balanced', 'reliability_priority']:
            weights = loader.get_fitness_weights(variant)
            print(f"  {variant}: w_E={weights['w_E']}, w_R={weights['w_R']}")
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60 + "\n")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)