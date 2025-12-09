# WBAN Sensor Placement Optimization - Technical Specification
**Autor:** Kamil Piejko  
**Data:** 2024  
**Wersja:** 1.0

---

## 📋 Spis Treści

1. [Architektura Systemu](#1-architektura-systemu)
2. [Model Danych](#2-model-danych)
3. [Funkcja Fitness - Szczegółowa Specyfikacja](#3-funkcja-fitness)
4. [Detekcja LOS/NLOS](#4-detekcja-losnlos)
5. [Model Energetyczny](#5-model-energetyczny)
6. [Model Propagacji](#6-model-propagacji)
7. [Algorytmy Optymalizacyjne](#7-algorytmy-optymalizacyjne)
8. [Pipeline Eksperymentów](#8-pipeline-eksperymentów)

---

## 1. Architektura Systemu

### 1.1 Struktura Modułów

```
wban_optimization/
├── config/
│   └── wban_params.yaml          # Konfiguracja (już utworzona)
├── src/
│   ├── core/
│   │   ├── body_model.py         # Model ciała i stref anatomicznych
│   │   ├── sensor.py             # Klasy Sensor i Hub
│   │   └── genotype.py           # Kodowanie/dekodowanie genotypu
│   ├── models/
│   │   ├── energy_model.py       # First Order Radio Model
│   │   ├── propagation_model.py  # IEEE 802.15.6 path loss
│   │   └── los_detector.py       # Detekcja LOS/NLOS
│   ├── optimization/
│   │   ├── fitness_function.py   # Funkcja celu
│   │   ├── ga_optimizer.py       # Genetic Algorithm (Mealpy)
│   │   └── pso_optimizer.py      # PSO (Mealpy)
│   ├── baselines/
│   │   ├── random_placement.py   # Losowe rozmieszczenie
│   │   └── naive_centroid.py     # Naiwny baseline
│   └── utils/
│       ├── config_loader.py      # Wczytywanie YAML
│       ├── logger.py             # Logowanie
│       └── validators.py         # Walidacja rozwiązań
├── experiments/
│   ├── run_scenarios.py          # Główny skrypt eksperymentów
│   ├── power_analysis.py         # Analiza mocy TX
│   └── collect_metrics.py        # Agregacja wyników
├── visualization/
│   ├── plot_convergence.py
│   ├── plot_placement.py
│   └── plot_metrics.py
└── results/
    ├── experiments/
    ├── plots/
    └── statistics/
```

---

## 2. Model Danych

### 2.1 Klasa `Sensor`

```python
@dataclass
class Sensor:
    """
    Reprezentacja pojedynczego sensora w sieci WBAN.
    """
    id: int
    type: str  # np. 'ECG', 'SpO2', 'Accelerometer'
    position: np.ndarray  # [x, y] w przestrzeni znormalizowanej [0,1]²
    assigned_zone: str  # np. 'chest', 'left_wrist'
    
    # Parametry komunikacyjne
    data_rate: float  # bps
    packet_size: int  # bits
    
    # Stan energetyczny
    energy_remaining: float  # J
    energy_initial: float  # J
    
    # Metryki transmisji
    transmitted_packets: int = 0
    energy_consumed: float = 0.0
    
    def is_alive(self) -> bool:
        """Czy sensor ma energię?"""
        return self.energy_remaining > 0.01  # J threshold
```

### 2.2 Klasa `Hub`

```python
@dataclass
class Hub:
    """
    Koncentrator (sink) - odbiera dane ze wszystkich sensorów.
    """
    position: np.ndarray  # [x, y]
    zone: str = 'waist'  # Preferowana strefa
    
    # Hub nie zużywa energii (podłączony do zasilania)
    energy_unlimited: bool = True
    
    # Metryki
    received_packets: int = 0
    total_throughput: float = 0.0  # bits/s
```

### 2.3 Genotyp (Kodowanie Rozwiązania)

```python
class Genotype:
    """
    Kodowanie pozycji sensorów i Hub jako wektor liczb rzeczywistych.
    
    Struktura:
        g = [x1, y1, x2, y2, ..., xN, yN, x_hub, y_hub]
    
    Wymiar: D = 2*N + 2
    Zakres: [0, 1] dla wszystkich współrzędnych
    """
    
    def __init__(self, n_sensors: int):
        self.n_sensors = n_sensors
        self.dimension = 2 * n_sensors + 2
        self.bounds = [(0.0, 1.0)] * self.dimension
    
    @staticmethod
    def decode(genome: np.ndarray, sensor_config: List[Dict]) -> Tuple[List[Sensor], Hub]:
        """
        Dekodowanie genotypu na obiekty Sensor i Hub.
        
        Args:
            genome: Wektor [x1,y1,...,xN,yN,x_hub,y_hub]
            sensor_config: Konfiguracja sensorów (typy, strefy)
        
        Returns:
            (sensors, hub): Lista sensorów i obiekt Hub
        """
        n_sensors = (len(genome) - 2) // 2
        sensors = []
        
        for i in range(n_sensors):
            x, y = genome[2*i], genome[2*i + 1]
            config = sensor_config[i]
            
            sensor = Sensor(
                id=i+1,
                type=config['type'],
                position=np.array([x, y]),
                assigned_zone=config['zone'],
                data_rate=config['data_rate'],
                packet_size=config['packet_size'],
                energy_remaining=E_INIT,  # z config
                energy_initial=E_INIT
            )
            sensors.append(sensor)
        
        x_hub, y_hub = genome[-2], genome[-1]
        hub = Hub(position=np.array([x_hub, y_hub]))
        
        return sensors, hub
    
    @staticmethod
    def is_valid_position(position: np.ndarray, zone: str, body_zones: Dict) -> bool:
        """
        Sprawdza, czy pozycja (x,y) jest w przypisanej strefie anatomicznej.
        
        Args:
            position: [x, y]
            zone: Nazwa strefy (np. 'chest')
            body_zones: Słownik ze strefami z config
        
        Returns:
            True jeśli pozycja jest w strefie
        """
        if zone not in body_zones:
            return False
        
        zone_def = body_zones[zone]
        x, y = position
        
        in_x = zone_def['x_range'][0] <= x <= zone_def['x_range'][1]
        in_y = zone_def['y_range'][0] <= y <= zone_def['y_range'][1]
        
        return in_x and in_y
```

---

## 3. Funkcja Fitness

### 3.1 Definicja Matematyczna

```
F(g) = w_E × E_total(g) + w_R × P_rel(g) + P_geo(g)

gdzie:
- E_total(g): Całkowita energia zużyta przez wszystkie sensory [J]
- P_rel(g): Kara za niską niezawodność (słabe marginesy łączy) [bezwymiarowa]
- P_geo(g): Kara za naruszenia geometryczne (sensor poza strefą) [bezwymiarowa]
- w_E, w_R: Wagi (np. 0.7, 0.3)
```

### 3.2 Pseudokod

```python
def fitness_function(genome: np.ndarray, config: Dict, weights: Dict) -> float:
    """
    Główna funkcja fitness do minimalizacji.
    
    Args:
        genome: Genotyp [x1,y1,...,xN,yN,x_hub,y_hub]
        config: Parametry z wban_params.yaml
        weights: {w_E: float, w_R: float}
    
    Returns:
        fitness: Wartość fitness (niższa = lepsza)
    """
    # ========================================
    # KROK 1: Dekodowanie genotypu
    # ========================================
    sensors, hub = Genotype.decode(genome, config['scenarios']['SX']['sensor_config'])
    
    # ========================================
    # KROK 2: Walidacja geometryczna
    # ========================================
    penalty_geo = compute_geometric_penalty(sensors, hub, config['body_model']['zones'])
    
    if penalty_geo > 0:
        # Rozwiązanie niedopuszczalne - zwróć ogromną karę
        return 1e6 + penalty_geo
    
    # ========================================
    # KROK 3: Obliczenie całkowitej energii
    # ========================================
    E_total = 0.0
    link_margins = []
    
    for sensor in sensors:
        # 3a. Oblicz odległość euklidesową sensor-hub
        distance = np.linalg.norm(sensor.position - hub.position)
        
        # 3b. Detekcja LOS/NLOS
        los_status = detect_LOS_NLOS(
            sensor.position, 
            hub.position, 
            config['body_model']['torso_cylinder']
        )
        
        # 3c. Oblicz path loss
        PL = compute_path_loss(
            distance, 
            los_status, 
            config['propagation_model']
        )
        
        # 3d. Wymagana moc transmisji
        P_TX_required = config['propagation_model']['receiver_sensitivity'] + PL + \
                       config['propagation_model']['link_margin']
        
        # 3e. Margines łącza (czy mamy wystarczającą moc?)
        P_TX_max = config['scenarios']['SX']['P_TX_max']
        link_margin = P_TX_max - P_TX_required
        link_margins.append(link_margin)
        
        # 3f. Energia transmisji (First Order Radio Model)
        E_TX = compute_transmission_energy(
            sensor.packet_size,
            sensor.data_rate,
            distance,
            config['energy_model']
        )
        
        # 3g. Energia odbioru (na Hub)
        E_RX = config['energy_model']['E_elec_RX'] * sensor.packet_size
        
        # 3h. Suma dla tego sensora
        E_total += E_TX + E_RX
    
    # ========================================
    # KROK 4: Kara za niską niezawodność
    # ========================================
    min_margin = min(link_margins)
    
    if min_margin < 0:
        # Łącze nie może być ustanowione - kara
        penalty_rel = abs(min_margin) * config['fitness_function']['penalties']['reliability_penalty_factor']
    else:
        penalty_rel = 0.0
    
    # ========================================
    # KROK 5: Agregacja fitness
    # ========================================
    fitness = weights['w_E'] * E_total + weights['w_R'] * penalty_rel
    
    return fitness


def compute_geometric_penalty(sensors, hub, body_zones):
    """
    Kara za naruszenia geometryczne:
    1. Sensor poza przypisaną strefą
    2. Zbyt małe odległości między sensorami (nakładanie)
    3. Hub poza dopuszczalną strefą (opcjonalnie)
    """
    penalty = 0.0
    
    # Sprawdź każdy sensor
    for sensor in sensors:
        if not Genotype.is_valid_position(sensor.position, sensor.assigned_zone, body_zones):
            penalty += 1e6  # Ogromna kara - rozwiązanie niedopuszczalne
    
    # Sprawdź kolizje między sensorami (odległość minimalna)
    MIN_DISTANCE = 0.05  # 5% w skali znormalizowanej
    for i, s1 in enumerate(sensors):
        for s2 in sensors[i+1:]:
            dist = np.linalg.norm(s1.position - s2.position)
            if dist < MIN_DISTANCE:
                penalty += 1e5 * (MIN_DISTANCE - dist)
    
    return penalty
```

---

## 4. Detekcja LOS/NLOS

### 4.1 Geometryczny Algorytm

**Założenie:** Tors (chest) reprezentowany jako cylinder o promieniu `R` i zakresie wysokości `[y_min, y_max]`.

**Reguła:**  
Jeśli linia prosta między sensorem a Hub **przecina** cylinder torsu → **NLOS**  
W przeciwnym razie → **LOS**

### 4.2 Pseudokod

```python
def detect_LOS_NLOS(sensor_pos: np.ndarray, hub_pos: np.ndarray, torso: Dict) -> str:
    """
    Detekcja stanu łącza: Line-of-Sight (LOS) lub Non-Line-of-Sight (NLOS).
    
    Metoda: Ray-cylinder intersection test
    
    Args:
        sensor_pos: [x, y] pozycja sensora
        hub_pos: [x, y] pozycja Hub
        torso: Parametry cylindra torsu {center_x, center_y, radius, height_range}
    
    Returns:
        'LOS' lub 'NLOS'
    """
    # Wyciągnij parametry torsu
    cx = torso['center_x']
    cy = torso['center_y']
    R = torso['radius']
    y_min, y_max = torso['height_range']
    
    # Wektor kierunku promienia (sensor → hub)
    ray_origin = sensor_pos
    ray_direction = hub_pos - sensor_pos
    ray_length = np.linalg.norm(ray_direction)
    ray_direction /= ray_length  # Normalizacja
    
    # Sparametryzowany promień: P(t) = ray_origin + t * ray_direction, t ∈ [0, ray_length]
    
    # ===========================================================
    # TEST 1: Czy którykolwiek z punktów jest wewnątrz cylindra?
    # ===========================================================
    def is_inside_cylinder(point):
        x, y = point
        in_height = y_min <= y <= y_max
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        return in_height and dist_from_center <= R
    
    if is_inside_cylinder(sensor_pos) or is_inside_cylinder(hub_pos):
        return 'NLOS'
    
    # ===========================================================
    # TEST 2: Czy promień przecina cylinder?
    # ===========================================================
    # Uproszczenie 2D: Cylinder → koło o promieniu R w płaszczyźnie XY
    # Sprawdzamy przecięcie linii z kołem
    
    # Odległość punktu od prostej (sensor → hub) do środka koła (cx, cy)
    # Wzór: d = |ax + by + c| / sqrt(a² + b²)
    # gdzie: ax + by + c = 0 to postać ogólna prostej
    
    # Prosta przechodząca przez sensor_pos i hub_pos:
    # (y - y1) = m(x - x1), przekształcamy do ax + by + c = 0
    
    x1, y1 = sensor_pos
    x2, y2 = hub_pos
    
    if abs(x2 - x1) < 1e-9:  # Linia pionowa
        # x = x1
        dist_to_line = abs(x1 - cx)
    else:
        m = (y2 - y1) / (x2 - x1)
        # y - y1 = m(x - x1)
        # mx - y + (y1 - m*x1) = 0
        a = m
        b = -1
        c = y1 - m * x1
        
        dist_to_line = abs(a * cx + b * cy + c) / np.sqrt(a**2 + b**2)
    
    # Jeśli odległość od środka koła do prostej < R → przecięcie
    if dist_to_line < R:
        # Dodatkowe sprawdzenie: czy punkt przecięcia jest MIĘDZY sensor i hub?
        # (nie za nimi)
        
        # Znajdujemy najbliższy punkt na linii do środka cylindra
        t_closest = np.dot(np.array([cx - x1, cy - y1]), ray_direction)
        
        if 0 <= t_closest <= ray_length:
            return 'NLOS'
    
    # ===========================================================
    # Jeśli żaden test nie wykrył NLOS → LOS
    # ===========================================================
    return 'LOS'
```

### 4.3 Wizualizacja (dla debugowania)

```python
def visualize_LOS_detection(sensor_pos, hub_pos, torso, result):
    """
    Rysuje sensor, hub, tors i linię łącza z kolorem (zielony=LOS, czerwony=NLOS).
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Tors (cylinder jako koło w 2D)
    circle = plt.Circle((torso['center_x'], torso['center_y']), 
                        torso['radius'], 
                        color='gray', alpha=0.3, label='Torso')
    ax.add_patch(circle)
    
    # Sensor i Hub
    ax.plot(*sensor_pos, 'bo', markersize=10, label='Sensor')
    ax.plot(*hub_pos, 'r*', markersize=15, label='Hub')
    
    # Linia łącza
    color = 'green' if result == 'LOS' else 'red'
    ax.plot([sensor_pos[0], hub_pos[0]], 
            [sensor_pos[1], hub_pos[1]], 
            color=color, linewidth=2, label=result)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
```

---

## 5. Model Energetyczny

### 5.1 First Order Radio Model

**Energia transmisji pakietu o rozmiarze `k` bitów na odległość `d`:**

```
E_TX(k, d) = E_elec_TX × k + E_amp × k × d^α

gdzie:
- E_elec_TX: Energia elektroniki nadajnika [J/bit]
- E_amp: Energia wzmacniacza [J/bit/m^α]
- α: Wykładnik (2 dla Free Space, 4 dla Multipath)
- d: Odległość [m]
```

**Energia odbioru:**

```
E_RX(k) = E_elec_RX × k
```

### 5.2 Implementacja

```python
def compute_transmission_energy(packet_size: int, 
                               data_rate: float,
                               distance: float, 
                               energy_params: Dict) -> float:
    """
    Oblicza energię transmisji pakietu zgodnie z First Order Radio Model.
    
    Args:
        packet_size: Rozmiar pakietu [bits]
        data_rate: Szybkość transmisji [bps]
        distance: Odległość sensor-hub [m w skali rzeczywistej]
        energy_params: Parametry energetyczne z config
    
    Returns:
        E_TX: Energia transmisji [J]
    """
    k = packet_size
    d = distance
    
    # Energia elektroniki
    E_elec = energy_params['E_elec_TX'] * k
    
    # Wybór modelu amplifikacji (Free Space vs. Multipath)
    d_threshold = energy_params['d_threshold']
    
    if d < d_threshold:
        # Free Space (d²)
        E_amp = energy_params['E_amp_fs'] * k * (d ** 2)
    else:
        # Multipath (d⁴)
        E_amp = energy_params['E_amp_mp'] * k * (d ** 4)
    
    E_TX = E_elec + E_amp
    
    return E_TX


def compute_reception_energy(packet_size: int, energy_params: Dict) -> float:
    """
    Energia odbioru pakietu.
    """
    return energy_params['E_elec_RX'] * packet_size
```

---

## 6. Model Propagacji

### 6.1 IEEE 802.15.6 Path Loss Model (CM3)

**Wzór:**

```
PL(d) [dB] = PL_d0 + 10 × n × log10(d / d0) + X_σ

gdzie:
- PL_d0: Path loss at reference distance d0 [dB]
- n: Path loss exponent (zależy od LOS/NLOS)
- d0: Reference distance (zwykle 1 m)
- X_σ: Shadowing (losowa zmienna ~ N(0, σ²)) [dB]
```

**Parametry z config (Deepak & Babu Tabela 6):**

| Typ | PL_d0 [dB] | n | σ [dB] |
|-----|-----------|---|--------|
| LOS | 35.2 | 3.11 | 6.1 |
| NLOS | 48.4 | 5.9 | 5.0 |

### 6.2 Implementacja

```python
def compute_path_loss(distance: float, 
                     los_status: str, 
                     propagation_params: Dict,
                     include_shadowing: bool = True,
                     random_seed: int = None) -> float:
    """
    Oblicza straty propagacji zgodnie z IEEE 802.15.6 CM3.
    
    Args:
        distance: Odległość [m]
        los_status: 'LOS' lub 'NLOS'
        propagation_params: Parametry z config
        include_shadowing: Czy dodać losowe cieniowanie?
        random_seed: Seed dla powtarzalności (jeśli None → losowe)
    
    Returns:
        PL: Path loss [dB]
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Wybór parametrów LOS/NLOS
    if los_status == 'LOS':
        PL_d0 = propagation_params['LOS']['PL_d0']
        n = propagation_params['LOS']['path_loss_exponent']
        sigma = propagation_params['LOS']['shadowing_std']
    else:  # NLOS
        PL_d0 = propagation_params['NLOS']['PL_d0']
        n = propagation_params['NLOS']['path_loss_exponent']
        sigma = propagation_params['NLOS']['shadowing_std']
    
    d0 = propagation_params['d0']
    
    # Path loss (deterministyczny)
    if distance < d0:
        distance = d0  # Zapobiegaj log10(0) lub ujemnym wartościom
    
    PL_deterministic = PL_d0 + 10 * n * np.log10(distance / d0)
    
    # Shadowing (losowy)
    if include_shadowing:
        X_sigma = np.random.normal(0, sigma)
        PL = PL_deterministic + X_sigma
    else:
        PL = PL_deterministic
    
    return PL


def compute_required_tx_power(path_loss: float, propagation_params: Dict) -> float:
    """
    Oblicza wymaganą moc transmisji aby osiągnąć czułość odbiornika.
    
    P_TX_req = P_sens + PL + M_safe
    
    Args:
        path_loss: Straty propagacji [dB]
        propagation_params: Parametry z config
    
    Returns:
        P_TX_required: Wymagana moc [dBm]
    """
    P_sens = propagation_params['receiver_sensitivity']  # -85 dBm
    M_safe = propagation_params['link_margin']  # 10 dB
    
    P_TX_required = P_sens + path_loss + M_safe
    
    return P_TX_required
```

---

## 7. Algorytmy Optymalizacyjne

### 7.1 Integracja z Mealpy

**Mealpy** to biblioteka implementująca 200+ algorytmów metaheurystycznych. Używamy jej zamiast pisać GA/PSO od zera.

```python
from mealpy import FloatVar, GA, PSO

def create_problem(config, scenario_name, weight_variant):
    """
    Tworzy obiekt Problem dla Mealpy.
    """
    scenario = config['scenarios'][scenario_name]
    n_sensors = scenario['n_sensors']
    dimension = 2 * n_sensors + 2
    
    # Bounds: wszystkie współrzędne w [0, 1]
    bounds = FloatVar(lb=[0.0]*dimension, ub=[1.0]*dimension)
    
    # Wrapper funkcji fitness dla Mealpy
    def fitness_wrapper(solution):
        return fitness_function(solution, config, weight_variant)
    
    problem = {
        "obj_func": fitness_wrapper,
        "bounds": bounds,
        "minmax": "min",  # Minimalizacja
        "log_to": None  # Bez logowania (ręcznie zbieramy metryki)
    }
    
    return problem


def run_GA(problem, config):
    """
    Uruchamia Genetic Algorithm.
    """
    ga_params = config['optimization']['GA']
    
    model = GA.BaseGA(
        epoch=ga_params['max_iterations'],
        pop_size=ga_params['population_size'],
        pc=ga_params['crossover']['probability'],
        pm=ga_params['mutation']['probability'],
        selection=ga_params['selection']['type'],
        k_way=ga_params['selection']['tournament_size'],
        crossover=ga_params['crossover']['type'],
        mutation=ga_params['mutation']['type']
    )
    
    # Uruchom optymalizację
    g_best = model.solve(problem)
    
    return g_best, model.history


def run_PSO(problem, config):
    """
    Uruchamia Particle Swarm Optimization.
    """
    pso_params = config['optimization']['PSO']
    
    model = PSO.OriginalPSO(
        epoch=pso_params['max_iterations'],
        pop_size=pso_params['population_size'],
        c1=pso_params['cognitive_coefficient'],
        c2=pso_params['social_coefficient'],
        w_min=0.4,  # Inertia weight decay
        w_max=pso_params['inertia_weight']
    )
    
    g_best = model.solve(problem)
    
    return g_best, model.history
```

### 7.2 Historia Zbieżności

```python
def extract_convergence_curve(history):
    """
    Wyciąga krzywą zbieżności (best fitness per iteration).
    
    Args:
        history: Obiekt history z Mealpy
    
    Returns:
        convergence: Lista wartości fitness [iter0, iter1, ..., iterN]
    """
    # Mealpy przechowuje historię w history.list_global_best_fit
    convergence = history.list_global_best_fit
    
    return convergence
```

---

## 8. Pipeline Eksperymentów

### 8.1 Główny Skrypt

```python
# experiments/run_scenarios.py

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.optimization.fitness_function import fitness_function
from src.optimization.ga_optimizer import run_GA, create_problem
from src.optimization.pso_optimizer import run_PSO
from src.baselines.random_placement import run_random_baseline
from src.baselines.naive_centroid import run_naive_baseline


def load_config(config_path='config/wban_params.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_single_experiment(scenario_name, weight_variant, algorithm, run_id, config):
    """
    Uruchamia pojedynczy eksperyment.
    
    Returns:
        results: Dict z metrykami
    """
    # Utwórz problem
    problem = create_problem(config, scenario_name, weight_variant)
    
    # Ustaw seed dla powtarzalności
    seed = 42 + run_id
    np.random.seed(seed)
    
    # Uruchom algorytm
    if algorithm == 'GA':
        g_best, history = run_GA(problem, config)
    elif algorithm == 'PSO':
        g_best, history = run_PSO(problem, config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Zbierz metryki
    best_fitness = g_best.target.fitness
    convergence_curve = extract_convergence_curve(history)
    
    # Dodatkowe metryki (dekoduj najlepsze rozwiązanie)
    sensors, hub = Genotype.decode(g_best.solution, config['scenarios'][scenario_name]['sensor_config'])
    
    E_total = compute_total_energy(sensors, hub, config)
    T_life = compute_network_lifetime(sensors, config)
    M_min = compute_min_link_margin(sensors, hub, config)
    
    results = {
        'scenario': scenario_name,
        'weight_variant': weight_variant['label'],
        'algorithm': algorithm,
        'run_id': run_id,
        'seed': seed,
        'best_fitness': best_fitness,
        'total_energy': E_total,
        'network_lifetime': T_life,
        'min_link_margin': M_min,
        'convergence_curve': convergence_curve,
        'best_solution': g_best.solution.tolist()
    }
    
    return results


def run_main_experiment(config):
    """
    Główny eksperyment: 3 scenariusze × 3 wagi × 2 algorytmy × 50 runs = 900 eksperymentów
    """
    scenarios = ['S1', 'S2', 'S3']
    weight_variants = config['fitness_function']['weight_variants'].values()
    algorithms = ['GA', 'PSO']
    n_runs = config['experiments']['main_experiment']['n_runs']
    
    all_results = []
    
    total_experiments = len(scenarios) * len(weight_variants) * len(algorithms) * n_runs
    
    with tqdm(total=total_experiments, desc="Running experiments") as pbar:
        for scenario in scenarios:
            for weight_variant in weight_variants:
                for algorithm in algorithms:
                    for run_id in range(n_runs):
                        result = run_single_experiment(
                            scenario, weight_variant, algorithm, run_id, config
                        )
                        all_results.append(result)
                        pbar.update(1)
    
    # Zapisz wyniki
    df = pd.DataFrame(all_results)
    output_dir = Path(config['output']['base_dir']) / config['output']['subdirs']['experiments']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / 'main_experiment_results.csv', index=False)
    
    print(f"✅ Saved results to {output_dir / 'main_experiment_results.csv'}")
    
    return df


if __name__ == '__main__':
    config = load_config()
    results_df = run_main_experiment(config)
    
    # Wyświetl podsumowanie
    print("\n📊 Experiment Summary:")
    print(results_df.groupby(['scenario', 'algorithm'])['best_fitness'].describe())
```

---

## 8.2 Analiza Mocy (Power Analysis)

```python
# experiments/power_analysis.py

def run_power_analysis(config):
    """
    Dodatkowy eksperyment: Testowanie różnych mocy TX dla S2.
    
    5 poziomów mocy × 50 runs × PSO = 250 eksperymentów
    """
    scenario = 'S2'
    weight_variant = config['fitness_function']['weight_variants']['balanced']
    algorithm = 'PSO'
    
    power_levels = config['experiments']['power_analysis']['P_TX_range']
    n_runs = config['experiments']['power_analysis']['n_runs']
    
    all_results = []
    
    for P_TX in power_levels:
        # Tymczasowo nadpisz P_TX_max w config
        config['scenarios'][scenario]['P_TX_max'] = P_TX
        
        for run_id in range(n_runs):
            result = run_single_experiment(scenario, weight_variant, algorithm, run_id, config)
            result['P_TX_level'] = P_TX
            all_results.append(result)
    
    # Zapisz wyniki
    df = pd.DataFrame(all_results)
    output_dir = Path(config['output']['base_dir']) / config['output']['subdirs']['experiments']
    df.to_csv(output_dir / 'power_analysis_results.csv', index=False)
    
    print(f"✅ Saved power analysis to {output_dir}")
    
    return df
```

---

## 9. Diagram Przepływu Systemu

```
┌─────────────────────────────────────────────────────────────┐
│                    WBAN OPTIMIZATION SYSTEM                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Load Config     │
                    │  (YAML)          │
                    └──────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │  Generate Experiment Matrix         │
            │  - Scenarios (S1, S2, S3)          │
            │  - Weight variants (3)              │
            │  - Algorithms (GA, PSO)             │
            │  - Runs (50)                        │
            └─────────────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │  FOR EACH Configuration:            │
            └─────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
                ▼                            ▼
        ┌───────────────┐          ┌───────────────┐
        │   Run GA      │          │   Run PSO     │
        │   (Mealpy)    │          │   (Mealpy)    │
        └───────────────┘          └───────────────┘
                │                            │
                └─────────────┬──────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Fitness         │
                    │  Evaluation      │
                    └──────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    ┌────────────┐   ┌────────────┐   ┌────────────┐
    │ Decode     │   │ Validate   │   │ Compute    │
    │ Genotype   │   │ Geometry   │   │ Energy     │
    └────────────┘   └────────────┘   └────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                    ┌──────────────────┐
                    │  FOR EACH Sensor:│
                    └──────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    ┌────────────┐   ┌────────────┐   ┌────────────┐
    │ Compute    │   │ Detect     │   │ Compute    │
    │ Distance   │   │ LOS/NLOS   │   │ Path Loss  │
    └────────────┘   └────────────┘   └────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                    ┌──────────────────┐
                    │ Compute TX Power │
                    │ & Energy         │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Aggregate       │
                    │  Fitness =       │
                    │  w_E*E + w_R*P   │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Return to       │
                    │  Algorithm       │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Collect Metrics │
                    │  - Best fitness  │
                    │  - Convergence   │
                    │  - Energy        │
                    │  - Lifetime      │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Save Results    │
                    │  (CSV + Plots)   │
                    └──────────────────┘
```

---

## 10. Następne Kroki

### ✅ Gotowe:
- [x] Kompletna konfiguracja (YAML)
- [x] Szczegółowa specyfikacja techniczna (ten dokument)

### 🔄 Do zrobienia:
1. **Implementacja modułów core** (Dzień 2-3)
   - `body_model.py`
   - `energy_model.py`
   - `propagation_model.py`
   - `los_detector.py`
   - `fitness_function.py`

2. **Integracja z Mealpy** (Dzień 3)
   - `ga_optimizer.py`
   - `pso_optimizer.py`

3. **Baselines** (Dzień 3)
   - `random_placement.py`
   - `naive_centroid.py`

4. **Pipeline eksperymentów** (Dzień 4)
   - `run_scenarios.py`
   - `power_analysis.py`
   - `collect_metrics.py`

5. **Wizualizacje** (Dzień 4-5)
   - Wszystkie 7 wykresów

6. **Testy i walidacja** (Dzień 5)
   - Unit testy dla każdego modułu
   - Weryfikacja poprawności fitness function

7. **Uruchomienie pełnych eksperymentów** (Dzień 6-7)
   - 1150 eksperymentów
   - Analiza statystyczna (Wilcoxon rank-sum test)

8. **Rozszerzenia LaTeX** (Dzień 8)
   - Wstawki do rozdziałów 2, 3, 4, 5
   - Opisy wykresów i tabel

---

**Koniec specyfikacji technicznej v1.0**
