# WBAN Sensor Placement Optimization System

**Autor:** Kamil Piejko  
**Politechnika Rzeszowska, 2025**

Optymalizacja rozmieszczenia sensorów w sieciach WBAN (Wireless Body Area Networks) przy użyciu algorytmów nature-inspired (GA, PSO).

---

## 📂 Struktura Projektu

```
wban_optimization/
├── config/
│   └── wban_params.yaml          # ✅ Główna konfiguracja systemu
├── docs/
│   ├── technical_specification.md # ✅ Szczegółowa dokumentacja techniczna
│   └── experiment_matrix.csv      # ✅ Matryca eksperymentów
├── src/                           # 🔄 Implementacja (następny krok)
│   ├── core/
│   ├── models/
│   ├── optimization/
│   ├── baselines/
│   └── utils/
├── experiments/                   # 🔄 Skrypty uruchamiające
├── visualization/                 # 🔄 Generowanie wykresów
└── results/                       # Wyniki eksperymentów
    ├── experiments/
    ├── plots/
    └── statistics/
```

**Legenda:**
- ✅ Gotowe
- 🔄 Do implementacji
- ⏳ Oczekuje na uruchomienie

---

## 🚀 Quickstart (Po Implementacji)

### 1. Instalacja Zależności

```bash
pip install numpy pandas matplotlib seaborn pyyaml mealpy tqdm scipy
```

### 2. Sprawdzenie Konfiguracji

```bash
python -c "import yaml; print(yaml.safe_load(open('config/wban_params.yaml')))"
```

### 3. Uruchomienie Głównego Eksperymentu

```bash
# Pełny eksperyment (18 konfiguracji × 50 runs = 900 eksperymentów)
python experiments/run_scenarios.py

# Szacowany czas: ~6-8 godzin na standardowym laptopie (4 rdzenie)
```

### 4. Analiza Mocy

```bash
# Dodatkowy eksperyment: testowanie różnych P_TX
python experiments/power_analysis.py

# Szacowany czas: ~2 godziny
```

### 5. Generowanie Wizualizacji

```bash
# Wszystkie wykresy
python visualization/generate_all_plots.py

# Pojedyncze wykresy
python visualization/plot_convergence.py
python visualization/plot_placement.py
```

---

## 📊 Scenariusze Eksperymentalne

| Scenario | N | P_TX | Opis |
|----------|---|------|------|
| **S1** | 6 | 0 dBm | Sparse/Robust - podstawowe monitorowanie, duże odległości |
| **S2** | 15 | -5 dBm | Balanced/Advanced - rozszerzone monitorowanie |
| **S3** | 25 | -10 dBm | Dense/Efficiency - pełny monitoring, gęsta sieć |

**Uzasadnienie odwrotnej logiki mocy:**
- **S1 (rzadki):** Duże odległości → wysoka moc potrzebna do pokonania body shadowing
- **S3 (gęsty):** Krótkie odległości → niska moc wystarczająca, redukcja interferencji

---

## 🎯 Warianty Funkcji Celu

| Wariant | w_E | w_R | Zastosowanie |
|---------|-----|-----|--------------|
| **Energy-Priority** | 0.7 | 0.3 | Długoterminowy monitoring (żywotność baterii priorytetem) |
| **Balanced** | 0.5 | 0.5 | Standardowe działanie |
| **Reliability-Priority** | 0.3 | 0.7 | Aplikacje krytyczne (EKG, wykrywanie zawału) |

---

## 📈 Metryki Ewaluacyjne

### Podstawowe:
- **Fitness Value** (F): Wartość funkcji celu (niższa = lepsza)
- **Total Energy** (E_total): Całkowita energia zużyta [mJ]
- **Network Lifetime** (T_life): Czas do FND (First Node Dies) [rounds]
- **Min Link Margin** (M_min): Minimalny margines łącza [dB]

### Dodatkowe:
- Throughput [packets/round]
- Propagation Delay [ms]
- Packet Loss Rate [%]
- LOS/NLOS Ratio

---

## 📊 Wizualizacje

### 1. Energy vs. Number of Sensors
**Plik:** `results/plots/energy_vs_sensors.png`  
Porównanie GA, PSO, Random dla scenariuszy S1, S2, S3

### 2. Network Lifetime vs. Number of Sensors
**Plik:** `results/plots/lifetime_vs_sensors.png`  
Wpływ optymalizacji na czas życia sieci

### 3. Convergence Curves (GA vs PSO)
**Plik:** `results/plots/convergence_S2_balanced.png`  
Krzywe zbieżności dla scenariusza S2 (wagi 0.5/0.5)

### 4. Sensor Placement Visualization
**Plik:** `results/plots/placement_best_S1.png`  
Najlepsze rozwiązanie na modelu ciała z zaznaczonymi łączami LOS/NLOS

### 5. Energy Distribution per Sensor
**Plik:** `results/plots/energy_distribution_best.png`  
Rozkład zużycia energii między sensorami

### 6. Propagation Delay vs. Sensors
**Plik:** `results/plots/delay_vs_sensors.png`

### 7. Power Sensitivity Analysis
**Plik:** `results/plots/power_sensitivity_boxplot.png`  
Box plot dla różnych poziomów P_TX ∈ {-10, -5, 0, +3, +5} dBm

---

## 🔬 Parametry z Literatury

### Energia (First Order Radio Model)
**Źródło:** [11] Al-Mishmish et al. 2018, [13] Ajmi et al. 2021

```yaml
E_elec_TX: 50 nJ/bit
E_elec_RX: 50 nJ/bit
E_amp_fs: 10 pJ/bit/m²
E_init: 0.5 J
```

### Propagacja (IEEE 802.15.6 CM3)
**Źródło:** [19] Deepak & Babu 2016, Tabela 6

| Typ | PL_d0 [dB] | n | σ [dB] |
|-----|-----------|---|--------|
| LOS | 35.2 | 3.11 | 6.1 |
| NLOS | 48.4 | 5.9 | 5.0 |

### Strefy Anatomiczne
**Źródło:** [10] Alam & Ben Hamida 2014, Tabela 16

- **ECG:** Chest (Low Mobility)
- **EEG:** Head (Low Mobility)
- **SpO2:** Shoulder/Wrist (Static/Low-to-High)
- **Accelerometer:** Legs/Arms (High Mobility)

---

## 📝 Użycie w Pracy Dyplomowej

### Rozdziały:

#### **Rozdział 2: Model Matematyczny**
- Sekcja 2.2: Model energetyczny → użyj parametrów z `config/wban_params.yaml`
- Sekcja 2.3: Model propagacji → cytuj Tabela 6 z [19]
- Sekcja 2.4: Funkcja celu → użyj pseudokodu z `docs/technical_specification.md`

#### **Rozdział 3: Algorytmy**
- Tabela 3.1: Parametry GA/PSO → skopiuj z YAML
- Sekcja 3.2: Kodowanie genotypu → wyjaśnij strukturę [x1,y1,...,xN,yN,x_hub,y_hub]

#### **Rozdział 4: Eksperyment**
- Tabela 4.1: Scenariusze → użyj tabeli z README
- Sekcja 4.2: Matryca eksperymentów → załącz `docs/experiment_matrix.csv`

#### **Rozdział 5: Wyniki**
- Sekcja 5.1: Porównanie GA vs PSO → użyj wykresów 1, 2, 3
- Sekcja 5.2: Wpływ wag → porównaj Energy-Priority vs Reliability-Priority
- Sekcja 5.3: Analiza wrażliwości → użyj wykresu 7 (Power Sensitivity)
- Sekcja 5.4: Wizualizacja najlepszych rozwiązań → wykres 4

---

## 🛠️ Rozszerzenia (Opcjonalne)

### 1. Dodatkowe Algorytmy
```python
# Przykład: Whale Optimization Algorithm (używany w [6])
from mealpy import WOA

def run_WOA(problem, config):
    model = WOA.OriginalWOA(
        epoch=100,
        pop_size=50
    )
    return model.solve(problem)
```

### 2. Multi-Objective Optimization
```python
# NSGA-II dla frontu Pareto (E_total vs T_life)
from pymoo.algorithms.moo.nsga2 import NSGA2

# Zwróć wektor celów zamiast skalarnej sumy
def fitness_multi_objective(genome):
    return [E_total, -T_life]  # Minimalizacja obu
```

### 3. Dynamiczny Ruch Sensorów
```python
# Symulacja mobilności (dla High Mobility sensors)
def update_sensor_position(sensor, time_step):
    if sensor.mobility == 'high':
        sensor.position += random_walk(step_size=0.01)
```

---

## 📚 Bibliografia Kluczowych Parametrów

**[19]** Deepak, K. K., & Babu, A. V. (2016). *Energy consumption analysis of modulation schemes in IEEE 802.15.6 based wireless body area networks.* ICCS 2016.
- **Użyte:** Tabela 6 (propagacja), Tabela 7 (energia)

**[11]** Al-Mishmish, H., et al. (2018). *Critical Data-Based Incremental Cooperative Communication for Wireless Body Area Network.* Sensors.
- **Użyte:** Parametry energetyczne (str. 12)

**[13]** Ajmi, N., et al. (2021). *MWCSGA: Multi-Weight Chicken Swarm Based Genetic Algorithm.* Sensors.
- **Użyte:** Tabela 1 (kompletne parametry symulacji)

**[10]** Alam, M. M., & Ben Hamida, E. (2014). *Surveying WBAN for IoT Domains.* MDPIElectronics.
- **Użyte:** Tabela 16 (lokalizacje sensorów)

---

## 🤝 Wsparcie i Kontakt

**Autor:** Kamil Piejko  
**Email:** [twój email]  
**Promotor:** [imię promotora]

---

## 📄 Licencja

Kod źródłowy: MIT License  
Praca dyplomowa: © Politechnika Rzeszowska 2025

---

## ✅ Checklist Implementacji

### Faza 1: Core Modules (Dzień 2-3)
- [ ] `src/core/body_model.py`
- [ ] `src/core/sensor.py`
- [ ] `src/core/genotype.py`
- [ ] `src/models/energy_model.py`
- [ ] `src/models/propagation_model.py`
- [ ] `src/models/los_detector.py`

### Faza 2: Optimization (Dzień 3)
- [ ] `src/optimization/fitness_function.py`
- [ ] `src/optimization/ga_optimizer.py`
- [ ] `src/optimization/pso_optimizer.py`
- [ ] `src/baselines/random_placement.py`
- [ ] `src/baselines/naive_centroid.py`

### Faza 3: Experiments (Dzień 4)
- [ ] `experiments/run_scenarios.py`
- [ ] `experiments/power_analysis.py`
- [ ] `experiments/collect_metrics.py`

### Faza 4: Visualization (Dzień 4-5)
- [ ] `visualization/plot_energy_vs_sensors.py`
- [ ] `visualization/plot_lifetime_vs_sensors.py`
- [ ] `visualization/plot_convergence.py`
- [ ] `visualization/plot_placement.py`
- [ ] `visualization/plot_energy_distribution.py`
- [ ] `visualization/plot_delay_vs_sensors.py`
- [ ] `visualization/plot_power_sensitivity.py`

### Faza 5: Testing & Validation (Dzień 5)
- [ ] Unit tests dla każdego modułu
- [ ] Integration tests
- [ ] Weryfikacja fitness function (hand-calculated examples)

### Faza 6: Execution (Dzień 6-7)
- [ ] Uruchomienie głównego eksperymentu (900 runs)
- [ ] Uruchomienie power analysis (250 runs)
- [ ] Weryfikacja wyników (sanity checks)

### Faza 7: Analysis & LaTeX (Dzień 8)
- [ ] Analiza statystyczna (Wilcoxon rank-sum)
- [ ] Generowanie wszystkich wykresów
- [ ] Rozszerzenia do rozdziałów LaTeX
- [ ] Tabele wyników

---

**Status:** ✅ Dokumentacja gotowa | 🔄 Implementacja w toku | ⏳ Oczekuje na uruchomienie

**Ostatnia aktualizacja:** 2024-12-08
