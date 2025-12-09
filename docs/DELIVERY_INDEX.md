# 📦 WBAN Optimization - Complete Documentation Index

**Project:** Optymalizacja rozmieszczania elementów w sieciach IoMT/WBAN  
**Author:** Kamil Piejko  
**Institution:** Politechnika Rzeszowska  
**Date:** December 2024

---

## ✅ DELIVERED - Pakiet Dokumentacji

### 🎯 Pakiet 1: Konfiguracja Systemu

#### 📄 `config/wban_params.yaml` (18 KB)
**Pełna konfiguracja systemu w formacie YAML**

**Zawartość:**
- ✅ Model ciała - 11 stref anatomicznych (chest, head, arms, wrists, waist, legs)
- ✅ Typy sensorów - 7 typów (ECG, EEG, SpO2, Temperature, Accelerometer, Gyroscope, Heart_Rate)
- ✅ Parametry energetyczne (z literatury [11], [13])
  - E_elec_TX/RX: 50 nJ/bit
  - E_amp_fs: 10 pJ/bit/m²
  - E_init: 0.5 J
- ✅ Parametry propagacji (z [19] Deepak & Babu, Tabela 6)
  - LOS: PL_d0=35.2 dB, n=3.11, σ=6.1 dB
  - NLOS: PL_d0=48.4 dB, n=5.9, σ=5.0 dB
- ✅ 3 scenariusze eksperymentalne (S1, S2, S3)
  - S1: 6 sensors, 0 dBm (Sparse/Robust)
  - S2: 15 sensors, -5 dBm (Balanced)
  - S3: 25 sensors, -10 dBm (Dense/Efficiency)
- ✅ Konfiguracja algorytmów (GA, PSO)
- ✅ Funkcja celu - 3 warianty wag
- ✅ Plan eksperymentów (główny + power analysis)
- ✅ Metryki i wizualizacje

**Użycie:**
```python
import yaml
config = yaml.safe_load(open('config/wban_params.yaml'))
```

---

### 📚 Pakiet 2: Dokumentacja Techniczna

#### 📄 `docs/technical_specification.md` (34 KB)
**Szczegółowa specyfikacja techniczna - 10 sekcji**

**Zawartość:**

**1. Architektura Systemu**
- Struktura katalogów projektu
- Moduły: core, models, optimization, baselines, experiments, visualization

**2. Model Danych**
- Klasa `Sensor` (12 atrybutów + metody)
- Klasa `Hub`
- Klasa `Genotype` (kodowanie/dekodowanie)

**3. Funkcja Fitness - SZCZEGÓŁOWA SPECYFIKACJA**
- Kompletny pseudokod (50+ linii)
- Krok po kroku:
  1. Dekodowanie genotypu
  2. Walidacja geometryczna
  3. Dla każdego sensora: odległość → LOS/NLOS → path loss → energia
  4. Kara za niezawodność
  5. Agregacja
- Funkcja `compute_geometric_penalty`

**4. Detekcja LOS/NLOS**
- Algorytm ray-cylinder intersection
- Pseudokod (30+ linii)
- Wizualizacja (matplotlib)

**5. Model Energetyczny**
- First Order Radio Model
- Implementacja `compute_transmission_energy`
- Free Space vs. Multipath

**6. Model Propagacji**
- IEEE 802.15.6 CM3
- Implementacja `compute_path_loss`
- Shadowing (losowe cieniowanie)

**7. Algorytmy Optymalizacyjne**
- Integracja z Mealpy
- `run_GA()` - Genetic Algorithm
- `run_PSO()` - Particle Swarm Optimization
- Ekstrakcja krzywych zbieżności

**8. Pipeline Eksperymentów**
- `run_scenarios.py` - główny skrypt (900 eksperymentów)
- `power_analysis.py` - analiza mocy (250 eksperymentów)
- Zbieranie metryk

**9. Diagram Przepływu**
- ASCII flowchart całego systemu

**10. Następne Kroki**
- Checklist implementacji (8 etapów)
- Timeline (Dzień 1-8)

**Użycie:**
To jest BIBLIA projektu - przed napisaniem kodu, przeczytaj ten dokument!

---

#### 📄 `docs/architecture_diagrams.md` (12 KB)
**7 diagramów Mermaid + Gantt Chart**

**Diagramy:**

1. **High-Level System Architecture** (graf modułów)
2. **Fitness Function Detailed Flow** (flowchart 20+ kroków)
3. **LOS/NLOS Detection Algorithm** (decision tree)
4. **Experiment Execution Flow** (sequence diagram)
5. **Data Model Class Diagram** (UML 9 klas)
6. **Visualization Pipeline** (data flow)
7. **Deployment Timeline** (Gantt chart 5 faz)

**Jak użyć:**
```bash
# Skopiuj diagram do https://mermaid.live/
# Eksportuj jako PNG/SVG

# Włóż do LaTeX:
\includegraphics[width=0.8\textwidth]{figures/system_architecture.png}
```

**Przykład:**
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/fitness_flow.png}
\caption{Szczegółowy algorytm funkcji fitness (Eq. 2.6). 
Pokazano proces dekodowania genotypu, walidację geometryczną, 
obliczanie energii dla każdego sensora oraz agregację kar.}
\label{fig:fitness_flow}
\end{figure}
```

---

#### 📄 `docs/experiment_matrix.csv` (2.5 KB)
**Matryca 23 konfiguracji eksperymentów**

**Format CSV:**
```
Exp_ID,Category,Scenario,N_Sensors,P_TX_dBm,w_E,w_R,Weight_Label,Algorithm,N_Runs,Est_Time_min,Priority,Status
E001,Main,S1,6,0.0,0.7,0.3,Energy-Priority,GA,50,15,HIGH,TODO
...
E023,PowerAnalysis,S2,15,5.0,0.5,0.5,Balanced,PSO,50,20,LOW,TODO
```

**Podsumowanie:**
- Główny eksperyment: 18 konfiguracji × 50 runs = **900 eksperymentów**
- Power analysis: 5 konfiguracji × 50 runs = **250 eksperymentów**
- **TOTAL: 1150 eksperymentów**
- Szacowany czas: ~530 godzin CPU (5-6 dni wall-clock z 4 rdzeniami)

**Użycie:**
```python
import pandas as pd
matrix = pd.read_csv('docs/experiment_matrix.csv')
matrix[matrix['Priority'] == 'HIGH']  # Priorytetowe eksperymenty
```

---

#### 📄 `docs/fitness_calculation_examples.md` (11 KB)
**3 ręcznie obliczone przykłady**

**Przykład 1: Rozwiązanie VALID (3 sensory)**
- Genotyp: [0.45, 0.65, 0.18, 0.50, 0.50, 0.62, 0.50, 0.45]
- Walidacja: ✅ Wszystkie w strefach
- LOS/NLOS: NLOS (sensor w cylindrze torsu)
- Path Loss: ~10 dB (mała odległość)
- Energia: E_total = 17 μJ
- Fitness: **F = 0.0085** (niska = dobra)

**Przykład 2: Rozwiązanie INVALID**
- Sensor poza strefą → P_geo = 1e6
- Fitness: **F = 1,000,000** (odrzucone)

**Przykład 3: Rozwiązanie z Niskim Marginesem**
- Duża odległość (d=2.0 m) + niska moc (-10 dBm)
- Margines: M = -9.2 dB (ujemny!)
- Kara: P_rel = 920
- Fitness: **F = 460** (słabe rozwiązanie)

**Wnioski:**
- LOS vs NLOS: różnica 13 dB = 20× więcej mocy
- Inverse Power Scheme: S3 wymusza d < 0.5 m, S1 pozwala d < 1.5 m

**Użycie:**
To są TEST CASES do weryfikacji implementacji:
```python
def test_valid_solution():
    genome = np.array([0.45, 0.65, 0.18, 0.50, 0.50, 0.62, 0.50, 0.45])
    fitness = fitness_function(genome, config, weights)
    assert 0.005 < fitness < 0.030  # Expected ~0.0085
```

---

### 📖 Pakiet 3: Instrukcje Użytkownika

#### 📄 `README.md` (9.4 KB)
**Kompletny przewodnik po projekcie**

**Sekcje:**

1. **Quickstart** - instalacja, uruchamianie
2. **Scenariusze** - tabela S1, S2, S3 z uzasadnieniem
3. **Warianty funkcji celu** - Energy vs Reliability
4. **Metryki** - podstawowe i dodatkowe
5. **Wizualizacje** - lista 7+ wykresów
6. **Parametry z literatury** - źródła dla każdej wartości
7. **Użycie w pracy dyplomowej** - które rozdziały, które tabele
8. **Rozszerzenia opcjonalne** - WOA, NSGA-II, dynamiczny ruch
9. **Bibliografia** - 4 kluczowe źródła z konkretnymi tabelami
10. **Checklist implementacji** - 7 faz, 30+ zadań

**Kluczowe fragmenty:**

**Uruchomienie:**
```bash
python experiments/run_scenarios.py  # 6-8h
python experiments/power_analysis.py  # 2h
python visualization/generate_all_plots.py
```

**Dla pracy dyplomowej:**
```latex
% Rozdział 2.2: Model energetyczny
Przyjęto parametry zgodnie z literaturą [11], [13]:
- E_elec_TX = 50 nJ/bit (elektronika nadajnika)
- E_amp_fs = 10 pJ/bit/m² (wzmacniacz - Free Space)
- E_init = 0.5 J (początkowa energia baterii)

% Tabela 2.1: Parametry energetyczne
\begin{table}[htbp]
\centering
\caption{Parametry modelu energetycznego First Order Radio}
\label{tab:energy_params}
\begin{tabular}{lcc}
\toprule
Parametr & Wartość & Źródło \\
\midrule
$E_{elec,TX}$ & 50 nJ/bit & [11] \\
$E_{elec,RX}$ & 50 nJ/bit & [11] \\
$E_{amp,fs}$ & 10 pJ/bit/m$^2$ & [13] \\
$E_{init}$ & 0.5 J & [13] \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 📊 PODSUMOWANIE PAKIETU

### Statystyki:

| Plik | Rozmiar | Linie | Sekcje | Cel |
|------|---------|-------|--------|-----|
| `wban_params.yaml` | 18 KB | ~500 | 12 | Kompletna konfiguracja systemu |
| `technical_specification.md` | 34 KB | ~1000 | 10 | Specyfikacja techniczna + pseudokod |
| `architecture_diagrams.md` | 12 KB | ~400 | 7 | Diagramy Mermaid |
| `experiment_matrix.csv` | 2.5 KB | 24 | 1 | Matryca eksperymentów |
| `fitness_calculation_examples.md` | 11 KB | ~350 | 3 | Przykłady obliczeniowe |
| `README.md` | 9.4 KB | ~300 | 10 | Przewodnik użytkownika |

**TOTAL:** ~87 KB dokumentacji, ~2550 linii, 43 sekcje

---

## 🚀 CO DALEJ - TWOJE DZIAŁANIA

### Priorytet 1: Przejrzyj Dokumentację (1-2h)

1. **Przeczytaj README.md** (10 min) - zrozum ogólny obraz
2. **Przejrzyj wban_params.yaml** (15 min) - zapoznaj się z konfiguracją
3. **Studiuj technical_specification.md** (30-45 min) - to jest KLUCZ do implementacji
4. **Zrozum fitness_calculation_examples.md** (15 min) - jak działa funkcja celu

### Priorytet 2: Zatwierdź Konfigurację

Sprawdź w YAML, czy wszystko się zgadza:
- [ ] Strefy anatomiczne (body_zones) - czy współrzędne mają sens?
- [ ] Parametry energetyczne - czy wartości z literatury są OK?
- [ ] Parametry propagacji - czy zgadzają się z Tabelą 6 z [19]?
- [ ] Scenariusze S1, S2, S3 - czy liczby sensorów są finalne?

**Jeśli coś trzeba zmienić → edytuj YAML i daj mi znać.**

### Priorytet 3: Przygotuj Środowisko (30 min)

```bash
# Utwórz środowisko wirtualne
python -m venv venv_wban
source venv_wban/bin/activate  # Linux/Mac
# venv_wban\Scripts\activate  # Windows

# Instaluj zależności
pip install numpy pandas matplotlib seaborn pyyaml mealpy tqdm scipy

# Sprawdź konfigurację
python -c "import yaml; print(yaml.safe_load(open('config/wban_params.yaml'))['scenarios'])"
```

### Priorytet 4: Decyzja - Czy Zaczynamy Implementację?

**Opcja A:** ✅ Wszystko OK → **START IMPLEMENTACJI** (następny pakiet: kod źródłowy)

**Opcja B:** ⚠️ Coś do poprawki → **KOREKTY** (podaj listę zmian)

---

## 📞 NASTĘPNE PAKIETY (po Twoim zatwierdzeniu)

### Pakiet 4: Core Implementation (2-3 dni)
- `src/core/body_model.py`
- `src/models/energy_model.py`
- `src/models/propagation_model.py`
- `src/models/los_detector.py`
- `src/optimization/fitness_function.py`

### Pakiet 5: Optimization & Baselines (1 dzień)
- `src/optimization/ga_optimizer.py`
- `src/optimization/pso_optimizer.py`
- `src/baselines/random_placement.py`
- `src/baselines/naive_centroid.py`

### Pakiet 6: Experiments Pipeline (1 dzień)
- `experiments/run_scenarios.py`
- `experiments/power_analysis.py`
- `experiments/collect_metrics.py`

### Pakiet 7: Visualization (1 dzień)
- `visualization/plot_*.py` (7+ skryptów)

### Pakiet 8: LaTeX Extensions (1 dzień)
- Rozszerzenia do Rozdziałów 2, 3, 4, 5
- Tabele wyników
- Opisy wykresów

---

## ✅ CHECKLIST ZATWIERDZENIA

Przed rozpoczęciem implementacji, potwierdź:

- [ ] Przeczytałem całą dokumentację
- [ ] Rozumiem architekturę systemu
- [ ] Parametry w YAML są poprawne (lub podałem listę zmian)
- [ ] Scenariusze S1, S2, S3 są finalne
- [ ] Rozumiem działanie funkcji fitness
- [ ] Mam zainstalowane środowisko Python
- [ ] Jestem gotowy na implementację

**Gdy zaznaczysz wszystkie punkty → daj mi znać, a dostarczę Pakiet 4 (implementację)!**

---

## 📧 KONTAKT

**Status:** ✅ Dokumentacja gotowa  
**Oczekuję na:** Twoje zatwierdzenie konfiguracji  
**Następny krok:** Implementacja modułów core

**Pytania? Zmiany? Daj znać!**

---

**Ostatnia aktualizacja:** 2024-12-08
