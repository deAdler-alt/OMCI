# WBAN Optimization - Fitness Function Test Example
# 
# Ten plik zawiera ręcznie obliczony przykład działania funkcji fitness
# dla prostego scenariusza (3 sensory + 1 Hub), aby zweryfikować poprawność implementacji.

## ========================================================================
## PRZYKŁAD: Scenariusz "Toy Example" (3 sensory)
## ========================================================================

### Konfiguracja:
- **Scenariusz:** 3 sensory (ECG, SpO2, Temperature) + 1 Hub
- **Wagi:** w_E = 0.5, w_R = 0.5 (balanced)
- **P_TX_max:** 0 dBm

### Genotyp (pozycje):
```
g = [0.45, 0.65,  ← Sensor 1 (ECG) w strefie 'chest' [0.35-0.65, 0.50-0.75]
     0.18, 0.50,  ← Sensor 2 (SpO2) w strefie 'left_wrist' [0.10-0.20, 0.45-0.55]
     0.50, 0.62,  ← Sensor 3 (Temperature) w strefie 'chest'
     0.50, 0.45]  ← Hub w strefie 'waist' [0.40-0.60, 0.40-0.50]
```

---

## KROK 1: Walidacja Geometryczna

### Sprawdzenie przynależności do stref:

**Sensor 1 (ECG):** (0.45, 0.65)
- Assigned zone: 'chest' → x ∈ [0.35, 0.65], y ∈ [0.50, 0.75]
- Check: 0.35 ≤ 0.45 ≤ 0.65 ✅, 0.50 ≤ 0.65 ≤ 0.75 ✅
- **Wynik:** VALID

**Sensor 2 (SpO2):** (0.18, 0.50)
- Assigned zone: 'left_wrist' → x ∈ [0.10, 0.20], y ∈ [0.45, 0.55]
- Check: 0.10 ≤ 0.18 ≤ 0.20 ✅, 0.45 ≤ 0.50 ≤ 0.55 ✅
- **Wynik:** VALID

**Sensor 3 (Temperature):** (0.50, 0.62)
- Assigned zone: 'chest' → x ∈ [0.35, 0.65], y ∈ [0.50, 0.75]
- Check: 0.35 ≤ 0.50 ≤ 0.65 ✅, 0.50 ≤ 0.62 ≤ 0.75 ✅
- **Wynik:** VALID

**Hub:** (0.50, 0.45)
- Preferred zone: 'waist' → x ∈ [0.40, 0.60], y ∈ [0.40, 0.50]
- Check: 0.40 ≤ 0.50 ≤ 0.60 ✅, 0.40 ≤ 0.45 ≤ 0.50 ✅
- **Wynik:** VALID

**P_geo = 0** (wszystkie sensory w swoich strefach) ✅

---

## KROK 2: Dla Każdego Sensora

### 2.1 Sensor 1 (ECG)

#### 2.1.1 Odległość do Hub:
```
d1 = sqrt((0.45 - 0.50)² + (0.65 - 0.45)²)
   = sqrt((-0.05)² + (0.20)²)
   = sqrt(0.0025 + 0.04)
   = sqrt(0.0425)
   = 0.206 m (w skali znormalizowanej → rzeczywista odległość ~0.20 m)
```

#### 2.1.2 Detekcja LOS/NLOS:
**Cylinder torsu:** center = (0.50, 0.60), radius = 0.15

**Ray:** from (0.45, 0.65) to (0.50, 0.45)
- Direction: (0.50 - 0.45, 0.45 - 0.65) = (0.05, -0.20)
- Normalized: (0.05, -0.20) / 0.206 = (0.243, -0.971)

**Test:** Czy którykolwiek punkt jest w cylindrze?
- Sensor 1: dist_to_center = sqrt((0.45-0.50)² + (0.65-0.60)²) = sqrt(0.0025 + 0.0025) = 0.071 < 0.15
  → **Sensor 1 jest WEWNĄTRZ cylindra!**
  
**Wynik:** **NLOS** (ciało jest na linii)

#### 2.1.3 Path Loss (NLOS):
Parametry z config:
- PL_d0 = 48.4 dB (NLOS)
- n = 5.9
- d0 = 1.0 m
- sigma = 5.0 dB

```
PL_deterministic = PL_d0 + 10 × n × log10(d / d0)
                 = 48.4 + 10 × 5.9 × log10(0.206 / 1.0)
                 = 48.4 + 59 × log10(0.206)
                 = 48.4 + 59 × (-0.686)
                 = 48.4 - 40.47
                 = 7.93 dB  ← (bardzo mała odległość!)

# W rzeczywistości dodajmy shadowing (załóżmy X_sigma = +2 dB dla tego przykładu)
PL_actual = 7.93 + 2 = 9.93 dB ≈ 10 dB
```

#### 2.1.4 Wymagana Moc TX:
```
P_TX_required = P_sens + PL + M_safe
              = -85 dBm + 10 dB + 10 dB
              = -65 dBm
```

#### 2.1.5 Margines Łącza:
```
M1 = P_TX_max - P_TX_required
   = 0 dBm - (-65 dBm)
   = +65 dB  ← Bardzo duży margines! ✅
```

#### 2.1.6 Energia Transmisji:
Parametry z config:
- E_elec_TX = 50 nJ/bit
- E_amp_fs = 10 pJ/bit/m²
- d_threshold = 0.5 m
- packet_size = 100 bits (ECG)

```
d1 = 0.206 m < d_threshold (0.5 m) → używamy Free Space (d²)

E_TX1 = E_elec_TX × k + E_amp_fs × k × d²
      = 50e-9 × 100 + 10e-12 × 100 × (0.206)²
      = 5.0e-6 + 10e-10 × 0.0424
      = 5.0e-6 + 4.24e-11
      ≈ 5.0 μJ (elektronika dominuje, amplifikacja znikoma dla małej odległości)
```

---

### 2.2 Sensor 2 (SpO2)

#### 2.2.1 Odległość:
```
d2 = sqrt((0.18 - 0.50)² + (0.50 - 0.45)²)
   = sqrt((-0.32)² + (0.05)²)
   = sqrt(0.1024 + 0.0025)
   = sqrt(0.1049)
   = 0.324 m
```

#### 2.2.2 LOS/NLOS:
**Ray:** from (0.18, 0.50) to (0.50, 0.45)
- Sensor 2 jest daleko od cylindra (left_wrist, poza torsem)
- Sprawdźmy przecięcie linii z cylindrem...

**Uproszczenie dla przykładu:** Załóżmy, że linia PRZECINA tors → **NLOS**
(w pełnej implementacji użyjemy ray-cylinder intersection test)

#### 2.2.3 Path Loss (NLOS):
```
PL_deterministic = 48.4 + 59 × log10(0.324 / 1.0)
                 = 48.4 + 59 × log10(0.324)
                 = 48.4 + 59 × (-0.489)
                 = 48.4 - 28.85
                 = 19.55 dB

# Z shadowingiem (załóżmy +3 dB)
PL_actual = 19.55 + 3 = 22.55 dB ≈ 23 dB
```

#### 2.2.4 Wymagana Moc:
```
P_TX_required = -85 + 23 + 10 = -52 dBm
```

#### 2.2.5 Margines:
```
M2 = 0 - (-52) = +52 dB ✅
```

#### 2.2.6 Energia:
```
d2 = 0.324 m < 0.5 m → Free Space
packet_size = 50 bits (SpO2)

E_TX2 = 50e-9 × 50 + 10e-12 × 50 × (0.324)²
      = 2.5e-6 + 5.25e-10
      ≈ 2.5 μJ
```

---

### 2.3 Sensor 3 (Temperature)

#### 2.3.1 Odległość:
```
d3 = sqrt((0.50 - 0.50)² + (0.62 - 0.45)²)
   = sqrt(0 + 0.17²)
   = 0.17 m
```

#### 2.3.2 LOS/NLOS:
**Ray:** pionowy (0.50, 0.62) → (0.50, 0.45)
- Oba punkty mają x = 0.50 (środek cylindra)
- Sensor 3 też jest w cylindrze (chest) → **NLOS** (ale bardzo blisko)

#### 2.3.3 Path Loss:
```
PL = 48.4 + 59 × log10(0.17 / 1.0)
   = 48.4 + 59 × (-0.770)
   = 48.4 - 45.4
   = 3.0 dB

# Z shadowingiem (+1.5 dB)
PL_actual = 4.5 dB ≈ 5 dB
```

#### 2.3.4 Energia:
```
packet_size = 20 bits (Temperature)

E_TX3 = 50e-9 × 20 + 10e-12 × 20 × (0.17)²
      = 1.0e-6 + 5.78e-11
      ≈ 1.0 μJ
```

---

## KROK 3: Agregacja

### 3.1 Całkowita Energia:
```
E_total = E_TX1 + E_TX2 + E_TX3 + E_RX1 + E_RX2 + E_RX3
        = 5.0 μJ + 2.5 μJ + 1.0 μJ + (50e-9 × 100) + (50e-9 × 50) + (50e-9 × 20)
        = 8.5 μJ + 5.0 μJ + 2.5 μJ + 1.0 μJ
        = 17.0 μJ
        = 0.017 mJ
```

### 3.2 Kara za Niezawodność:
```
link_margins = [+65 dB, +52 dB, ~+70 dB]  ← wszystkie dodatnie
min_margin = +52 dB

Since min_margin > 0:
    P_rel = 0  ← Brak kary, wszystkie łącza OK ✅
```

### 3.3 Fitness:
```
F(g) = w_E × E_total + w_R × P_rel + P_geo
     = 0.5 × 0.017 mJ + 0.5 × 0 + 0
     = 0.0085 mJ
     = 0.0085
```

**Wynik końcowy:** **F = 0.0085** (niska wartość = dobre rozwiązanie)

---

## ========================================================================
## PRZYKŁAD 2: Rozwiązanie Niedopuszczalne (Invalid)
## ========================================================================

### Genotyp:
```
g_invalid = [0.70, 0.65,  ← Sensor 1 (ECG) POZA strefą 'chest' [0.35-0.65, ...]
             0.18, 0.50,
             0.50, 0.62,
             0.50, 0.45]
```

### Walidacja:
**Sensor 1:** (0.70, 0.65)
- Assigned zone: 'chest' → x ∈ [0.35, 0.65]
- Check: 0.35 ≤ **0.70** ≤ 0.65 ❌  ← NARUSZENIE!

**P_geo = 1e6** (ogromna kara)

### Fitness:
```
F(g_invalid) = 1e6 + 1e6 = 1,000,000
```

**Wynik:** Algorytm odrzuci to rozwiązanie (zbyt wysoka fitness).

---

## ========================================================================
## PRZYKŁAD 3: Rozwiązanie z Niskim Marginesem (Penalty)
## ========================================================================

### Scenariusz:
- **P_TX_max = -10 dBm** (bardzo niska moc, jak w S3)
- Sensor daleko od Hub → wysokie PL

### Załóżmy:
```
d = 0.8 m (duża odległość)
LOS status: NLOS
PL = 48.4 + 59 × log10(0.8 / 1.0) + 5 (shadowing)
   = 48.4 + 59 × (-0.097) + 5
   = 48.4 - 5.7 + 5
   = 47.7 dB

P_TX_required = -85 + 47.7 + 10 = -27.3 dBm
```

### Margines:
```
M = P_TX_max - P_TX_required
  = -10 dBm - (-27.3 dBm)
  = +17.3 dB  ← Jeszcze OK
```

**Ale załóżmy bardziej ekstremalny przypadek:**
```
d = 1.5 m (bardzo daleko, np. sensor na nodze, hub na ramieniu)
PL = 48.4 + 59 × log10(1.5) + 5
   = 48.4 + 59 × 0.176 + 5
   = 48.4 + 10.4 + 5
   = 63.8 dB

P_TX_required = -85 + 63.8 + 10 = -11.2 dBm

M = -10 - (-11.2) = +1.2 dB  ← Bardzo mały margines (marginal link)
```

### Jeszcze gorszy przypadek:
```
d = 2.0 m + wysokie shadowing (+8 dB)
PL = 48.4 + 59 × log10(2.0) + 8
   = 48.4 + 17.8 + 8
   = 74.2 dB

P_TX_required = -85 + 74.2 + 10 = -0.8 dBm

M = -10 - (-0.8) = -9.2 dB  ← UJEMNY! Łącze niemożliwe ❌
```

### Kara:
```
P_rel = |M| × 100
      = 9.2 × 100
      = 920
```

### Fitness:
```
Załóżmy E_total = 0.05 mJ

F(g) = 0.5 × 0.05 + 0.5 × 920
     = 0.025 + 460
     = 460.025
```

**Wynik:** Rozwiązanie prawdopodobnie odrzucone (zbyt wysoka kara).

---

## ========================================================================
## WNIOSKI - CO TO POKAZUJE
## ========================================================================

### 1. **Walidacja Geometryczna**
- Kluczowa dla eliminacji niedopuszczalnych rozwiązań
- P_geo = 1e6 natychmiast dyskwalifikuje genotyp

### 2. **Energia vs. Odległość**
- Krótkie odległości (0.17-0.32 m): E_TX ~ 1-5 μJ
- Elektronika dominuje (E_elec >> E_amp dla małych d)

### 3. **LOS/NLOS Impact**
- NLOS: PL_d0 = 48.4 dB vs. LOS: 35.2 dB
- Różnica: **13 dB** = ~20× więcej mocy potrzebnej!

### 4. **Wagi w Funkcji Celu**
- w_E = 0.7: Priorytet dla niskiej energii (akceptuje gorsze marginesy)
- w_R = 0.7: Priorytet dla niezawodności (akceptuje wyższą energię, ale żąda M > 0)

### 5. **Inverse Power Scheme**
- S3 (25 sensors, -10 dBm): Sensory muszą być BLISKO Hub (d < 0.5 m)
- S1 (6 sensors, 0 dBm): Sensory mogą być dalej (d < 1.5 m)
- To wymusza różne topologie!

---

## JAK UŻYĆ TEGO DO WERYFIKACJI IMPLEMENTACJI

```python
# test_fitness_function.py

def test_valid_solution():
    """Test przypadku z Przykładu 1"""
    genome = np.array([0.45, 0.65, 0.18, 0.50, 0.50, 0.62, 0.50, 0.45])
    
    fitness = fitness_function(genome, config, weights={'w_E': 0.5, 'w_R': 0.5})
    
    # Expected: ~0.008-0.020 (zależnie od shadowing)
    assert 0.005 < fitness < 0.030, f"Expected ~0.0085, got {fitness}"

def test_invalid_solution():
    """Test przypadku z Przykładu 2"""
    genome = np.array([0.70, 0.65, 0.18, 0.50, 0.50, 0.62, 0.50, 0.45])
    
    fitness = fitness_function(genome, config, weights={'w_E': 0.5, 'w_R': 0.5})
    
    # Expected: >> 1e5 (ogromna kara)
    assert fitness > 1e5, f"Expected penalty, got {fitness}"

def test_low_margin_solution():
    """Test przypadku z Przykładu 3"""
    # Symuluj scenariusz z dużymi odległościami i niską mocą
    # ... (wymaga pełnej implementacji)
```

---

**Koniec przykładów obliczeniowych.**
