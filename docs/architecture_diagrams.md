# WBAN Optimization - System Architecture Diagrams

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph Input
        CONFIG[config/wban_params.yaml]
        MATRIX[docs/experiment_matrix.csv]
    end
    
    subgraph Core["Core System"]
        BODY[Body Model<br/>Zones & Constraints]
        SENSOR[Sensor Model<br/>ECG, SpO2, etc.]
        GENO[Genotype<br/>Encode/Decode]
    end
    
    subgraph Models["Physical Models"]
        ENERGY[Energy Model<br/>First Order Radio]
        PROP[Propagation Model<br/>IEEE 802.15.6 CM3]
        LOS[LOS/NLOS Detector<br/>Ray-Cylinder Intersection]
    end
    
    subgraph Optimization["Optimization Layer"]
        FITNESS[Fitness Function<br/>F = w_E*E + w_R*P]
        GA[Genetic Algorithm<br/>Mealpy]
        PSO[Particle Swarm<br/>Mealpy]
    end
    
    subgraph Baselines["Baseline Methods"]
        RANDOM[Random Placement]
        NAIVE[Naive Centroid]
    end
    
    subgraph Experiments["Experiment Pipeline"]
        MAIN[Main Experiment<br/>18 configs × 50 runs]
        POWER[Power Analysis<br/>5 configs × 50 runs]
        COLLECT[Metrics Collector]
    end
    
    subgraph Output["Results & Visualization"]
        RESULTS[(CSV Results)]
        PLOTS[Plots Generator<br/>7+ visualizations]
        STATS[Statistical Analysis<br/>Wilcoxon test]
    end
    
    CONFIG --> BODY
    CONFIG --> ENERGY
    CONFIG --> PROP
    MATRIX --> MAIN
    MATRIX --> POWER
    
    BODY --> GENO
    SENSOR --> GENO
    GENO --> FITNESS
    
    ENERGY --> FITNESS
    PROP --> FITNESS
    LOS --> PROP
    
    FITNESS --> GA
    FITNESS --> PSO
    FITNESS --> RANDOM
    FITNESS --> NAIVE
    
    GA --> MAIN
    PSO --> MAIN
    PSO --> POWER
    RANDOM --> MAIN
    NAIVE --> MAIN
    
    MAIN --> COLLECT
    POWER --> COLLECT
    COLLECT --> RESULTS
    RESULTS --> PLOTS
    RESULTS --> STATS
    
    style CONFIG fill:#e1f5fe
    style RESULTS fill:#c8e6c9
    style FITNESS fill:#fff9c4
    style GA fill:#ffccbc
    style PSO fill:#ffccbc
```

## 2. Fitness Function Detailed Flow

```mermaid
graph TD
    START([Genotype g]) --> DECODE[Decode to<br/>Sensors + Hub]
    
    DECODE --> VALID{Geometric<br/>Validation}
    
    VALID -->|Invalid| PENALTY_GEO[P_geo = 1e6<br/>RETURN]
    VALID -->|Valid| LOOP[For Each Sensor]
    
    LOOP --> DIST[Compute Distance<br/>d = ||sensor - hub||]
    DIST --> LOS_DET[Detect LOS/NLOS<br/>Ray-Cylinder Test]
    
    LOS_DET --> PL[Compute Path Loss<br/>PL = PL_d0 + 10*n*log10(d/d0)]
    
    PL --> PTX[Required TX Power<br/>P_TX_req = P_sens + PL + M_safe]
    
    PTX --> MARGIN[Link Margin<br/>M = P_TX_max - P_TX_req]
    
    MARGIN --> ENERGY[Transmission Energy<br/>E_TX = E_elec + E_amp*d^α]
    
    ENERGY --> ACCUM[Accumulate<br/>E_total += E_TX]
    
    ACCUM --> MORE{More<br/>Sensors?}
    MORE -->|Yes| LOOP
    More -->|No| CHECK_REL{Min Margin<br/>< 0?}
    
    CHECK_REL -->|Yes| PENALTY_REL[P_rel = |M_min| × 100]
    CHECK_REL -->|No| NO_PENALTY[P_rel = 0]
    
    PENALTY_REL --> AGGREGATE
    NO_PENALTY --> AGGREGATE
    
    AGGREGATE[F = w_E × E_total + w_R × P_rel]
    
    AGGREGATE --> RETURN([Return Fitness F])
    
    style START fill:#e1f5fe
    style RETURN fill:#c8e6c9
    style PENALTY_GEO fill:#ffcdd2
    style PENALTY_REL fill:#ffcdd2
    style AGGREGATE fill:#fff9c4
```

## 3. LOS/NLOS Detection Algorithm

```mermaid
graph TD
    START([Sensor Pos<br/>Hub Pos]) --> PARAMS[Load Torso Params<br/>center_x, center_y, R]
    
    PARAMS --> RAY[Compute Ray<br/>origin = sensor<br/>direction = hub - sensor]
    
    RAY --> TEST1{Sensor or Hub<br/>inside cylinder?}
    
    TEST1 -->|Yes| NLOS1[RETURN: NLOS]
    TEST1 -->|No| DIST_CALC[Calculate Distance<br/>from ray to<br/>cylinder center]
    
    DIST_CALC --> TEST2{Distance < R?}
    
    TEST2 -->|No| LOS1[RETURN: LOS]
    TEST2 -->|Yes| TTEST[Find closest point<br/>on ray to center]
    
    TTEST --> TEST3{t_closest in<br/>[0, ray_length]?}
    
    TEST3 -->|No| LOS2[RETURN: LOS]
    TEST3 -->|Yes| NLOS2[RETURN: NLOS]
    
    style START fill:#e1f5fe
    style NLOS1 fill:#ffcdd2
    style NLOS2 fill:#ffcdd2
    style LOS1 fill:#c8e6c9
    style LOS2 fill:#c8e6c9
```

## 4. Experiment Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant Script as run_scenarios.py
    participant Config as YAML Config
    participant Optimizer as GA/PSO
    participant Fitness as Fitness Function
    participant Results as CSV Output
    
    User->>Script: python run_scenarios.py
    Script->>Config: Load wban_params.yaml
    Config-->>Script: Parameters
    
    Script->>Script: Generate experiment matrix<br/>(18 configs × 50 runs)
    
    loop For each configuration
        Script->>Optimizer: Run optimization<br/>(scenario, weights, algorithm)
        
        loop For 100 iterations
            Optimizer->>Fitness: Evaluate population
            
            loop For each genotype
                Fitness->>Fitness: Decode → Validate → Compute
                Fitness-->>Optimizer: Fitness value
            end
            
            Optimizer->>Optimizer: Selection → Crossover → Mutation
        end
        
        Optimizer-->>Script: Best solution + history
        Script->>Script: Extract metrics<br/>(E_total, T_life, M_min, convergence)
    end
    
    Script->>Results: Save all results to CSV
    Results-->>User: results/experiments/main_experiment_results.csv
    
    User->>Script: python visualization/generate_plots.py
    Script->>Results: Load CSV
    Script->>Script: Generate 7+ plots
    Script-->>User: results/plots/*.png
```

## 5. Data Model Class Diagram

```mermaid
classDiagram
    class Sensor {
        +int id
        +str type
        +ndarray position
        +str assigned_zone
        +float data_rate
        +int packet_size
        +float energy_remaining
        +float energy_initial
        +int transmitted_packets
        +float energy_consumed
        +is_alive() bool
        +transmit_packet() void
    }
    
    class Hub {
        +ndarray position
        +str zone
        +bool energy_unlimited
        +int received_packets
        +float total_throughput
        +receive_packet() void
    }
    
    class Genotype {
        +int n_sensors
        +int dimension
        +list bounds
        +decode(genome, config) tuple
        +is_valid_position(pos, zone) bool
        +validate_geometry(sensors, hub) float
    }
    
    class EnergyModel {
        +float E_elec_TX
        +float E_elec_RX
        +float E_amp_fs
        +float E_amp_mp
        +float d_threshold
        +compute_transmission_energy() float
        +compute_reception_energy() float
    }
    
    class PropagationModel {
        +dict LOS_params
        +dict NLOS_params
        +float d0
        +float P_sens
        +float M_safe
        +compute_path_loss() float
        +compute_required_power() float
    }
    
    class LOSDetector {
        +dict torso_params
        +detect_LOS_NLOS() str
        +ray_cylinder_intersection() bool
    }
    
    class FitnessFunction {
        +dict weights
        +dict config
        +evaluate(genome) float
        +compute_total_energy() float
        +compute_reliability_penalty() float
    }
    
    class GAOptimizer {
        +int population_size
        +int max_iterations
        +float p_crossover
        +float p_mutation
        +solve(problem) Solution
    }
    
    class PSOOptimizer {
        +int population_size
        +int max_iterations
        +float w
        +float c1
        +float c2
        +solve(problem) Solution
    }
    
    Genotype --> Sensor : decodes to
    Genotype --> Hub : decodes to
    FitnessFunction --> Sensor : evaluates
    FitnessFunction --> Hub : evaluates
    FitnessFunction --> EnergyModel : uses
    FitnessFunction --> PropagationModel : uses
    PropagationModel --> LOSDetector : uses
    GAOptimizer --> FitnessFunction : optimizes
    PSOOptimizer --> FitnessFunction : optimizes
```

## 6. Visualization Pipeline

```mermaid
graph LR
    subgraph Input
        CSV[CSV Results<br/>main_experiment.csv]
        CONFIG[YAML Config]
    end
    
    subgraph Processing
        LOAD[Load & Parse<br/>Pandas DataFrame]
        GROUP[Group by<br/>Scenario, Algorithm]
        STATS[Compute Statistics<br/>Mean, Std, Median]
    end
    
    subgraph Plots["7 Visualizations"]
        PLOT1[Energy vs Sensors<br/>Line plot + error bars]
        PLOT2[Lifetime vs Sensors<br/>Line plot + error bars]
        PLOT3[Convergence Curves<br/>Line plot + ribbon]
        PLOT4[Placement Map<br/>Scatter + links]
        PLOT5[Energy Distribution<br/>Bar chart]
        PLOT6[Delay vs Sensors<br/>Line plot]
        PLOT7[Power Sensitivity<br/>Box plot]
    end
    
    subgraph Output
        PNG[PNG Files<br/>results/plots/]
        PDF[PDF Files<br/>for thesis]
    end
    
    CSV --> LOAD
    CONFIG --> LOAD
    LOAD --> GROUP
    GROUP --> STATS
    
    STATS --> PLOT1
    STATS --> PLOT2
    STATS --> PLOT3
    STATS --> PLOT4
    STATS --> PLOT5
    STATS --> PLOT6
    STATS --> PLOT7
    
    PLOT1 --> PNG
    PLOT2 --> PNG
    PLOT3 --> PNG
    PLOT4 --> PNG
    PLOT5 --> PNG
    PLOT6 --> PNG
    PLOT7 --> PNG
    
    PNG --> PDF
    
    style CSV fill:#e1f5fe
    style PNG fill:#c8e6c9
    style PDF fill:#c8e6c9
```

## 7. Deployment Timeline (Gantt Chart)

```mermaid
gantt
    title WBAN Optimization - Implementation & Execution Timeline
    dateFormat  YYYY-MM-DD
    section Phase 1: Core
    Body Model           :done, p1a, 2024-12-08, 1d
    Energy Model         :done, p1b, 2024-12-08, 1d
    Propagation Model    :done, p1c, 2024-12-08, 1d
    LOS Detector         :done, p1d, 2024-12-08, 1d
    
    section Phase 2: Optimization
    Fitness Function     :active, p2a, 2024-12-09, 1d
    GA Optimizer         :p2b, after p2a, 1d
    PSO Optimizer        :p2c, after p2a, 1d
    Baselines            :p2d, after p2a, 1d
    
    section Phase 3: Experiments
    Main Pipeline        :p3a, after p2d, 1d
    Power Analysis       :p3b, after p3a, 1d
    Unit Tests           :p3c, after p2d, 1d
    
    section Phase 4: Execution
    Test Run (10 runs)   :p4a, after p3c, 1d
    Full Experiment      :crit, p4b, after p4a, 2d
    
    section Phase 5: Analysis
    Visualization        :p5a, after p4b, 1d
    Statistical Tests    :p5b, after p4b, 1d
    LaTeX Extensions     :p5c, after p5b, 1d
```

---

## How to Use These Diagrams

### In Thesis (LaTeX):

1. **Architecture Diagram** → Rozdział 4.1 (Architektura systemu)
2. **Fitness Flow** → Rozdział 2.3.3 (Funkcja celu - szczegóły)
3. **LOS/NLOS Algorithm** → Rozdział 2.2.2 (Detekcja przesłonięcia)
4. **Class Diagram** → Rozdział 4.2 (Implementacja - struktura klas)

### Rendering Diagrams:

```bash
# Online: https://mermaid.live/
# Paste diagram code → Export as PNG/SVG

# Local (if you have mermaid-cli):
mmdc -i diagrams.md -o diagram1.png
```

### Integration with LaTeX:

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/system_architecture.png}
\caption{Architektura systemu optymalizacji WBAN. Pokazano główne moduły: model ciała, modele fizyczne (energia, propagacja), algorytmy optymalizacyjne (GA, PSO) oraz pipeline eksperymentów.}
\label{fig:architecture}
\end{figure}
```

---

**Note:** Te diagramy można również wygenerować programowo w Python używając biblioteki `diagrams`:

```python
from diagrams import Diagram, Cluster
from diagrams.generic.compute import Rack
from diagrams.programming.flowchart import Decision

# Diagram architecture
with Diagram("WBAN System", show=False):
    # ... (similar structure)
```
