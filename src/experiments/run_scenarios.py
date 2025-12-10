"""
WBAN Optimization - Main Scenarios Pipeline
Author: Kamil Piejko
Date: 2024

Pipeline do uruchomienia głównych eksperymentów.
Wersja zaktualizowana: Dynamiczne wczytywanie wariantów wag.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging
import time
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Import na górze pliku (niezbędne dla multiprocessing)
from src.experiments.single_run import run_single_experiment

logger = logging.getLogger(__name__)


def proxy_single_run(run_id, config, scenario_name, algorithm_name, weight_variant):
    """
    Wrapper dla multiprocessingu.
    """
    return run_single_experiment(
        config=config,
        scenario_name=scenario_name,
        algorithm_name=algorithm_name,
        weight_variant=weight_variant,
        run_id=run_id
    )


def run_single_config(
    config: Dict,
    scenario_name: str,
    algorithm_name: str,
    weight_variant: str,
    n_runs: int = 50,
    output_dir: str = 'results/main_scenarios',
    parallel: bool = False,
    n_jobs: int = 4
) -> pd.DataFrame:
    """
    Uruchamia wszystkie runs dla jednej konfiguracji.
    """
    config_name = f"{scenario_name}_{algorithm_name}_{weight_variant}"
    logger.info(f"Running configuration: {config_name} ({n_runs} runs)")
    
    results_list = []
    
    if parallel:
        # Równoległe wykonanie
        run_func = partial(
            proxy_single_run,
            config=config,
            scenario_name=scenario_name,
            algorithm_name=algorithm_name,
            weight_variant=weight_variant
        )
        
        with mp.Pool(processes=n_jobs) as pool:
            # imap jest leniwy, więc list() wymusza wykonanie
            results_list = list(tqdm(
                pool.imap(run_func, range(1, n_runs + 1)),
                total=n_runs,
                desc=f"{config_name}"
            ))
    else:
        # Sekwencyjne wykonanie
        for run_id in tqdm(range(1, n_runs + 1), desc=f"{config_name}"):
            result = run_single_experiment(
                config=config,
                scenario_name=scenario_name,
                algorithm_name=algorithm_name,
                weight_variant=weight_variant,
                run_id=run_id,
                seed=run_id
            )
            results_list.append(result)
    
    # Konwersja do DataFrame
    df_results = pd.DataFrame(results_list)
    
    # Zapisz wyniki (tymczasowe pliki dla bezpieczeństwa)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Nie usuwamy kolumn - zachowujemy dane do wykresów
    csv_file = output_path / f"{config_name}.csv"
    df_results.to_csv(csv_file, index=False)
    
    return df_results


def run_all_scenarios(
    config_path: str = 'config/wban_params.yaml',
    n_runs: int = 50,
    output_dir: str = 'results/main_scenarios',
    parallel: bool = False,
    n_jobs: int = 4
) -> pd.DataFrame:
    """
    Uruchamia wszystkie zdefiniowane scenariusze.
    """
    import sys
    from pathlib import Path
    
    # Add project root to path if needed
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.utils.config_loader import load_config
    
    # Wczytaj konfigurację
    config = load_config(config_path)
    
    # --- DYNAMICZNE POBIERANIE KONFIGURACJI ---
    scenarios = ['S1', 'S2', 'S3']
    algorithms = ['GA', 'PSO']
    
    # Pobieramy klucze z configu, zamiast wpisywać na sztywno
    # Dzięki temu skrypt sam "zauważy", że jest tylko 'balanced'
    weight_variants = list(config['fitness_function']['weight_variants'].keys())
    
    total_configs = len(scenarios) * len(algorithms) * len(weight_variants)
    
    logger.info("="*60)
    logger.info("MAIN SCENARIOS PIPELINE")
    logger.info("="*60)
    logger.info(f"Configurations: {total_configs}")
    logger.info(f"Scenarios: {scenarios}")
    logger.info(f"Variants: {weight_variants}")
    logger.info(f"Runs per config: {n_runs}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*60)
    
    all_results = []
    start_time = time.time()
    
    counter = 1
    for scenario in scenarios:
        for weight in weight_variants:
            for algorithm in algorithms:
                logger.info(f"\n[{counter}/{total_configs}] Running: {scenario}, {algorithm}, {weight}")
                
                try:
                    df_config = run_single_config(
                        config=config,
                        scenario_name=scenario,
                        algorithm_name=algorithm,
                        weight_variant=weight,
                        n_runs=n_runs,
                        output_dir=output_dir,
                        parallel=parallel,
                        n_jobs=n_jobs
                    )
                    all_results.append(df_config)
                except Exception as e:
                    logger.error(f"Failed to run config {scenario}_{algorithm}_{weight}: {e}")
                    # Kontynuuj mimo błędu (żeby nie stracić wszystkiego)
                
                counter += 1
    
    if not all_results:
        logger.error("No results generated!")
        return pd.DataFrame()

    # Połącz wszystkie wyniki
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Zapisz zbiorczy plik (z pełnymi danymi!)
    all_csv = Path(output_dir) / 'all_main_scenarios.csv'
    df_all.to_csv(all_csv, index=False)
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Total experiments: {len(df_all)}")
    logger.info(f"Total time: {elapsed_time:.2f}s ({elapsed_time/3600:.2f}h)")
    logger.info(f"Results saved to: {all_csv}")
    logger.info("="*60)
    
    return df_all


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run main WBAN optimization scenarios')
    parser.add_argument('--config', type=str, default='config/wban_params.yaml',
                       help='Path to config file')
    parser.add_argument('--runs', type=int, default=50,
                       help='Number of runs per configuration')
    parser.add_argument('--output', type=str, default='results/main_scenarios',
                       help='Output directory')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel execution')
    parser.add_argument('--jobs', type=int, default=4,
                       help='Number of parallel jobs')
    parser.add_argument('--test', action='store_true',
                       help='Test mode (only 3 runs per config)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path(args.output) / f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Test mode
    n_runs = 3 if args.test else args.runs
    
    if args.test:
        logger.info("⚠️  TEST MODE: Running only 3 runs per configuration")
    
    # Uruchom pipeline
    df_results = run_all_scenarios(
        config_path=args.config,
        n_runs=n_runs,
        output_dir=args.output,
        parallel=args.parallel,
        n_jobs=args.jobs
    )
    
    # Podstawowe statystyki (jeśli są wyniki)
    if not df_results.empty:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        try:
            summary = df_results.groupby(['scenario', 'algorithm', 'weight_variant']).agg({
                'best_fitness': ['mean', 'std', 'min', 'max'],
                'network_lifetime': ['mean'],
                'E_total': 'mean',
                'elapsed_time': 'sum'
            }).round(6)
            
            print(summary)
        except Exception as e:
            print(f"Could not generate summary table: {e}")
    
    print("\n✓ Pipeline complete!")