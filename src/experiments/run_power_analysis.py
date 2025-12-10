"""
WBAN Optimization - Power Analysis Pipeline
Author: Kamil Piejko
Date: 2024

Pipeline do analizy wpływu mocy transmisji:
- Scenariusz S2 (15 sensors)
- 5 poziomów mocy: {-10, -5, 0, +3, +5} dBm
- Balanced weights
- PSO only
= 5 konfiguracji × 50 runs = 250 eksperymentów
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging
import time
from datetime import datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)

#NOWA FUNCKJA FIX

def proxy_power_run(run_id, config, scenario_name, algorithm_name, weight_variant):
    return run_single_experiment(
        config=config,
        scenario_name=scenario_name,
        algorithm_name=algorithm_name,
        weight_variant=weight_variant,
        run_id=run_id
    )


def run_power_analysis(
    config_path: str = 'config/wban_params.yaml',
    n_runs: int = 50,
    output_dir: str = 'results/power_analysis',
    parallel: bool = False,
    n_jobs: int = 4
) -> pd.DataFrame:
    """
    Uruchamia analizę mocy transmisji.
    
    Args:
        config_path: Ścieżka do konfiguracji
        n_runs: Liczba uruchomień na poziom mocy
        output_dir: Katalog wynikowy
        parallel: Równoległe wykonanie
        n_jobs: Liczba procesów
    
    Returns:
        df_all: DataFrame z wynikami
    """
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.utils.config_loader import load_config
    from src.experiments.single_run import run_single_experiment
    from functools import partial
    import multiprocessing as mp
    
    logger.info("="*60)
    logger.info("POWER ANALYSIS PIPELINE")
    logger.info("="*60)
    logger.info(f"Scenario: S2 (15 sensors)")
    logger.info(f"Algorithm: PSO")
    logger.info(f"Weights: balanced")
    logger.info(f"Power levels: 5")
    logger.info(f"Runs per level: {n_runs}")
    logger.info(f"Total experiments: {5 * n_runs}")
    logger.info(f"Output: {output_dir}")
    logger.info("="*60)
    
    # Wczytaj konfigurację
    config = load_config(config_path)
    
    # Poziomy mocy do przetestowania
    power_levels = [-10, -5, 0, 3, 5]  # dBm
    
    all_results = []
    start_time = time.time()
    
    for i, P_TX_max in enumerate(power_levels, 1):
        logger.info(f"\n[{i}/5] Testing P_TX_max = {P_TX_max} dBm")
        
        # Modyfikuj konfigurację dla tego poziomu mocy
        config_modified = config.copy()
        config_modified['scenarios']['S2']['P_TX_max'] = P_TX_max
        
        # Uruchom runs dla tego poziomu mocy
        if parallel:
            run_func = partial(
                proxy_power_run,
                config=config_modified,
                scenario_name='S2',
                algorithm_name='PSO',
                weight_variant='balanced'
            )
            
            with mp.Pool(processes=n_jobs) as pool:
                results_list = list(tqdm(
                    pool.imap(run_func, range(1, n_runs + 1)),
                    total=n_runs,
                    desc=f"P_TX={P_TX_max} dBm"
                ))
        else:
            results_list = []
            for run_id in tqdm(range(1, n_runs + 1), desc=f"P_TX={P_TX_max} dBm"):
                result = run_single_experiment(
                    config=config_modified,
                    scenario_name='S2',
                    algorithm_name='PSO',
                    weight_variant='balanced',
                    run_id=run_id,
                    seed=run_id
                )
                results_list.append(result)
        
        # Dodaj do wszystkich wyników
        df_power = pd.DataFrame(results_list)
        all_results.append(df_power)
        
        # Zapisz wyniki dla tego poziomu mocy
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_path / f"power_P_TX_{P_TX_max:+d}dBm.csv"
        df_csv = df_power.drop(columns=['convergence_curve', 'best_genome'], errors='ignore')
        df_csv.to_csv(csv_file, index=False)
        logger.info(f"  Saved to {csv_file}")
        
        # Szybkie statystyki
        logger.info(f"  Mean fitness: {df_power['best_fitness'].mean():.6f}")
        logger.info(f"  Mean E_total: {df_power['E_total'].mean()*1000:.4f} mJ")
        logger.info(f"  Mean lifetime: {df_power['network_lifetime'].mean():.0f} rounds")
    
    # Połącz wszystkie wyniki
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Zapisz zbiorczy plik
    all_csv = Path(output_dir) / 'all_power_analysis.csv'
    df_all_csv = df_all.drop(columns=['convergence_curve', 'best_genome'], errors='ignore')
    df_all_csv.to_csv(all_csv, index=False)
    
    elapsed_time = time.time() - start_time
    
    logger.info("\n" + "="*60)
    logger.info("POWER ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Total experiments: {len(df_all)}")
    logger.info(f"Total time: {elapsed_time:.2f}s ({elapsed_time/3600:.2f}h)")
    logger.info(f"Results saved to: {all_csv}")
    logger.info("="*60)
    
    return df_all


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run power analysis for WBAN optimization')
    parser.add_argument('--config', type=str, default='config/wban_params.yaml',
                       help='Path to config file')
    parser.add_argument('--runs', type=int, default=50,
                       help='Number of runs per power level')
    parser.add_argument('--output', type=str, default='results/power_analysis',
                       help='Output directory')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel execution')
    parser.add_argument('--jobs', type=int, default=4,
                       help='Number of parallel jobs')
    parser.add_argument('--test', action='store_true',
                       help='Test mode (only 3 runs per level)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path(args.output) / f'power_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
        logger.info("⚠️  TEST MODE: Running only 3 runs per power level")
    
    # Uruchom pipeline
    df_results = run_power_analysis(
        config_path=args.config,
        n_runs=n_runs,
        output_dir=args.output,
        parallel=args.parallel,
        n_jobs=args.jobs
    )
    
    # Analiza wyników
    print("\n" + "="*60)
    print("POWER ANALYSIS SUMMARY")
    print("="*60)
    
    summary = df_results.groupby('P_TX_max').agg({
        'best_fitness': ['mean', 'std', 'min'],
        'E_total': 'mean',
        'network_lifetime': 'mean',
        'min_link_margin': 'mean',
        'los_ratio': 'mean'
    }).round(6)
    
    print(summary)
    
    # Korelacja
    print("\n" + "="*60)
    print("CORRELATION: P_TX vs Metrics")
    print("="*60)
    
    correlations = df_results[['P_TX_max', 'best_fitness', 'E_total', 
                               'network_lifetime', 'min_link_margin']].corr()['P_TX_max']
    print(correlations)
    
    print("\n✓ Power analysis complete!")
