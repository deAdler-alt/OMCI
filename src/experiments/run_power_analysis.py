"""
WBAN Optimization - Power Sensitivity Analysis
Author: Kamil Piejko
Date: 2024

Analiza wpływu mocy nadawania (P_TX) na wyniki.
Wersja zaktualizowana: Kompatybilność z nowym configiem.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import logging
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import copy

# Import na górze (wymagane dla MP)
from src.experiments.single_run import run_single_experiment

logger = logging.getLogger(__name__)


def proxy_power_run(run_id, config, scenario_name, algorithm_name, weight_variant):
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


def run_power_analysis(
    config_path: str = 'config/wban_params.yaml',
    output_dir: str = 'results/power_analysis',
    parallel: bool = False,
    n_jobs: int = 4,
    test_mode: bool = False
) -> pd.DataFrame:
    """
    Uruchamia analizę wrażliwości na moc nadawania.
    """
    import sys
    from pathlib import Path
    
    # Add project root
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.utils.config_loader import load_config
    
    # Wczytaj config
    config = load_config(config_path)
    
    # Pobierz parametry eksperymentu z sekcji experiments -> power_analysis
    exp_config = config['experiments']['power_analysis']
    
    scenario = exp_config['scenario']
    algorithm = exp_config['algorithm']
    weight_variant = exp_config['weight_variant']
    power_levels = exp_config['P_TX_range']
    n_runs = 3 if test_mode else exp_config['n_runs']
    
    logger.info("="*60)
    logger.info("POWER SENSITIVITY ANALYSIS")
    logger.info("="*60)
    logger.info(f"Scenario: {scenario}")
    logger.info(f"Algorithm: {algorithm}")
    logger.info(f"Power Levels (dBm): {power_levels}")
    logger.info(f"Runs per level: {n_runs}")
    logger.info("="*60)
    
    all_results = []
    
    for i, P_TX in enumerate(power_levels):
        logger.info(f"\n[{i+1}/{len(power_levels)}] Testing P_TX = {P_TX} dBm")
        
        # Modyfikacja konfiguracji w locie
        # Kopiujemy config, żeby nie nadpisać oryginału dla innych iteracji
        config_modified = copy.deepcopy(config)
        config_modified['scenarios'][scenario]['P_TX_max'] = float(P_TX)
        
        # Uruchomienie runs
        if parallel:
            run_func = partial(
                proxy_power_run,
                config=config_modified,
                scenario_name=scenario,
                algorithm_name=algorithm,
                weight_variant=weight_variant
            )
            
            with mp.Pool(processes=n_jobs) as pool:
                results = list(tqdm(
                    pool.imap(run_func, range(1, n_runs + 1)),
                    total=n_runs,
                    desc=f"P_TX={P_TX}"
                ))
        else:
            results = []
            for run_id in tqdm(range(1, n_runs + 1), desc=f"P_TX={P_TX}"):
                res = run_single_experiment(
                    config=config_modified,
                    scenario_name=scenario,
                    algorithm_name=algorithm,
                    weight_variant=weight_variant,
                    run_id=run_id
                )
                results.append(res)
        
        # Dodaj informację o P_TX do wyników
        for res in results:
            res['P_TX_max'] = P_TX
            all_results.append(res)
            
    # Zapisz wyniki
    df_power = pd.DataFrame(all_results)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_path / 'all_power_analysis.csv'
    df_power.to_csv(csv_file, index=False)
    
    logger.info("\n" + "="*60)
    logger.info("POWER ANALYSIS COMPLETE")
    logger.info(f"Saved to: {csv_file}")
    logger.info("="*60)
    
    return df_power


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if args.test:
        logger.info("⚠️  TEST MODE ENABLED")
        
    df = run_power_analysis(
        parallel=args.parallel,
        test_mode=args.test,
        n_jobs=args.jobs
    )
    
    # Podgląd statystyk
    if not df.empty:
        print("\n" + "="*60)
        print("POWER ANALYSIS SUMMARY")
        print("="*60)
        summary = df.groupby('P_TX_max').agg({
            'best_fitness': ['mean', 'std'],
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
        cols = ['P_TX_max', 'best_fitness', 'E_total', 'network_lifetime', 'min_link_margin']
        print(df[cols].corr()['P_TX_max'])
    
    print("\n✓ Power analysis complete!")
